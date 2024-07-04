from os.path import join as pjoin

import numpy as np
import os
# from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.common.quaternion_jt import *
# from data_loaders.humanml.utils.paramUtil import *

import jittor as jt 

from tqdm import tqdm




# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = jt.zeros_like(rot_vel) #.to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = jt.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = jt.zeros(data.shape[:-1] + (4,)) #.to(data.device)
    r_rot_quat[..., 0] = jt.cos(r_rot_ang)
    r_rot_quat[..., 2] = jt.sin(r_rot_ang)

    r_pos = jt.zeros(data.shape[:-1] + (3,)) #.to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # print("r_pos.shape = ", r_pos.shape)
    # r_pos = jt.cumsum(r_pos, dim=-2)
    # r_pos.shape =  [1,1,196,3,]
    # jt does not support negative dim (<=-1), so change to positive dim.
    r_pos = jt.cumsum(r_pos, dim=2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos



def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = jt.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions






