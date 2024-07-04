import numpy as np 
import jittor as jt 
import time 

from atomic_lib.math_utils_jt import * 

'''
task: velocity assigned at first, middle and last frames.
'''


TEXT_PROMPT="a man walks forwards and then stops."
TEXT_TOKEN=None
LENGTH=80
DEMO_NUM=1

# optimizer params
lr=0.005
# iterations=80
iterations=100
decay_steps=None


def get_velocity_gt(device):
    v0 = jt.array([0.0,0.0,0.05])
    v1 = jt.array([0.05,0.0,0.0])
    v2 = jt.array([0.0,0.0,-0.05])
    return [v0,v1,v2]


def f_loss(self, sample, sample_0, steps):

    joints = self.sample_to_joints(sample)
    v0_gt, v1_gt, v2_gt = get_velocity_gt(None)

    length = self.length 
    # define t and joint
    t_all = length - 2
    t_0 = 0
    t_middle = (length-1)//2

    joint_control = pelvis

    v0_pred = keyframe(get_joint(get_velocity(joints), joint_control),t_0).squeeze()
    v1_pred = keyframe(get_joint(get_velocity(joints), joint_control),t_middle).squeeze()
    v2_pred = keyframe(get_joint(get_velocity(joints), joint_control),t_all).squeeze()
    
    loss = equal_sum(v0_pred, v0_gt) + \
                equal_sum(v1_pred, v1_gt) + \
                equal_sum(v2_pred, v2_gt)
    loss = loss / 3
    return loss 


def f_eval(self, sample, sample_0):

    joints = self.sample_to_joints(sample)
    v0_gt, v1_gt, v2_gt = get_velocity_gt(None)

    length = self.length 
    # define t and joint
    t_all = length - 2
    t_0 = 0
    t_middle = (length-1)//2

    joint_control = pelvis

    v0_pred = keyframe(get_joint(get_velocity(joints), joint_control),t_0).squeeze()
    v1_pred = keyframe(get_joint(get_velocity(joints), joint_control),t_middle).squeeze()
    v2_pred = keyframe(get_joint(get_velocity(joints), joint_control),t_all).squeeze()
    
    loss = equal_sum(v0_pred, v0_gt) + \
                equal_sum(v1_pred, v1_gt) + \
                equal_sum(v2_pred, v2_gt)
    loss = loss / 3
    loss = loss.sqrt()
    return loss 










