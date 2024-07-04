# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# import torch
import numpy as np
import jittor as jt 

_EPS4 = np.finfo(float).eps * 4.0

_FLOAT_EPS = np.finfo(np.float).eps




# jittor-based 
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    # uv = torch.cross(qvec, v, dim=1)
    # uuv = torch.cross(qvec, uv, dim=1)
    uv = jt.cross(qvec, v, dim=1)
    uuv = jt.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)



def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    # mask = torch.ones_like(q)
    mask = jt.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


# PyTorch-backed implementations

