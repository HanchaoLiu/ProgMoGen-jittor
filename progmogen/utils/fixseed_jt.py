import numpy as np
import random
import jittor as jt


def fixseed(seed):

    random.seed(seed)
    np.random.seed(seed)
    jt.set_seed(seed)

    try:
        import cupy
        cupy.random.seed(seed)
    except:
        pass



# SEED = 10
# EVALSEED = 0
# # Provoc warning: not fully functionnal yet
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# fixseed(SEED)
