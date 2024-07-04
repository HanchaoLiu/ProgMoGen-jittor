import numpy as np 
import jittor as jt 


def main_clip_weight():

    # change the path to your own
    torch_weight_path = '/path/to/.cache/clip/ViT-B-32.pt'
    npy_weight_path   = './save_jittor/ViT-B-32.npy'
    jt_weight_path    = './save_jittor/ViT-B-32.pkl'


    # first step 
    if True:
        import torch
        clip = torch.load(torch_weight_path).state_dict()

        for k in clip.keys():
            clip[k] = clip[k].float().cpu().numpy()
        np.save(npy_weight_path, dict(clip))

    #  second step 
    if True:
        import jittor as jt
        clip = np.load(npy_weight_path, allow_pickle=True).item()

        jt.save(clip, jt_weight_path)



    
def main_mdm_weight():

    torch_weight_path = './save/humanml_trans_enc_512/model000475000.pt'
    npy_weight_path   = './save_jittor/model000475000.npy'
    jt_weight_path    = './save_jittor/model000475000.pkl'

    # first step 
    if True:
        import torch
        clip = torch.load(torch_weight_path, map_location='cpu')

        for k in clip.keys():
            clip[k] = clip[k].float().detach().cpu().numpy()
        np.save(npy_weight_path, dict(clip))

    #  second step 
    if True:
        import jittor as jt
        clip = np.load(npy_weight_path, allow_pickle=True).item()

        jt.save(clip, jt_weight_path)



    


if __name__ == "__main__":
    pass
    # main_clip_weight()
    # main_mdm_weight()
