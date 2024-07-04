import numpy as np 
import jittor as jt 

from IPython import embed 
import argparse

from .mha import MultiheadAttention
from .mdm_naive_jt import MDM



def main_check_mha():

    # load data for check 
    file_name = "./tmp/data_check_attn.npy"
    res_dt = np.load(file_name, allow_pickle=True).item()
    x0 = res_dt["x0"]
    x1 = res_dt["x1"]
    weights = res_dt["weights"]
    for k, v in weights.items():
        print(k,v.shape)

    torch_state_dict = {k: jt.array(v) for k,v in weights.items()}


    jt.flags.use_cuda = 1 

    # d_model=512, nhead=4, dropout=0.1, batch_first=False, factory_kwargs={'device': None, 'dtype': None}
    self_attn = MultiheadAttention(embed_dim=512, num_heads=4, batch_first=False)


    self_attn.load_state_dict(torch_state_dict)

    self_attn.eval()

    # x=torch.Size([197, 1, 512]),attn_mask=None,key_padding_mask=None
    attn_mask = None 
    key_padding_mask = None
    # x = np.zeros(shape=(197,1,512),dtype=np.float32)
    x = jt.array(x0)
    
    with jt.no_grad():
        res  = self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        res = res.numpy()
    print(res.shape, x1.shape)
    diff = np.abs(res-x1).max()
    print("diff = ", diff)

    keys = list(dict(self_attn.named_parameters()).keys())
    print(keys)

    for k,v in self_attn.named_parameters():
        print(k,  v.shape)

    from IPython import embed 
    embed()



def main():
     
    # modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
    #              latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
    #              ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
    #             #  arch='trans_enc', emb_trans_dec=False, clip_version=None
    # 263 1 1 True rot6d True True 512 1024 8 4 0.1 None gelu False hml_vec humanml 512 trans_enc False ViT-B/32


    kwargs = {'cond_mode': 'text', 'cond_mask_prob': 0.1, 'action_emb': 'tensor', 'diffusion-steps': 1000, 'batch_size': 32, 'use_tta': False, 
              'trans_emb': False, 'concat_trans_emb': False, 
              'args': argparse.Namespace(arch='trans_enc', batch_size=32, concat_trans_emb=False, cond_mask_prob=0.1, cropping_sampler=False, 
                                cuda=True, data_dir='', dataset='humanml', device=0, diffusion_steps=1000, emb_trans_dec=False, 
                                eval_mode='debug', filter_noise=True, guidance_param=2.5, inpainting_mask='root_horizontal', 
                                lambda_fc=0.0, lambda_rcxyz=0.0, lambda_vel=0.0, latent_dim=512, layers=8, mask_type='root_horizontal', 
                                model_path='./save/humanml_trans_enc_512/model000475000.pt', noise_schedule='cosine', num_samples_limit=32, 
                                num_unfoldings=1, overwrite=False, replication_times=None, ret_type='pos', save_fig_dir='result/demo/or3_n32/ours_pos_npy', 
                                save_tag='or3', seed=10, short_db=False, sigma_small=True, task_config='task_or_config', text_split='test_plane_v0_id', 
                                trans_emb=False, transition_margins=0, use_ddim_tag=1, use_tta=False)}
    
    mdm = MDM(modeltype='', njoints=263, nfeats=1, num_actions=1, translation=True, pose_rep='rot6d', glob=True, glob_rot=True,
                 latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation='gelu', legacy=False, data_rep='hml_vec', dataset='humanml', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version='./tmp/ViT-B-32.pkl', **kwargs)
    
    mdm.load_state_dict(jt.load('./tmp/model000475000.pkl'))
    mdm.eval()
    for p in mdm.parameters():
        p.requires_grad=False

    check_mdm_data = np.load("./tmp/data_check_mdm.npy", allow_pickle=True).item()
    print(check_mdm_data.keys())
    x = check_mdm_data['x']
    timesteps = check_mdm_data['timesteps']
    output = check_mdm_data['output']

    x_jt = jt.array(x)
    timesteps_jt = jt.array(timesteps)
    y = {'text': ['a man is walking.']}

    res_jt = mdm(x_jt, timesteps_jt, y)
    res_jt = res_jt.numpy()
    print(res_jt.shape)

    print(output.shape)

    diff = np.abs(res_jt - output).max()
    print(diff)

    # embed()
    # weights = mdm.named_parameters()
    # for k, v in weights:
    #     print(k, v.shape)






    


if __name__ == "__main__":

    # main_check_mha()
    main()
