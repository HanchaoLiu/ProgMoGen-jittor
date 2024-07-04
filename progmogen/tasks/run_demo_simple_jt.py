
import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs"))

import numpy as np 
import jittor as jt 

from datetime import datetime
from collections import OrderedDict
import copy
from IPython import embed 
import types

from utils.parser_util import evaluation_inpainting_parser, evaluation_inpainting_parser_add_args
from utils.fixseed_jt import fixseed

from utils.model_util_v2_jt import load_model_blending_and_diffusion
from data_loaders.humanml.scripts.motion_process_jt import recover_from_ric

from diffusion import logger

from config_data import MODEL_PATH, ABS_BASE_PATH, ROOT_DIR

# torch.multiprocessing.set_sharing_strategy('file_system')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



####################################################################
# args
####################################################################

def f_add_args(parser):
    parser.add_argument("--use_ddim_tag", default=0, type=int, help="")
    parser.add_argument("--save_tag", default='', type=str, help="")
    parser.add_argument("--mask_type", default='', type=str, help="")
    parser.add_argument("--save_fig_dir", default='', type=str, help="")
    parser.add_argument("--ret_type", default='', type=str, help="from pos/rot")
    parser.add_argument("--text_split", default='', type=str, help="test text split for evaluation")
    parser.add_argument("--num_samples_limit", default=0, type=int, help="")

    parser.add_argument("--task_config", default='none', type=str, help="task configs")
    return parser






####################################################################
# get gt / generated motions
####################################################################

def get_gen_motion(args, model, dataloader, num_samples_limit, scale, init_motion_type):

    clip_denoised = False  # FIXME - hardcoded
    # self.max_motion_length = max_motion_length
    # sample_fn = (
    #     diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    # )
    # real_num_batches = len(dataloader)
    # if num_samples_limit is not None:
    #     real_num_batches = num_samples_limit // dataloader.batch_size + 1
    batch_size = 32
    real_num_batches = num_samples_limit // batch_size + 1
    print('real_num_batches', real_num_batches)

    generated_motion = []
    loss_list = []
    length_list = []
    text_list = []
    constraint_list = []

    caption_list = []
    tokens_list =  []
    cap_len_list = []


    model.eval()

    for v in model.parameters():
        v.requires_grad=False

    # here has a grad!!!
    # with torch.no_grad():

    # for _ in range(real_num_batches//len(dataloader)):
    for _ in range(1):
        # for i, (motion, model_kwargs) in enumerate(dataloader):
        for i in range(1):

            model_kwargs = {'y': {}}
            motion = jt.zeros(batch_size,263,1,196).float()

            # print("len(generated_motion)=", len(generated_motion))

            if num_samples_limit is not None and len(generated_motion) >= real_num_batches:
                break

            text_prompt = import_class(f"{args.task_config}.TEXT_PROMPT")
            motion_length = import_class(f"{args.task_config}.LENGTH")
            text_token = import_class(f"{args.task_config}.TEXT_TOKEN")
            
            model_kwargs['y']['text'] = [text_prompt]*batch_size
            # model_kwargs['y']['tokens'] = [None]*32
            model_kwargs['y']['tokens'] = [text_token]*batch_size
            # model_kwargs['y']['lengths'] = jt.LongTensor([motion_length]*batch_size)
            model_kwargs['y']['lengths'] = jt.array([motion_length]*batch_size, dtype=jt.int32)

            
            tokens = [(t.split('_') if t is not None else [] ) for t in model_kwargs['y']['tokens']]


            # add CFG scale to batch
            if scale != 1.:
                model_kwargs['y']['scale'] = jt.ones(motion.shape[0]) * scale
            
            # model_kwargs['y']['inpainted_motion'] = motion.to(dist_util.dev())
            # model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).float().to(dist_util.dev())

            model_kwargs['y']['inpainted_motion'] = jt.zeros(batch_size,263,1,196).float()
            model_kwargs['y']['inpainting_mask'] = jt.zeros(batch_size,263,1,196).float()

            repeat_times=1
            for t in range(repeat_times):

                diffusion.load_inv_normalization_data(None)

                f_loss = import_class(f"{args.task_config}.f_loss")
                f_eval = import_class(f"{args.task_config}.f_eval")
                # ddim_sample_loop_opt_fn=import_class(f"{args.task_config}.ddim_sample_loop_opt_fn")

                diffusion.f_loss = types.MethodType(f_loss, diffusion)
                diffusion.f_eval = types.MethodType(f_eval, diffusion)
                # diffusion.ddim_sample_loop_opt_fn = types.MethodType(ddim_sample_loop_opt_fn, diffusion)
                sample_fn = diffusion.ddim_sample_loop_opt_fn
                # sample_fn = diffusion.ddim_sample_loop_opt_fn_base

                sample = []
                loss = []
                constraint = []
                lengths = model_kwargs['y']['lengths']
                bs = motion.shape[0]

                demo_num = import_class(f"{args.task_config}.DEMO_NUM")
                for ii in range(bs):
                    print(ii,bs)

                    diffusion.np_seed = np.random.randint(0,1000)+1
                    model_kwargs_each = get_slice_model_kwargs(model_kwargs, ii)

                    motion_each = model_kwargs_each['y']['inpainted_motion']
                    length_each = model_kwargs_each['y']['lengths'].item()

                    diffusion.length = length_each

                    # load optimizer params.
                    diffusion.lr = import_class(f"{args.task_config}.lr")
                    diffusion.iterations = import_class(f"{args.task_config}.iterations")
                    diffusion.decay_steps = import_class(f"{args.task_config}.decay_steps")
                    

                    sample_each = sample_fn(
                        model,
                        motion[ii:ii+1].shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs_each,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        # init_image=motion.to(dist_util.dev()),
                        init_image=None,
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    sample.append(sample_each)
                    loss.append(diffusion.loss_ret_val)
                    constraint.append([0,0,0])

                    print("-->loss_ret_val = ", diffusion.loss_ret_val.mean())

                    if (demo_num is not None) and len(sample)>=demo_num:
                        break

                sample = jt.cat(sample, 0)
                # assert sample.shape==motion.shape
                lengths = model_kwargs['y']['lengths']
                texts = model_kwargs['y']['text']
                loss = np.array(loss)
                constraint = np.array(constraint)

                if (demo_num is not None):
                    sample = sample[:demo_num]
                    lengths = lengths[:demo_num]
                    texts = texts[:demo_num]

            generated_motion.append(sample.data)
            length_list.append( lengths.data )
            text_list += texts
            
            # caption = model_kwargs['y']['text']
            # tokens = tokens
            # cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))]
            if demo_num is not None:
                caption = model_kwargs['y']['text'][:demo_num]
                tokens = tokens[:demo_num]
                cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))][:demo_num]
            else:
                caption = model_kwargs['y']['text']
                tokens = tokens
                cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))]

            caption_list += caption
            tokens_list += tokens 
            cap_len_list += cap_len 

            loss_list.append(loss)
            constraint_list.append(constraint)


    generated_motion = jt.cat(generated_motion, 0)
    length_list = jt.cat(length_list, 0)
    print("final len(generated_motion)=", len(generated_motion), generated_motion.shape)
    assert len(length_list) == len(text_list )
    loss_list = np.concatenate(loss_list, 0)
    constraint_list = np.concatenate(constraint_list, 0)
    assert len(loss_list) == len(length_list)
    assert len(constraint_list) == len(length_list)
    # return generated_motion, length_list, text_list 
    
    assert len(caption_list) == len(length_list)
    assert len(tokens_list)  == len(length_list)
    assert len(cap_len_list) == len(length_list)

    # return [generated_motion, loss_list, constraint_list], length_list, text_list 
    return [generated_motion, loss_list, constraint_list], length_list, text_list, [caption_list, tokens_list, cap_len_list] 



def get_slice_model_kwargs(model_kwargs, i):
    res={}
    res['y']={}
    for k,v in model_kwargs['y'].items():
        res['y'][k] = v[i:i+1]
    return res 



####################################################################
# transform and geometric loss
####################################################################

class DataTransform(object):
    def __init__(self, device='cpu') -> None:
        self.load_inv_normalization_data(device)


    def load_inv_normalization_data(self, device):
        # model.nfeats = 1
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        root_dir = ROOT_DIR
        mean_file = os.path.join(root_dir, "dataset/HumanML3D/Mean.npy")
        std_file  = os.path.join(root_dir, "dataset/HumanML3D/Std.npy")

        d_mean = np.load(mean_file)
        d_std  = np.load(std_file)
        d_mean = jt.array(d_mean)
        d_std  = jt.array(d_std)
        self.d_mean = d_mean[None,:,None,None]
        self.d_std  = d_std[None,:,None,None]

        # t2m_for_eval 
        # omitted


    def do_inv_norm(self, data):
        # model.nfeats = 1
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)

        assert data.shape[1]==self.d_mean.shape[1] and data.shape[2]==self.d_mean.shape[2]
        assert data.shape[1]==self.d_std.shape[1] and data.shape[2]==self.d_std.shape[2]

        return data * self.d_std + self.d_mean
        

    def do_norm(self, data):
        assert data.shape[1]==self.d_mean.shape[1] and data.shape[2]==self.d_mean.shape[2]
        assert data.shape[1]==self.d_std.shape[1] and data.shape[2]==self.d_std.shape[2]
        return (data - self.d_mean) / self.d_std
    


    def sample_to_joints(self, sample):
        # # torch.Size([1, 22, 3, 196])
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        x_pred = sample
        x_pred = self.do_inv_norm(x_pred)

        print("x_pred.shape = ", x_pred.shape)
        
        sample = recover_from_ric(x_pred.permute(0,2,3,1).contiguous(), 22)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        # torch.Size([1, 22, 3, 196])
        # assert sample.shape[0]==1
        return sample
    
    
    def nvct2ntvc(self, joints):
        # torch.Size([1, 22, 3, 196])
        # (seq_len, joints_num, 3)
        joints = joints.permute(0,3,1,2).contiguous()
        return joints



def get_loss_stat(loss_list):
    loss_mean = loss_list.mean()
    return loss_mean




####################################################################
# save
####################################################################

def save_to_npy_with_motion_gen(out_path, all_motions, all_text, all_lengths, fid, motion_gen, loss,
                                constraint):
    '''
    all in np.ndarray
    all_motions:   # [bs, njoints, 3, seqlen], .e.g (1, 22, 3, 196)
    all_text:     list 
    all_lengths:  np.ndarray of int
    '''
    # npy_path = os.path.join(out_path, 'results.npy')
    npy_path = out_path
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': 1, 'num_repetitions': 1,
             'fid': np.array([fid]),
             'motion_gen': motion_gen,
             'loss': loss,
             'constraint': constraint})

class Gen_loader(object):
    def __init__(self):
        self.dataset = None


if __name__ == '__main__':
    # args_list = evaluation_inpainting_parser()

    jt.flags.use_cuda = 1

    args_list = evaluation_inpainting_parser_add_args(f_add_args)
    args = args_list[0]
    fixseed(args.seed)
    print("args.seed=", args.seed)

    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    if args.use_ddim_tag==1:
        use_ddim_tag=True
    elif args.use_ddim_tag==0:
        use_ddim_tag=False 
    else:
        raise ValueError()
    

    # set mask_type 
    mask_type = args.mask_type 
    assert mask_type in ['root_horizontal', 'left_wrist']
    args_list[0].inpainting_mask = mask_type
    args.inpainting_mask         = mask_type

    id_str = ''
    log_file = os.path.join(os.path.dirname(args.model_path), f'debug_ddim{int(use_ddim_tag)}_{args.save_tag}_{id_str}'+'_eval_humanml_{}_{}'.format(name, niter))


    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    if args.inpainting_mask != '':
        log_file += f'_mask_{args.inpainting_mask}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')
    if os.path.exists(log_file):
        os.remove(log_file)
    assert args.overwrite or not os.path.exists(log_file), "Log file already exists!"

    
    # replication_times = replication_times if args.replication_times is None else args.replication_times


    # dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = args.text_split

    # gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='gt')
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='eval')
    # num_actions = gen_loader.dataset.num_actions
    gen_loader = Gen_loader()



    logger.log("Creating model and diffusion...")
    # from diffusion.ddim import InpaintingGaussianDiffusion
    # from model.mdm_naive import MDM
    from diffusion.ddim_jt import InpaintingGaussianDiffusion
    from model.mdm_naive_jt import MDM

    # args.filter_noise = True
    DiffusionClass =  InpaintingGaussianDiffusion if args.filter_noise else None
    ModelClass = MDM
    model, diffusion = load_model_blending_and_diffusion(args_list, gen_loader, None, ModelClass=ModelClass, DiffusionClass=DiffusionClass)


    
    data_transform = DataTransform(device='cpu')


    replication_times=1
    fid_all_list = []
    res_list=[]
    loss_all_list = []
    for ii in range(replication_times):

        num_samples_limit = args.num_samples_limit

        # generate motions
        motion_gen_all, length_gen, texts_gen ,(caption_list, tokens_list, cap_len_list)  = get_gen_motion(args, model, gen_loader, num_samples_limit, args.guidance_param,
                                    init_motion_type=None)
        
        motion_gen, loss_head_gen, constraint_gen = motion_gen_all
        print("constraint_gen.shape = ", constraint_gen.shape)
        
        if args.ret_type=="pos":
            motion_gen_joints = data_transform.sample_to_joints(motion_gen)
        else:
            raise ValueError()
        motion_gen_joints_copy = motion_gen_joints.detach().clone()
        print(f"--> motion_gen = {motion_gen.shape}, motion_gen_joints = {motion_gen_joints.shape}")

        
        # save result
        os.makedirs(args.save_fig_dir, exist_ok=True)
        do_evaluation = False 
        if not do_evaluation:
            # save_to_npy_with_motion_gen
            save_npy_path = os.path.join(args.save_fig_dir, "gen.npy")
            fid = None
            motion_gen=None
            save_to_npy_with_motion_gen(save_npy_path, 
                        all_motions=motion_gen_joints_copy.numpy(), 
                        all_text=list(texts_gen),
                        all_lengths = length_gen.numpy(),
                        fid = fid,
                        motion_gen = motion_gen,
                        loss = loss_head_gen, 
                        constraint = constraint_gen 
            )
            
        
