import os 

# for mdm pretrained weight. Be sure to have args.json under the same directory.
# MODEL_PATH="./save/humanml_trans_enc_512/model000475000.pt"
MODEL_PATH="./save_jittor/model000475000.pkl"


CLIP_VERSION="./save_jittor/ViT-B-32.pkl"


# Mean.npy/Std.npy, t2m_mean.npy/t2m_std.npy in ddim_*.py and DataTranform in run_demo_*.py 
# t2m checkpoint in progmogen/data_loaders/humanml/networks/evaluator_wrapper.py
# $ROOT_DIR/dataset/HumanML3D in dataloaders/humanml/data/dataset.py
# place HumanML3D dataset under $ROOT_DIR/dataset
ROOT_DIR="."


# for glove.
# Motion_all_dataset in run_demo_*.py, eval_task_*.py 
ABS_BASE_PATH=ROOT_DIR


# for smpl model
# visualize/joints2smpl/src/config.py
SMPL_MODEL_DIR_0 = os.path.join(ROOT_DIR, "body_models/")
# utils/config.py
SMPL_DATA_PATH_0 = os.path.join(SMPL_MODEL_DIR_0, "smpl")




