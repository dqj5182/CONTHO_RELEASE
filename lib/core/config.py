import os
import torch
import shutil
import datetime
import numpy as np
import os.path as osp
from easydict import EasyDict as edict

from core.logger import ColorLogger


def init_dirs(dir_list):
    for dir in dir_list:
        if osp.exists(dir) and osp.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(osp.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
KST = datetime.timezone(datetime.timedelta(hours=9)) # CHANGE TIMEZONE FROM HERE


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.name = ''
cfg.DATASET.workers = 4
cfg.DATASET.random_seed = 123
cfg.DATASET.bbox_expand_ratio = 1.3
cfg.DATASET.obj_set = 'behave'


""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.input_img_shape = (512, 512)
cfg.MODEL.input_body_shape = (256, 256)
cfg.MODEL.input_hand_shape = (256, 256)
cfg.MODEL.img_feat_shape = (8, 8, 8)
cfg.MODEL.weight_path = ''


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.batch_size = 16
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 50
cfg.TRAIN.warmup_epoch = 3
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 1.0e-4
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.lr_step = [30]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.momentum = 0
cfg.TRAIN.weight_decay = 0
cfg.TRAIN.beta1 = 0.5
cfg.TRAIN.beta2 = 0.999
cfg.TRAIN.print_freq = 10

cfg.TRAIN.loss_names = ['contact', 'vert', 'edge', 'param', 'coord', 'hand_bbox']
cfg.TRAIN.contact_loss_weight = 1.0
cfg.TRAIN.smpl_vert_loss_weight = 1.0
cfg.TRAIN.obj_vert_loss_weight = 1.0
cfg.TRAIN.smpl_edge_loss_weight = 1.0
cfg.TRAIN.smpl_pose_loss_weight = 1.0
cfg.TRAIN.smpl_shape_loss_weight = 1.0
cfg.TRAIN.obj_pose_loss_weight = 1.0
cfg.TRAIN.obj_trans_loss_weight = 1.0
cfg.TRAIN.smpl_3dkp_loss_weight = 1.0
cfg.TRAIN.smpl_2dkp_loss_weight = 1.0
cfg.TRAIN.pos_2dkp_loss_weight = 1.0
cfg.TRAIN.hand_bbox_loss_weight = 1.0


""" Augmentation """
cfg.AUG = edict()
cfg.AUG.scale_factor = 0.2
cfg.AUG.rot_factor = 30
cfg.AUG.shift_factor = 0
cfg.AUG.color_factor = 0.2
cfg.AUG.blur_factor = 0
cfg.AUG.flip = False


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 32
cfg.TEST.shuffle = False
cfg.TEST.do_eval = True
cfg.TEST.eval_metrics = ['contact_est_p', 'contact_est_r', 'cd_human', 'cd_object', 'contact_rec_p', 'contact_rec_r']
cfg.TEST.print_freq = 10
cfg.TEST.contact_thres = 0.05


""" CAMERA """
cfg.CAMERA = edict()
cfg.CAMERA.focal = (1000, 1000)
cfg.CAMERA.princpt = (cfg.MODEL.input_img_shape[1]/2, cfg.MODEL.input_img_shape[0]/2)
cfg.CAMERA.depth_factor = 4.4
cfg.CAMERA.obj_depth_factor = 2.2*2

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)
torch.backends.cudnn.benchmark = True
logger = None

    
def update_config(dataset_name='', exp_dir='', ckpt_path=''):
    if dataset_name != '':
        dataset_name_dict = {'behave': 'BEHAVE', 'intercap': 'InterCap'}
        cfg.DATASET.name = dataset_name_dict[dataset_name.lower()]
        cfg.DATASET.obj_set = dataset_name

    if exp_dir == '':
        save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-13]
        save_folder = save_folder.replace(" ", "_")
        save_folder_path = 'experiment/{}'.format(save_folder)
    else:
        save_folder_path = 'experiment/{}'.format(exp_dir)

    if ckpt_path != '':
        cfg.MODEL.weight_path = ckpt_path

    cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
    cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
    cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
    cfg.res_dir = osp.join(cfg.output_dir, 'results')
    cfg.log_dir = osp.join(cfg.output_dir, 'log')
    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')

    print("Experiment Data on {}".format(cfg.output_dir))

    init_dirs([cfg.output_dir, cfg.log_dir, cfg.res_dir, cfg.vis_dir, cfg.checkpoint_dir])
    os.system(f'cp -r lib {cfg.output_dir}/codes')
    
    global logger; logger = ColorLogger(cfg.log_dir)