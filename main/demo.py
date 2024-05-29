import os
import torch
import argparse
import __init_path

from core.config import update_config, cfg
from utils.img_utils import load_demo_inputs
from utils.vis_utils import vis_results_demo


parser = argparse.ArgumentParser(description='Demo CONTHO')
parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
parser.add_argument('--checkpoint', type=str, default='', required=True, help='model path for evaluation')
parser.add_argument('--exp', type=str, default='demo_out', help='assign experiments directory')


# Organize arguments
args = parser.parse_args()
update_config(exp_dir=args.exp, ckpt_path=args.checkpoint)

from core.config import logger
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")


# Initialize paths
img_path = 'asset/examples/img'
mask_h_path = 'asset/examples/mask_h'
mask_o_path = 'asset/examples/mask_o'
bbox_path = 'asset/examples/bbox'
obj_path = 'asset/examples/obj_name'


# Load inputs
img_files, cropped_image_list, obj_id_list, obj_name_list = load_demo_inputs(img_path, mask_h_path, mask_o_path, bbox_path, obj_path)


# Model
from core.base import prepare_network
contho_model, _ = prepare_network(args, args.checkpoint, False)
contho_model = contho_model.cuda()
contho_model = contho_model.eval()


# Iterate over inputs
for idx in range(len(img_files)):
    # Organize input
    input = {'img': cropped_image_list[idx:idx+1], 'obj_id': obj_id_list[idx:idx+1]}
    img_file = img_files[idx:idx+1]
    obj_name = obj_name_list[idx:idx+1]

    # Forward input
    with torch.no_grad():
        output = contho_model(input)

    # Save mesh
    vis_results_demo(output, img_file, obj_name)