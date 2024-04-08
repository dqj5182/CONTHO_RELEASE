import os
import argparse
import __init_path

from core.config import update_config, cfg
parser = argparse.ArgumentParser(description='Test CONTHO')
parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
parser.add_argument('--dataset', type=str, default='behave', choices=['behave', 'intercap'], help='dataset')
parser.add_argument('--checkpoint', type=str, default='', help='model path for evaluation')
parser.add_argument('--exp', type=str, default='', help='assign experiments directory')


# Organize arguments
args = parser.parse_args()
update_config(dataset_name=args.dataset, exp_dir=args.exp, ckpt_path=args.checkpoint)

from core.config import logger
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")


# Prepare tester
from core.base import Tester
tester = Tester(args, load_dir=cfg.MODEL.weight_path)


# Test CONTHO
print("===> Start Evaluation...")
tester.run(0)