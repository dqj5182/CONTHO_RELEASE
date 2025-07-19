import os
import argparse
import torch
from torch.nn.parallel import DataParallel
import __init_path

from core.config import cfg, update_config, logger
from core.base import Trainer, Tester
from train_utils import save_checkpoint, check_data_parallel

# Argument parser
parser = argparse.ArgumentParser(description='Train CONTHO')
parser.add_argument('--resume_training', action='store_true', help='resume training')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--dataset', type=str, default='behave', choices=['behave', 'intercap'], help='dataset')
parser.add_argument('--exp', type=str, default='', help='assign experiments directory')
parser.add_argument('--checkpoint', type=str, default='', help='model path for resuming')

# Organize arguments
args = parser.parse_args()
update_config(dataset_name=args.dataset.lower(), exp_dir=args.exp, ckpt_path=args.checkpoint)

# GPU Setup
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")

# Initialize Trainer & Tester
trainer = Trainer(args, load_dir=cfg.MODEL.weight_path)
tester = Tester(args)

# Move model to GPU
trainer.model.to(device)

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    trainer.model = DataParallel(trainer.model)
    logger.info(f"Using {torch.cuda.device_count()} GPUs for training")

# Initialize Evaluation History
if hasattr(trainer, 'eval_history'):
    tester.eval_history = trainer.eval_history

# Train CONTHO
logger.info(f"===> Start training...")
for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    trainer.run(epoch)

    # Validate CONTHO
    if epoch % 10 == 1 or epoch == cfg.TRAIN.end_epoch:
        tester.run(epoch, current_model=trainer.model)    
        tester.save_history(tester.eval_history)

    # Save checkpoint of CONTHO
    if epoch % 10 == 1 or epoch == cfg.TRAIN.end_epoch:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': check_data_parallel(trainer.model.state_dict()),
            'optim_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
            'train_log': trainer.loss_history,
            'test_log': tester.eval_history
        }, epoch)
