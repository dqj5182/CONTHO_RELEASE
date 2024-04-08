import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from core.config import cfg, logger
from models.templates import smplh, obj_dict
from funcs_utils import rotmat_to_6d
from train_utils import get_dataloader, train_setup, load_checkpoint, AverageMeterDict
from eval_utils import eval_chamfer_distance, eval_contact_score, eval_contact_estimation

 
class BaseTrainer:
    def __init__(self, args, load_dir):
        self.model, checkpoint = prepare_network(args, load_dir, True)
        self.optimizer, self.lr_scheduler, loss_history, eval_history = train_setup(self.model, checkpoint)
        self.loss = prepare_criterion()
        self.batch_generator, _ = get_dataloader(cfg.DATASET.name, is_train=True)
        
        if loss_history is not None: self.loss_history = loss_history
        else: self.loss_history = {}
        if eval_history is not None: self.eval_history = eval_history

        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model) 
        logger.info(f"# of model parameters: {self.count_parameters(self.model)}")


class Trainer(BaseTrainer):
    def __init__(self, args, load_dir):
        super(Trainer, self).__init__(args, load_dir)
        loss_keys = ['total'] + cfg.TRAIN.loss_names
        for k in loss_keys: self.loss_history[k] = []

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run(self, epoch):
        self.model.train()

        running_losses = AverageMeterDict(list(self.loss_history.keys()))
        batch_generator = tqdm(self.batch_generator)
        
        for i, (inputs, targets, meta_info) in enumerate(batch_generator):
            batch_size = inputs['img'].shape[0]

            # Feed-forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Loss calculation
            loss_dict = self.loss_calculation(outputs, targets, meta_info)

            # Optimizer step
            total_loss = 0.0
            for k,v in loss_dict.items(): total_loss += v
            loss_dict['total'] = total_loss
            total_loss.backward()
            self.optimizer.step()

            # Logging
            for k,v in loss_dict.items():
                loss_dict[k] = v.detach().item()
                running_losses[k].update(loss_dict[k], batch_size)
                
            if i % cfg.TRAIN.print_freq == 0:
                message = f'Epoch{epoch} ({i}/{len(batch_generator)})'
                for k,v in loss_dict.items(): 
                    if v != 0.0: message += f' {k}: {v:.3f}'
                batch_generator.set_description(message)

        for k in self.loss_history.keys():
            self.loss_history[k].append(running_losses[k].avg)
        
        message = f'Epoch{epoch} Loss:'
        for k in self.loss_history.keys():
            message += f' {k}: {running_losses[k].avg:.4f}'
        logger.info(message)

    def loss_calculation(self, outputs, targets, meta_info):
        loss_dict = {}
        # Contact loss
        loss_dict['contact'] = cfg.TRAIN.contact_loss_weight * self.loss['contact'](outputs['h_contacts'], targets['h_contacts'])
        loss_dict['contact'] += cfg.TRAIN.contact_loss_weight * self.loss['contact'](outputs['o_contacts'], targets['o_contacts'])

        # Vertex loss
        loss_dict['vert'] = cfg.TRAIN.smpl_vert_loss_weight * self.loss['vert'](outputs['smpl_verts'], targets['smpl_verts'])
        loss_dict['vert'] += cfg.TRAIN.obj_vert_loss_weight * self.loss['vert'](outputs['obj_verts'], targets['obj_verts'])
        
        # Edge loss
        loss_dict['edge'] = cfg.TRAIN.smpl_edge_loss_weight * self.loss['edge'](outputs['smpl_verts'], targets['smpl_verts'])

        # Parameter loss
        loss_dict['param'] = cfg.TRAIN.smpl_pose_loss_weight * self.loss['param'](outputs['initial_smpl_pose'], rotmat_to_6d(targets['smpl_pose']), meta_info['has_smpl_param'])
        loss_dict['param'] += cfg.TRAIN.smpl_shape_loss_weight * self.loss['param'](outputs['initial_smpl_shape'], targets['smpl_shape'], meta_info['has_smpl_param'])
        loss_dict['param'] += cfg.TRAIN.obj_pose_loss_weight * self.loss['param'](outputs['initial_obj_pose'], rotmat_to_6d(targets['obj_pose']).squeeze(), meta_info['has_obj_param'])
        loss_dict['param'] += cfg.TRAIN.obj_trans_loss_weight * self.loss['param'](outputs['initial_obj_trans'], targets['obj_trans'], meta_info['has_obj_param'])
        
        # Coordinate loss
        loss_dict['coord'] = cfg.TRAIN.smpl_3dkp_loss_weight * self.loss['coord'](outputs['h3d_keypoints'], targets['h3d_keypoints'])
        loss_dict['coord'] += cfg.TRAIN.smpl_2dkp_loss_weight * self.loss['coord'](outputs['h2d_keypoints'], targets['h2d_keypoints'][:,:,:2])
        loss_dict['coord'] += cfg.TRAIN.pos_2dkp_loss_weight * self.loss['coord'](outputs['hp2d_keypoints'], targets['hp2d_keypoints'][:,:,:3])
    
        # Bbox loss
        loss_dict['hand_bbox'] = self.loss['hand_bbox'](outputs['hand_bbox'], targets['hand_bbox'])
        return loss_dict


class BaseTester:
    def __init__(self, args, load_dir=''):
        if load_dir != '':
            self.model, _ = prepare_network(args, load_dir, False)
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)

        self.val_loader, dataset_list = get_dataloader(cfg.DATASET.name, is_train=False)
        if dataset_list is not None:
            self.val_dataset = dataset_list[0]
            self.val_loader = self.val_loader[0]
        else:
            self.val_dataset = None

        self.eval_history = {}
        self.eval_vals = {}        
        self.outputs_db = {}
        
    def save_history(self, eval_history):
        if hasattr(self, 'eval_metrics'):
            for k in self.eval_metrics:
                if k in self.eval_vals:
                    eval_history[k].append(self.eval_vals[k])


class Tester(BaseTester):
    def __init__(self, args, load_dir=''):   
        super(Tester, self).__init__(args, load_dir)         
        self.eval_metrics = cfg.TEST.eval_metrics
        self.eval_history = {}
        for k in self.eval_metrics: self.eval_history[k] = []

        self.save_results = {}

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        running_evals = AverageMeterDict(self.eval_metrics)
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        val_loader = tqdm(self.val_loader)
        for i, (inputs, targets, meta_info) in enumerate(val_loader):

            # Feed-forward
            with torch.no_grad():
                outputs = self.model(inputs)

            for k,v in outputs.items():
                if torch.is_tensor(v): outputs[k] = v.cpu().numpy()

            # Evaluation
            if cfg.TEST.do_eval:
                eval_dict = self.evaluation(outputs, targets, meta_info) 

            # Logging
            for k in self.eval_metrics:
                running_evals[k].update(sum(eval_dict[k]), len(eval_dict[k]))
                if len(eval_dict[k]) == 0: eval_dict[k] = 0.0
                else: eval_dict[k] = sum(eval_dict[k])/len(eval_dict[k])

            if i % cfg.TEST.print_freq == 0:
                message = f'{eval_prefix}({i}/{len(self.val_loader)})'
                for k in self.eval_metrics: message += f' {k}: {eval_dict[k]:.2f}'
                val_loader.set_description(message)

        for k in self.eval_metrics:
            self.eval_vals[k] = running_evals[k].avg

        message = 'Finished Evaluation!\n'
        message += f'-------- Evaluation Results (Contact estimation) ------\n'
        message += f'Precision: {self.eval_vals["contact_est_p"]:.3f} / Recall: {self.eval_vals["contact_est_r"]:.3f}\n'
        message += f'---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals["cd_human"]:.2f} / Object: {self.eval_vals["cd_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals["contact_rec_p"]:.3f} / Recall: {self.eval_vals["contact_rec_r"]:.3f}'
        logger.info(message)


    def evaluation(self, outputs, targets, meta_info):
        tar_smpl_mesh_cam = (targets['smpl_mesh_cam']).numpy()
        pred_smpl_mesh_cam, pred_obj_mesh_cam, tar_obj_mesh_cam, obj_faces = [], [], [], []
        for i in range(len(tar_smpl_mesh_cam)):
            smpl_mesh_cam = outputs['smpl_verts'][i]
            smpl_joint_cam = smplh.joint_regressor @ smpl_mesh_cam
            
            smpl_mesh_cam = smpl_mesh_cam - smpl_joint_cam[smplh.root_joint_idx]
            pred_smpl_mesh_cam.append(smpl_mesh_cam)

            obj_name = meta_info['obj_name'][i]
            obj_mesh = obj_dict[obj_name].load_template()
            obj_faces.append(np.array(obj_mesh.faces))

            obj_mesh_cam = obj_dict.transform_object(obj_mesh.vertices, outputs['obj_pose'][i], outputs['obj_trans'][i])

            pred_obj_mesh_cam.append(obj_mesh_cam)
            obj_mesh_cam = obj_dict.transform_object(obj_mesh.vertices, targets['obj_pose'][i].numpy(), targets['obj_trans'][i].numpy())
            tar_obj_mesh_cam.append(obj_mesh_cam)
        pred_smpl_mesh_cam = np.stack(pred_smpl_mesh_cam)

        eval_dict = {}
        if 'cd_human' in self.eval_metrics:
            eval_dict['cd_human'], eval_dict['cd_object'] = eval_chamfer_distance(pred_smpl_mesh_cam, tar_smpl_mesh_cam, pred_obj_mesh_cam, tar_obj_mesh_cam, obj_faces)
            eval_dict['cd_human'], eval_dict['cd_object'] = eval_dict['cd_human']*100, eval_dict['cd_object']*100
 
        if 'contact_est_p' in self.eval_metrics:
            eval_dict['contact_est_p'], eval_dict['contact_est_r'], eval_dict['contact_e2p'], eval_dict['contact_e2r'] = eval_contact_estimation(outputs['h_contacts'], outputs['o_contacts'], targets['h_contacts'], targets['o_contacts'])

        if 'contact_rec_p' in self.eval_metrics:
            eval_dict['contact_rec_p'], eval_dict['contact_rec_r'] = eval_contact_score(pred_smpl_mesh_cam, pred_obj_mesh_cam,  targets['h_contacts'])
        return eval_dict


def prepare_network(args, load_dir='', is_train=True): 
    from models.model import get_model  
    from train_utils import count_parameters
    model = get_model()
    logger.info(f'# of model parameters: {count_parameters(model)}')
    if load_dir and (not is_train or args.resume_training):
        checkpoint = load_checkpoint(load_dir=load_dir)
        try:
            model.load_weights(checkpoint['model_state_dict'])
        except:
            model.load_weights(checkpoint)
    else:
        checkpoint = None
    return model, checkpoint


def prepare_criterion():
    from core.loss import get_loss
    criterion = get_loss()
    return criterion