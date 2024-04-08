import cv2
import copy
import torch
import numpy as np
from torch.utils.data import Dataset

from core.config import cfg
from models.templates import smplh, obj_dict
from funcs_utils import load_img, get_bbox, batch_rodrigues, transform_joint_to_other_db
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smplh_param_processing, obj_param_processing, flip_joint


class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_human_2d = False
        self.has_human_3d = False
        self.has_smpl_param = False
        self.has_obj_param = False
        self.has_contact = True
        self.load_mask = True

    def __len__(self):
        # return 2048
        return len(self.datalist)
  
    def __getitem__(self, index):
        data = copy.deepcopy(self.datalist[index])

        # image
        img_path, bbox = data['img_path'], data['bbox']
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, cfg.MODEL.input_img_shape, self.data_split)

        # h3d_keypoints
        if self.has_human_3d:
            h3d_keypoints, h3d_keypoints_valid = data['h3d_keypoints'][:,:3], data['h3d_keypoints'][:,-1]
            h3d_keypoints = coord3D_processing(h3d_keypoints, rot, do_flip, self.joint_set['flip_pairs'])
            root3d_keypoint = h3d_keypoints[self.root_joint_idx]
            h3d_keypoints = h3d_keypoints - root3d_keypoint
            if do_flip: h3d_keypoints_valid = flip_joint(h3d_keypoints_valid, None, self.joint_set['flip_pairs'])
            h3d_keypoints = np.concatenate((h3d_keypoints, h3d_keypoints_valid[:,None]), -1).astype(np.float32)
        else:
            h3d_keypoints = np.zeros((self.joint_set['joint_num'], 4)).astype(np.float32)
            root3d_keypoint = np.zeros((3,)).astype(np.float32)

        # h2d_keypoints
        if self.has_human_2d:
            h2d_keypoints, h2d_keypoints_valid = data['h2d_keypoints'][:, :2], data['h2d_keypoints'][:, -1]
            h2d_keypoints = coord2D_processing(h2d_keypoints, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
            if do_flip: h2d_keypoints_valid = flip_joint(h2d_keypoints_valid, None, self.joint_set['flip_pairs'])
            h2d_keypoints = np.concatenate((h2d_keypoints, h3d_keypoints[:,[2]], h2d_keypoints_valid[:,None]), -1).astype(np.float32)
        else:
            h2d_keypoints = np.zeros((self.joint_set['joint_num'], 4)).astype(np.float32)
        
        # SMPLH_param
        if self.has_smpl_param:
            smpl_pose_, smpl_shape = smplh_param_processing(data['smpl_param'], data['cam_param'], do_flip, rot)
            smpl_pose = batch_rodrigues(torch.tensor(smpl_pose_).reshape(-1, 3)).numpy()
            gender = data['smpl_param']['gender'] if 'gender' in data['smpl_param'] else 'neutral'
            has_smpl_param = np.array([1])
        else:
            smpl_pose_, smpl_shape = np.zeros((156,)).astype(np.float32), np.zeros((smplh.shape_param_dim,)).astype(np.float32)
            smpl_pose = batch_rodrigues(torch.tensor(smpl_pose_).reshape(-1, 3)).numpy()
            has_smpl_param = np.array([0])
            gender = 'neutral'

        # OBJ_param
        if self.has_obj_param:
            obj_pose, obj_trans, obj_name = obj_param_processing(data['obj_param'], data['cam_param'], root3d_keypoint, do_flip, rot)
            obj_pose = batch_rodrigues(torch.tensor(obj_pose).reshape(-1, 3)).squeeze().numpy()
            obj_id = np.array([obj_dict.get_obj_id(obj_name)])
            has_obj_param = np.array([1])
        else:
            obj_pose, obj_trans, obj_name = np.zeros((3,)).astype(np.float32), np.zeros((3,)).astype(np.float32), data['obj_param']['obj_name']
            obj_id = np.array([0]).astype(int)
            has_obj_param = np.array([0])

        # Get contacts
        if self.has_contact:
            h_contacts, o_contacts = data['h_contacts'], data['o_contacts']
        else:
            h_contacts, o_contacts = np.zeros((431,)).astype(np.float32), np.zeros((64,)).astype(np.float32)

        # Post processing
        img = self.transform(img.astype(np.float32)/255.0)
        hp2d_keypoints = transform_joint_to_other_db(h2d_keypoints, self.joint_set['joints_name'], smplh.pos_joints_name)
        human_keypoints = np.concatenate((h2d_keypoints, h3d_keypoints), 1)
        human_keypoints = transform_joint_to_other_db(human_keypoints, self.joint_set['joints_name'], smplh.joints_name)
        h2d_keypoints, h3d_keypoints = human_keypoints[:, :4], human_keypoints[:, 4:]
        
        # Get hand_bbox
        lhand_keypoints = hp2d_keypoints[smplh.pos_joint_part['lhand'],:2]
        lhand_bbox = get_bbox(lhand_keypoints, extend_ratio=4.0, xywh=False).reshape(2,2)
        rhand_keypoints = hp2d_keypoints[smplh.pos_joint_part['rhand'],:2]
        rhand_bbox = get_bbox(rhand_keypoints, extend_ratio=4.0, xywh=False).reshape(2,2)
        hand_bbox = np.concatenate((lhand_bbox, rhand_bbox))

        # 2D keypoints processing & normalization
        h2d_keypoints[:,:2] = h2d_keypoints[:,:2] / np.array(cfg.MODEL.input_img_shape) - 0.5
        h2d_keypoints[:,2] = h2d_keypoints[:,2] / cfg.CAMERA.depth_factor   
        hp2d_keypoints[:,:2] = hp2d_keypoints[:,:2] / np.array(cfg.MODEL.input_img_shape) - 0.5
        hp2d_keypoints[:,2] = hp2d_keypoints[:,2] / cfg.CAMERA.depth_factor
        hp2d_keypoints[smplh.pos_joint_part['lhand'],2] -= hp2d_keypoints[smplh.lwrist_idx,2] 
        hp2d_keypoints[smplh.pos_joint_part['rhand'],2] -= hp2d_keypoints[smplh.rwrist_idx,2] 
        hand_bbox[:,:2] = hand_bbox[:,:2] / np.array(cfg.MODEL.input_img_shape) - 0.5

        # Get human/object mesh
        smpl_mesh_cam, smpl_joint_cam = smplh.get_coords(smpl_pose_, smpl_shape, gender=gender)
        smpl_verts = smpl_mesh_cam - smpl_joint_cam[smplh.root_joint_idx]
        obj_verts = obj_dict[obj_name].load_verts()
        obj_verts = obj_dict.transform_object(obj_verts, obj_pose, obj_trans)
        obj_verts = np.concatenate((obj_verts, np.ones_like(obj_verts[:,[0]])), -1)
    
        if self.load_mask:
            if self.joint_set['name'] == 'BEHAVE':
                human_mask_path = img_path.replace('.color.jpg', '.person_mask.jpg')
                obj_mask_path = img_path.replace('.color.jpg', '.obj_rend_mask.jpg')
            elif self.joint_set['name'] == 'InterCap':
                img_name = img_path.split('/')[-1].split('.')[0]
                human_mask_path = img_path.replace('/color', '/mask').replace(f'{img_name}.jpg', f'{img_name}_human.png')
                obj_mask_path = img_path.replace('/color', '/mask').replace(f'{img_name}.jpg', f'{img_name}_object.png')
            else:
                assert "Invalid joint set!"
            
            human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)
            human_mask = cv2.warpAffine(human_mask, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
            human_mask[human_mask<128] = 0; human_mask[human_mask>=128] = 1
            
            obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
            obj_mask = cv2.warpAffine(obj_mask, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
            obj_mask[obj_mask<128] = 0; obj_mask[obj_mask>=128] = 1

            human_mask, obj_mask = torch.tensor(human_mask).float(), torch.tensor(obj_mask).float()
            img = torch.cat((img, human_mask[None], obj_mask[None]))
        else:
            obj_mask = torch.zeros((cfg.MODEL.input_img_shape[0], cfg.MODEL.input_img_shape[1], 2))

        if self.data_split == 'train':
            inputs = {'img': img, 'obj_id': obj_id}
            targets = {'hand_bbox': hand_bbox, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape, 'smpl_verts': smpl_verts, 'hp2d_keypoints': hp2d_keypoints, 'h2d_keypoints': h2d_keypoints, 'h3d_keypoints': h3d_keypoints,
                        'obj_pose': obj_pose, 'obj_trans': obj_trans, 'obj_verts': obj_verts,
                        'h_contacts': h_contacts, 'o_contacts': o_contacts}
            meta_info = {'has_smpl_param': has_smpl_param, 'has_obj_param': has_obj_param, 'ann_id': data['ann_id'], 'bbox':bbox, 'obj_name': obj_name, 'gender': gender}
        else:
            inputs = {'img': img, 'obj_id': obj_id}
            targets = {'smpl_mesh_cam': smpl_verts, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape, 'h2d_keypoints': h2d_keypoints, 'h3d_keypoints': h3d_keypoints, 
                        'obj_pose': obj_pose, 'obj_trans': obj_trans, 'obj_verts': obj_verts,
                        'h_contacts': h_contacts, 'o_contacts': o_contacts}
            meta_info = {'ann_id': data['ann_id'], 'bbox':bbox, 'img2bb_trans': img2bb_trans, 'bb2img_trans': bb2img_trans, 'img_path': img_path, 'obj_name': obj_name, 'gender': gender}

        return inputs, targets, meta_info