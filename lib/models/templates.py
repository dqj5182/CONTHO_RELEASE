import cv2
import json
import torch
import smplx
import trimesh
import numpy as np
import os.path as osp

from core.config import cfg
from funcs_utils import transform_joint_to_other_db, scipy_to_pytorch, adjmat_sparse, spmm


class SMPLH(object):
    def __init__(self):
        self.model_path = osp.join('data', 'base_data', 'human_models')
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_betas': False, 'create_transl': False, 'use_pca': False}
        self.layer = {'neutral': smplx.create(self.model_path, 'smplh', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(self.model_path, 'smplh', gender='MALE', **self.layer_arg), 'female': smplx.create(self.model_path, 'smplh', gender='FEMALE', **self.layer_arg)}

        self.faces = self.layer['neutral'].faces
        self.orig_joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)
        
        self.vertex_num = 6890
        self.shape_param_dim = 10
        self.hand_pose_num = 15

        self.body_pose_num = 21
        self.orig_joint_num = 52    # 1 + 21 + 15 + 15 
        self.orig_joints_name = (
            'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
            'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
            'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3'
        )
        self.orig_flip_pairs = (
            (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
            (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48), (34, 49), (35, 50), (36, 51))
        self.orig_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (9, 13), (13, 16), (16, 18), (18, 20), (9, 12), (12, 15),
            (20, 22), (22, 23), (23, 24), (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
            (21, 37), (37, 38), (38, 39), (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51))
        
        self.joint_num = 73    # 1 + 21 + 15 + 15 + 21
        self.joints_name = (
            'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
            'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
            'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
            'L_BigToe', 'L_SmallToe', 'L_Heel', 'R_BigToe',  'R_SmallToe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'
        )
        self.flip_pairs = (
            (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
            (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48), (34, 49), (35, 50), (36, 51),
            (52, 55), (53, 56), (54, 57), (58, 63), (59, 64), (60, 65), (61, 66), (62, 67), (69, 70), (71, 72))
        self.skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (9, 13), (13, 16), (16, 18), (18, 20), (9, 12), (12, 15),
            (20, 22), (22, 23), (23, 24), (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
            (21, 37), (37, 38), (38, 39), (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51),
            (7, 52), (7, 53), (7, 54), (8, 55), (8, 56), (8, 57), (36, 58), (24, 59), (27, 60), (33, 61), (30, 62), (51, 63), (39, 64), (42, 65), (48, 66), (45, 67), (12, 68), (68, 69), (68, 70), (69, 71), (70, 72))
    

        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_BigToe', 'L_SmallToe', 'L_Heel', 'R_BigToe', 'R_SmallToe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         )
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]

        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.joint_regressor = self.orig_joint_regressor
        
        self._A, self._U, self._D = self.get_graph_params(filename=osp.join('data', 'base_data', 'human_models', 'smpl_downsampling.npz'))
        self.num_downsampling = 2

        upsampler = np.load('data/base_data/human_models/smpl_upsampling.npz')
        self._U = []
        self._U.append(torch.Tensor(upsampler['U2']))
        self._U.append(torch.Tensor(upsampler['U1']))
        
        self.joint_regressor = self.make_joint_regressor()

    def make_joint_regressor(self):
        joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.joints_name)
        joint_regressor[self.joints_name.index('L_BigToe')] = np.eye(self.vertex_num)[3216]
        joint_regressor[self.joints_name.index('L_SmallToe')] = np.eye(self.vertex_num)[3226]
        joint_regressor[self.joints_name.index('L_Heel')] = np.eye(self.vertex_num)[3387]

        joint_regressor[self.joints_name.index('R_BigToe')] = np.eye(self.vertex_num)[6617]
        joint_regressor[self.joints_name.index('R_SmallToe')] = np.eye(self.vertex_num)[6624]
        joint_regressor[self.joints_name.index('R_Heel')] = np.eye(self.vertex_num)[6787]

        joint_regressor[self.joints_name.index('L_Thumb_4')] = np.eye(self.vertex_num)[2746]
        joint_regressor[self.joints_name.index('L_Index_4')] = np.eye(self.vertex_num)[2319]
        joint_regressor[self.joints_name.index('L_Middle_4')] = np.eye(self.vertex_num)[2445]
        joint_regressor[self.joints_name.index('L_Ring_4')] = np.eye(self.vertex_num)[2556]
        joint_regressor[self.joints_name.index('L_Pinky_4')] = np.eye(self.vertex_num)[2673]
        
        joint_regressor[self.joints_name.index('R_Thumb_4')] = np.eye(self.vertex_num)[6191]
        joint_regressor[self.joints_name.index('R_Index_4')] = np.eye(self.vertex_num)[5782]
        joint_regressor[self.joints_name.index('R_Middle_4')] = np.eye(self.vertex_num)[5905]
        joint_regressor[self.joints_name.index('R_Ring_4')] = np.eye(self.vertex_num)[6016]
        joint_regressor[self.joints_name.index('R_Pinky_4')] = np.eye(self.vertex_num)[6133]

        joint_regressor[self.joints_name.index('Nose')] = np.eye(self.vertex_num)[332]
        joint_regressor[self.joints_name.index('L_Eye')] = np.eye(self.vertex_num)[2800]
        joint_regressor[self.joints_name.index('R_Eye')] = np.eye(self.vertex_num)[6260]
        joint_regressor[self.joints_name.index('L_Ear')] = np.eye(self.vertex_num)[583]
        joint_regressor[self.joints_name.index('R_Ear')] = np.eye(self.vertex_num)[4071]
        return joint_regressor

    def get_coords(self, smpl_pose, smpl_shape, gender='neutral'):
        smpl_shape = torch.tensor(smpl_shape.reshape(-1, 10))
        smpl_pose = torch.tensor(smpl_pose.reshape(-1, 156))
        output = self.layer[gender](betas=smpl_shape, global_orient=smpl_pose[:, :3], body_pose=smpl_pose[:, 3:66], left_hand_pose=smpl_pose[:, 66:111], right_hand_pose=smpl_pose[:, 111:])
        smpl_mesh_cam = output.vertices
        smpl_joint_cam = torch.matmul(torch.tensor(self.joint_regressor[None,:,:]), smpl_mesh_cam)
        return smpl_mesh_cam.squeeze().numpy(), smpl_joint_cam.squeeze().numpy()

    def get_graph_params(self, filename, nsize=1):
        """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
        data = np.load(filename, encoding='latin1', allow_pickle=True)
        A = data['A']
        U = data['U']
        D = data['D']
        U, D = scipy_to_pytorch(A, U, D)
        A = [adjmat_sparse(a, nsize=nsize) for a in A]
        return A, U, D

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                x = spmm(self._D[i].to(x.device), x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(self._D[j].to(x.device), y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=2, n2=0):
        """Upsample mesh."""
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i].to(x.device), x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j].to(x.device), y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x


class InteractObject(object):
    def __init__(self, data):
        self.path, self.verts64 = data['path'], np.array(data['kps']).astype(np.float32)
        obj = trimesh.load(data['path'], process=False, maintain_order=True)
        self.vertices, self.faces = obj.vertices, obj.faces
        self.vertex_num = len(self.vertices)

    def load_template(self):
        template = trimesh.load(self.path, process=False, maintain_order=True)
        return template
    
    def load_verts(self):
        return self.verts64

    def transform_object(self, verts, pose, trans, scale=1.0):
        if pose.ndim == 1: rot, _ = cv2.Rodrigues(pose)
        else: rot = pose
        verts = np.matmul(np.array(verts), rot.T) + trans
        return scale * verts


class InteractObjectDict(object):
    def __init__(self):
        if cfg.DATASET.obj_set == 'behave':
            obj_info_path = osp.join('data', 'base_data', 'object_models', 'behave', '_info.json')
        elif cfg.DATASET.obj_set == 'intercap':
            obj_info_path = osp.join('data', 'base_data', 'object_models', 'intercap', '_info.json')
        else:
            assert "Invalid object set!"

        with open(obj_info_path) as f:
            self.obj_info = json.load(f)

        self.obj_names = []
        for k, v in self.obj_info.items():
            obj_name = v['path'].split('/')[-1].replace('.obj', '')
            
            obj_info = InteractObject(v)
            setattr(self, obj_name, obj_info)
            self.obj_names.append(obj_name)
        self.obj_num = len(self.obj_names)
    
    def get_obj_info(self):
        return self.obj_info

    def get_obj_id(self, name):
        return self.obj_names.index(name)

    def get_obj_verts(self, key):
        return getattr(self, key).vertices64

    def __getitem__(self, key):
        if type(key) is int: key = self.obj_names[key]
        return getattr(self, key)
    
    def transform_object(self, verts, pose, trans, scale=1.0):
        if pose.ndim == 1: rot, _ = cv2.Rodrigues(pose)
        else: rot = pose
        verts = np.matmul(np.array(verts), rot.T) + trans
        return scale * verts


smplh = SMPLH()
obj_dict = InteractObjectDict()