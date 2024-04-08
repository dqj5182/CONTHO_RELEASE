import numpy as np
import os.path as osp
from pycocotools.coco import COCO

from core.config import cfg
from funcs_utils import process_bbox
from dataset.base_dataset import BaseDataset

class BEHAVE(BaseDataset):
    def __init__(self, transform, data_split):
        super(BEHAVE, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'BEHAVE', 'sequences')
        self.annot_path = osp.join('data', 'base_data', 'annotations')

        self.joint_set = {
            'name': 'BEHAVE',
            'joint_num': 73,
            'joints_name': (
                'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
                'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
                'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
                'L_BigToe', 'L_SmallToe', 'L_Heel', 'R_BigToe',  'R_SmallToe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'
                ),
            'flip_pairs': (
                (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48), (34, 49), (35, 50), (36, 51),
                (52, 55), (53, 56), (54, 57), (58, 63), (59, 64), (60, 65), (61, 66), (62, 67), (69, 70), (71, 72)
                ),
            'skeleton': (
                (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (9, 13), (13, 16), (16, 18), (18, 20), (9, 12), (12, 15),
                (20, 22), (22, 23), (23, 24), (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
                (21, 37), (37, 38), (38, 39), (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51),
                (7, 52), (7, 53), (7, 54), (8, 55), (8, 56), (8, 57), (36, 58), (24, 59), (27, 60), (33, 61), (30, 62), (51, 63), (39, 64), (42, 65), (48, 66), (45, 67), (12, 68), (68, 69), (68, 70), (69, 71), (70, 72)
            )
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        self.has_human_2d = True
        self.has_human_3d = True
        self.has_smpl_param = True
        self.has_obj_param = True

        self.datalist= self.load_data()

    def load_data(self):        
        annot_path = osp.join(self.annot_path, f'behave_{self.data_split}.json')
        db = COCO(annot_path)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])

            bbox = process_bbox(ann['bbox'], (img['height'], img['width']), cfg.MODEL.input_img_shape, expand_ratio=cfg.DATASET.bbox_expand_ratio) 
            if bbox is None: continue

            h2d_keypoints = np.array(ann['h2d_keypoints'], dtype=np.float32).reshape(-1, 2)
            h2d_keypoints_valid = np.ones((len(h2d_keypoints), 1))
            h2d_keypoints = np.concatenate((h2d_keypoints, h2d_keypoints_valid), axis=-1).astype(np.float32)
        
            h3d_keypoints = np.array(ann['h3d_keypoints'], dtype=np.float32).reshape(-1, 3)
            h3d_keypoints_valid = np.ones((len(h3d_keypoints), 1))
            h3d_keypoints = np.concatenate((h3d_keypoints, h3d_keypoints_valid), -1).astype(np.float32)

            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            smpl_param = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in ann['smpl_param'].items()}
            obj_param = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in ann['obj_param'].items()}

            h_contacts = np.array(ann['h_contacts']).astype(np.float32)
            o_contacts = np.array(ann['o_contacts']).astype(np.float32)

            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'h2d_keypoints': h2d_keypoints, 
                'h3d_keypoints': h3d_keypoints,
                'h_contacts': h_contacts,
                'o_contacts': o_contacts,
                'cam_param': cam_param,
                'smpl_param': smpl_param,
                'obj_param': obj_param
                })

        return datalist