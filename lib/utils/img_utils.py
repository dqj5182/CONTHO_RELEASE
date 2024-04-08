import os
import cv2
import json
import torch
import numpy as np
import os.path as osp

from core.config import cfg
from models.templates import obj_dict
from utils.funcs_utils import process_bbox
from utils.aug_utils import img_processing


def load_demo_inputs(img_path, mask_h_path, mask_o_path, bbox_path, obj_path):
    # Sanity check
    img_files, mask_h_files, mask_o_files, bbox_files, obj_files = os.listdir(img_path), os.listdir(mask_h_path), os.listdir(mask_o_path), os.listdir(bbox_path), os.listdir(obj_path)
    assert len(img_files) == len(mask_h_files) == len(mask_o_files) == len(bbox_files) == len(obj_files)

    # Load inputs
    obj_id_list = []
    cropped_image_list = []
    obj_name_list = []
    object_info_data = obj_dict.get_obj_info()
    
    for each_idx in range(len(img_files)):
        # Read images and masks
        input_name = img_files[each_idx].split('.')[0]

        input_image = cv2.imread(osp.join(img_path, f'{input_name}.jpg'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1].astype(np.float32)
        input_mask_h = cv2.imread(osp.join(mask_h_path, f'{input_name}.jpg'), cv2.IMREAD_GRAYSCALE)
        input_mask_o = cv2.imread(osp.join(mask_o_path, f'{input_name}.jpg'), cv2.IMREAD_GRAYSCALE)
        width, height, _ = input_image.shape

        # Get bboxes
        with open(osp.join(bbox_path, f'{input_name}.json'), 'r') as f:
            bbox = json.load(f)
        bbox = bbox['bbox']
        bbox = process_bbox(bbox, (height, width), cfg.MODEL.input_img_shape, expand_ratio=cfg.DATASET.bbox_expand_ratio) 

        # Get object names
        with open(osp.join(obj_path, f'{input_name}.json')) as f:
            obj_name = json.load(f)
        obj_name = np.array(obj_name['obj_name'], dtype=object)
        obj_id = np.array([[*object_info_data].index(obj_name[0])])

        # Crop images and masks
        cropped_image, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(input_image, bbox, cfg.MODEL.input_img_shape, 'test')
        cropped_mask_h = cv2.warpAffine(input_mask_h, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
        cropped_mask_o = cv2.warpAffine(input_mask_o, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
        cropped_mask_h[cropped_mask_h<128] = 0; cropped_mask_h[cropped_mask_h>=128] = 1
        cropped_mask_o[cropped_mask_o<128] = 0; cropped_mask_o[cropped_mask_o>=128] = 1

        # Normalize
        cropped_image = torch.from_numpy(cropped_image.astype(np.float32)/255.0)
        cropped_mask_h, cropped_mask_o = torch.from_numpy(cropped_mask_h).float(), torch.from_numpy(cropped_mask_o).float()
        cropped_image = torch.cat((cropped_image, cropped_mask_h[..., None], cropped_mask_o[..., None]), dim=-1)
        cropped_image = cropped_image.permute(2, 0, 1)

        obj_name_list.append(obj_name[None,...])
        obj_id_list.append(obj_id[None,...])

        cropped_image_list.append(cropped_image[None, ...])

    cropped_image_list = torch.from_numpy(np.vstack(cropped_image_list)).cuda()
    obj_id_list = torch.from_numpy(np.vstack(obj_id_list)).cuda()
    obj_name_list = np.vstack(obj_name_list)

    return img_files, cropped_image_list, obj_id_list, obj_name_list