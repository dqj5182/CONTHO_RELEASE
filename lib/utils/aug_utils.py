import cv2
import torch
import random
import numpy as np

from core.config import cfg
from models.templates import smplh


def get_aug_config():
    scale = np.clip(np.random.randn(), -1.0, 1.0) * cfg.AUG.scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * cfg.AUG.rot_factor if random.random() <= 0.6 else 0
    shift = (random.uniform(-cfg.AUG.shift_factor, cfg.AUG.shift_factor), random.uniform(-cfg.AUG.shift_factor, cfg.AUG.shift_factor))
    c_up = 1.0 + cfg.AUG.color_factor
    c_low = 1.0 - cfg.AUG.color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    blur_sigma = random.uniform(0.01, cfg.AUG.blur_factor)
    
    if cfg.AUG.flip:
        do_flip = random.random() <= 0.5
    else:
        do_flip = False
        
    return scale, rot, shift, color_scale, blur_sigma, do_flip


def generate_patch_image(cvimg, bbox, scale, rot, shift, do_flip, out_shape):
    img = cvimg.copy()
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans, inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    
    if do_flip:
        img_patch = img_patch[:, ::-1, :]

    return img_patch, trans, inv_trans


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, shift=(0, 0), inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    t_x, t_y = src_w*shift[0], src_h*shift[1]
    src_center = np.array([c_x+t_x, c_y+t_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        inv_trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    trans, inv_trans = trans.astype(np.float32), inv_trans.astype(np.float32)
    return trans, inv_trans


def img_processing(img, bbox, img_shape, data_split):
    if data_split == 'train':
        scale, rot, shift, color_scale, blur_sigma, do_flip = get_aug_config()
    else:
        scale, rot, shift, color_scale, blur_sigma, do_flip = 1.0, 0.0, (0.0, 0.0), np.array([1,1,1]), 0, False

    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, shift, do_flip, img_shape)
    img = img * color_scale[None,None,:]
    if blur_sigma > 0 : img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    img = np.clip(img, 0, 255)
    return img, trans, inv_trans, rot, do_flip


def flip_joint(kp, width, flip_pairs):
    kp = kp.copy()
    
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    return kp


def flip_2d_joint(kp, width, flip_pairs):
    kp = kp.copy()
    kp[:, 0] = width - kp[:, 0] - 1
    
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    return kp


def flip_3d_joint(kp, flip_pairs):
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    kp[:,0] = - kp[:,0]
    return kp


def coord2D_processing(coord, trans, f, img_shape, flip_pairs=[], inv=False):
    if inv and f:
        coord = flip_2d_joint(coord, img_shape[1], flip_pairs)
    
    coord = np.concatenate((coord, np.ones_like(coord[:,:1])),1)
    coord = np.dot(trans, coord.transpose(1,0)).transpose(1,0)
    coord = coord[:,:2]

    if not inv and f:
        coord = flip_2d_joint(coord, img_shape[1], flip_pairs)

    return coord


def coord3D_processing(coord, r, f, flip_pairs=[]):
    # in-plane rotation
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        
    coord = np.einsum('ij,kj->ki', rot_mat, coord)

    # flip the x coordinates
    if f:
        coord = flip_3d_joint(coord, flip_pairs)
    coord = coord.astype('float32')

    return coord


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped


def smplh_param_processing(human_model_param, cam_param, do_flip, rot):
    pose, shape, trans = human_model_param['pose'], human_model_param['shape'][:10], human_model_param['trans']
    pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); trans = torch.FloatTensor(trans).view(1,-1)

    if 'twist' in human_model_param: twist = human_model_param['twist'].reshape(-1, 2)
    else: twist = np.zeros((23, 2))

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation 
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        root_pose = pose[smplh.root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        pose[smplh.root_joint_idx] = torch.from_numpy(root_pose).view(3)

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
        
    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[smplh.root_joint_idx,:]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
    pose[smplh.root_joint_idx] = root_pose.reshape(3)
    
    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in smplh.flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].copy(), pose[pair[0], :].copy()
            twist[pair[0]-1, :], twist[pair[1]-1, :] = twist[pair[1]-1, :].copy(), twist[pair[0]-1, :].copy()
        pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
        twist[:, 1] *= -1 # sin

    pose = pose.reshape(-1)
    # change to mean shape if beta is too far from it
    shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(-1)
    twist = twist.astype(np.float32)
    return pose, shape


def obj_param_processing(obj_param, cam_param, smpl_center, do_flip, rot):
    pose, trans, obj_name = obj_param['pose'], obj_param['trans'], obj_param['name']

    if do_flip: assert "Object cannot be flipped!"

    # apply camera extrinsic (rotation)
    # merge pose/trans and camera rotation 
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        pose, _ = cv2.Rodrigues(pose)
        pose, _ = cv2.Rodrigues(np.dot(R,pose))
        trans = np.dot(R, trans.T).reshape(-1)
    if 't' in cam_param:
        trans = trans + np.array(cam_param['t'], dtype=np.float32).reshape(-1)
    
    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)

    pose, _ = cv2.Rodrigues(pose)
    pose = cv2.Rodrigues(np.dot(rot_aug_mat,pose))[0].reshape(-1)
    trans = np.dot(rot_aug_mat, trans).reshape(-1)    
    trans = trans - smpl_center
    return pose, trans, obj_name