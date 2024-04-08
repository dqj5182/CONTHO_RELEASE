import cv2
import math
import torch
import scipy
import trimesh
import numpy as np
import torch.nn as nn
import torchgeometry as tgm
from torch.nn import functional as F
from skimage.util.shape import view_as_windows

from core.config import cfg


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB': img = img[:,:,::-1]
    img = img.astype(np.float32)
    return img
    

def add_pelvis_and_neck(joint_coord, joint_valid, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    
    if joint_valid[lhip_idx] > 0 and joint_valid[rhip_idx] > 0:
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))
        joint_valid = np.append(joint_valid, 1)
    else:
        pelvis = np.zeros_like(joint_coord[0, None, :])
        joint_valid = np.append(joint_valid, 0)

    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')

    if joint_valid[lshoulder_idx] > 0 and joint_valid[rshoulder_idx] > 0:
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))
        joint_valid = np.append(joint_valid, 1)
    else:
        neck = np.zeros_like(joint_coord[0, None, :])
        joint_valid = np.append(joint_valid, 0)

    joint_coord = np.concatenate((joint_coord, pelvis, neck))
    return joint_coord, joint_valid


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def center_scale_to_box(center, scale, pixel_std=200.0):
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
    return bbox


def get_bbox(joint_img, joint_valid=None, extend_ratio=1.2, xywh=True):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    if joint_valid is not None:
        x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    if xywh:
        bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    else:
        bbox = np.array([xmin, ymin, xmax, ymax]).astype(np.float32)
    return bbox


def masks2bbox(masks, thres=127):
    """
    convert a list of masks to an bbox of format xyxy
    :param masks:
    :param thres:
    :return:
    """
    mask_comb = np.zeros_like(masks[0])
    for m in masks:
        mask_comb += m
    mask_comb = np.clip(mask_comb, 0, 255)
    ret, threshed_img = cv2.threshold(mask_comb, thres, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bmin, bmax = np.array([50000, 50000]), np.array([-100, -100])
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bmin = np.minimum(bmin, np.array([x, y]))
        bmax = np.maximum(bmax, np.array([x+w, y+h]))
    return [bmin[0], bmin[1], bmax[0]-bmin[0], bmax[1]-bmin[1]]


def process_bbox(bbox, input_shape, target_shape, expand_ratio=1.25, do_sanitize=False):
    if do_sanitize:
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((input_shape[1] - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((input_shape[0] - 1, y1 + np.max((0, h - 1))))
        if w*h > 0 and x2 > x1 and y2 > y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1])
        else:
            return None
    
    # aspect ratio preserving bbox
    bbox = np.array(bbox)
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = target_shape[1] / target_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*expand_ratio
    bbox[3] = h*expand_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


def pixel2cam(coords, c, f):
    cam_coord = np.zeros((len(coords), 3))
    z = coords[..., 2].reshape(-1, 1)

    cam_coord[..., :2] = (coords[..., :2] - c) * z / f
    cam_coord[..., 2] = coords[..., 2]
    return cam_coord


def generate_heatmap(points, valid, image_size, heatmap_size, sigma=2):
    num_points = points.shape[0]
    image_size, heatmap_size = np.array(image_size[::-1]), np.array(heatmap_size[::-1])
    target = np.zeros((num_points, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    target_weight = valid[:,None]
    
    tmp_size = sigma * 3

    for joint_id in range(num_points):
        feat_stride = image_size / heatmap_size
        mu_x = int(points[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(points[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight.reshape(-1)


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def heatmap_to_coords(batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)
    maxvals = maxvals.reshape(maxvals.shape[0], maxvals.shape[1])
    
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px]-hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25
    
    coords = coords.copy()
    coords_valid = (maxvals > 0.5) * 1.0
    return coords, coords_valid


def image_bound_check(coord, image_size, val=None):
    if val is None:
        val = np.ones((len(coord),))
    else:
        val = val.copy()
    
    idxs = np.logical_or(coord[:,0] < 0, coord[:,0] > image_size[1])
    val[idxs] = 0
    
    idxs = np.logical_or(coord[:,1] < 0, coord[:,1] > image_size[0])
    val[idxs] = 0
    return val


def rot6d_to_aa(x):
    batch_size = x.shape[0]
    x = x.reshape(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
    
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).to(rot_mat.device).float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def axis_angle_to_6d(x):
    x = batch_rodrigues(x.reshape(-1, 3))
    x = rotmat_to_6d(x).reshape(-1, 6)
    return x


def rotmat_to_aa(x):
    if not torch.is_tensor(x): x = torch.tensor(x)
    x = x.reshape(-1, 3, 3)
    x = x[:, :, :2].reshape(-1, 6)
    x = x.reshape(-1, 6)

    x = rot6d_to_aa(x)
    return x


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def rotmat_to_6d(poses):
    curr_pose = poses.reshape(-1, 3, 3)
    orth6d = curr_pose[:, :, :2].reshape(-1, 6)
    orth6d = orth6d.reshape(poses.shape[0], -1, 6)
    return orth6d


def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []
    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    
    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices


def select_vid(db, target_vid=''):
    valid_names = db['vid_name']
    unique_names = np.unique(valid_names)
    for u_n in unique_names:
        if not target_vid in u_n:
            continue

        indexes = valid_names == u_n

        new_db = {}
        for k,v in db.items():
            new_db[k] = db[k][indexes]
       
    return new_db 


def get_sequence(start_index, end_index, seqlen, data):
    if start_index != end_index:
        return data[start_index:end_index+1].copy()
    else:
        return data[start_index:start_index+1].repeat(seqlen, axis=0).copy()


def rotate_mesh(mesh, degree, axis=(1,0,0)):
    R = np.array(trimesh.transformations.rotation_matrix(np.radians(degree), axis)[:3, :3])
    center = mesh.mean(0)[None,:]
    mesh = np.dot(mesh - center, R) + center
    return mesh


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def convert_focal_princpt(focal, princpt, trans):
    focal = np.array([[focal[0], 0], [0, focal[1]], [0, 0]])
    princpt = np.array([[princpt[0], 0], [0, princpt[1]], [1, 1]])

    focal = np.dot(trans, focal)
    princpt = np.dot(trans, princpt)

    focal = [focal[0][0], focal[1][1]]
    princpt = [princpt[0][0], princpt[1][1]]  
    return focal, princpt


def convert_focal_princpt_bbox(bbox=None):
    focal = [cfg.CAMERA.focal[0] / cfg.MODEL.input_img_shape [1] * bbox[2], cfg.CAMERA.focal[1] / cfg.MODEL.input_img_shape[0] * bbox[3]]
    princpt = [cfg.CAMERA.princpt[0] / cfg.MODEL.input_img_shape [1] * bbox[2] + bbox[0], cfg.CAMERA.princpt[1] / cfg.MODEL.input_img_shape [0] * bbox[3] + bbox[1]]
    return focal, princpt


def slide_window_to_sequence(slide_window,window_step=1):
    window_size = slide_window.shape[1]
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

    sequence = torch.stack(sequence)
    return sequence


def norm_heatmap(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], -1)

        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau

        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    elif norm_type == 'multiple_sampling':

        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])

        # [B, S, K, -1]
        return gumbel_heatmap.transpose(1, 2)
    else:
        raise NotImplementedError


def get_3dbbox(vertices):
    min_x, min_y, min_z = vertices[:,0].min(), vertices[:,1].min(), vertices[:,2].min()
    max_x, max_y, max_z = vertices[:,0].max(), vertices[:,1].max(), vertices[:,2].max()
    
    return np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [max_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [max_x, max_y, max_z]
        ])


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def batch_rigid_transform_3D(A, B):
    bs, n, dim = A.shape
    centroid_A = torch.mean(A, dim=1)
    centroid_B = torch.mean(B, dim=1)
    H = torch.matmul((A - centroid_A[:,None]).transpose(1,2), B - centroid_B[:,None]) / n
    U, s, V = torch.svd(H)
    
    R = torch.matmul(V.transpose(1,2), U.transpose(1,2))

    idxs = torch.det(R) < 0
    if idxs.sum() > 0:
        s[idxs, -1] = -s[idxs, -1]
        V[idxs, 2] = -V[idxs,2]
        R[idxs] = torch.matmul(V[idxs].transpose(1,2), U[idxs].transpose(1,2))

    varP = torch.var(A, dim=1).sum(1)
    c = 1/varP * s.sum(1)

    t = -torch.matmul(c[:,None,None]*R, centroid_A[:,None].transpose(1,2)) + centroid_B[:,None].transpose(1,2)
    t = t[:,:,0]
    return c, R, t


def bbox3d_to_Rt(bbox3d):
    trans = bbox3d.mean(1)

    x_axis = [bbox3d[:,1]-bbox3d[:,0], bbox3d[:,3]-bbox3d[:,2], bbox3d[:,5]-bbox3d[:,4], bbox3d[:,7]-bbox3d[:,6]]
    y_axis = [bbox3d[:,2]-bbox3d[:,0], bbox3d[:,3]-bbox3d[:,1], bbox3d[:,6]-bbox3d[:,4], bbox3d[:,7]-bbox3d[:,5]]
    z_axis = [bbox3d[:,4]-bbox3d[:,0], bbox3d[:,5]-bbox3d[:,1], bbox3d[:,6]-bbox3d[:,2], bbox3d[:,7]-bbox3d[:,3]]

    x_axis, y_axis, z_axis = torch.stack(x_axis).mean(0), torch.stack(y_axis).mean(0), torch.stack(z_axis).mean(0)
    x_axis = x_axis / torch.norm(x_axis, dim=-1)[:,None]
    y_axis = y_axis / torch.norm(y_axis, dim=-1)[:,None]
    z_axis = z_axis / torch.norm(z_axis, dim=-1)[:,None]
    rot = torch.stack((x_axis, y_axis, z_axis), dim=1)

    rot = batch_rodrigues(rotmat_to_aa(rot))
    return rot.transpose(1,2), trans
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def projection(xyz, camera, f):
    # xyz: unit: meter, u = f/256 * (x+dx) / (z+dz)
    transl = camera[1:3]
    scale = camera[0]
    z_cam = xyz[:, 2:] + f / (256.0 * scale) # J x 1
    uvd = np.zeros_like(xyz)
    uvd[:, 2] = xyz[:, 2] / cfg.CAMERA.depth_factor
    uvd[:, :2] = f / 256.0 * (xyz[:, :2] + transl) / z_cam
    return uvd
    

def back_projection(uvd, pred_camera, focal_length=1000.):
    camScale = pred_camera[:1].reshape(1, -1)
    camTrans = pred_camera[1:].reshape(1, -1)

    camDepth = focal_length / (256 * camScale)

    pred_xyz = np.zeros_like(uvd)
    pred_xyz[:, 2] = uvd[:, 2].copy()
    pred_xyz[:, :2] = (uvd[:, :2] * 256 / focal_length) * (pred_xyz[:, 2:] * cfg.CAMERA.depth_factor + camDepth) - camTrans
    pred_xyz[:, 2] = pred_xyz[:, 2] * cfg.CAMERA.depth_factor
    return pred_xyz


def calc_cam_scale_trans(xyz_29, uvd_29):
    xyz_29, uvd_29 = xyz_29.copy(), uvd_29.copy()
    f = np.sqrt(cfg.CAMERA.focal[0]*cfg.CAMERA.focal[1])

    weight = uvd_29[:,[-1]]
    xyz_29, uvd_29 = xyz_29[:,:3], uvd_29[:,:3]
    num_joints = len(uvd_29)

    if weight.sum() < 2:
        return np.zeros(3).astype(np.float32), np.array([0])

    Ax = np.zeros((num_joints, 3))
    Ax[:, 1] = -1
    Ax[:, 0] = uvd_29[:, 0]

    Ay = np.zeros((num_joints, 3))
    Ay[:, 2] = -1
    Ay[:, 0] = uvd_29[:, 1]

    Ax = Ax * weight
    Ay = Ay * weight
    A = np.concatenate([Ax, Ay], axis=0)

    bx = (xyz_29[:, 0] - 256 * uvd_29[:, 0] / f * xyz_29[:, 2]) * weight[:, 0]
    by = (xyz_29[:, 1] - 256 * uvd_29[:, 1] / f * xyz_29[:, 2]) * weight[:, 0]
    b = np.concatenate([bx, by], axis=0)

    A_s = np.dot(A.T, A)
    b_s = np.dot(A.T, b)

    cam_para = np.linalg.solve(A_s, b_s)
    trans = cam_para[1:]
    scale = 1.0 / cam_para[0]

    target_camera = np.zeros(3)
    target_camera[0] = scale
    target_camera[1:] = trans

    backed_projected_xyz = back_projection(uvd_29, target_camera, f)
    diff = np.sum((backed_projected_xyz-xyz_29)**2, axis=-1) * weight[:, 0]
    diff = np.sqrt(diff).sum() / (weight.sum()+1e-6) * 1000 # roughly mpjpe > 70

    target_camera = target_camera.astype(np.float32)
    # new_uvd = projection(xyz_29, target_camera, f)
    if diff < 70: return target_camera, np.array([1])
    else: return target_camera, np.array([0])


def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass


def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []
    
    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))
    
    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape)) 

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i,i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:,:,0] / (width-1) * 2 - 1
    y = joint_xy[:,:,1] / (height-1) * 2 - 1
    grid = torch.stack((x,y),2)[:,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,:,0] # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0,2,1).contiguous() # batch_size, joint_num, channel_dim
    return img_feat


def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio):
    bbox = bbox_center.view(-1,1,2) + torch.cat((-bbox_size.view(-1,1,2)/2., bbox_size.view(-1,1,2)/2.),1) # xyxy in (cfg.output_hm_shape[2], cfg.output_hm_shape[1]) space
    bbox[:,:,0] = bbox[:,:,0] / cfg.MODEL.img_feat_shape[2] * cfg.MODEL.input_body_shape[1]
    bbox[:,:,1] = bbox[:,:,1] / cfg.MODEL.img_feat_shape[1] * cfg.MODEL.input_body_shape[0]
    bbox = bbox.view(-1,4)

    # xyxy -> xywh
    bbox[:,2] = bbox[:,2] - bbox[:,0]
    bbox[:,3] = bbox[:,3] - bbox[:,1]
    
    # aspect ratio preserving bbox
    w = bbox[:,2]
    h = bbox[:,3]
    c_x = bbox[:,0] + w/2.
    c_y = bbox[:,1] + h/2.

    mask1 = w > (aspect_ratio * h)
    mask2 = w < (aspect_ratio * h)
    h[mask1] = w[mask1] / aspect_ratio
    w[mask2] = h[mask2] * aspect_ratio

    bbox[:,2] = w*extension_ratio
    bbox[:,3] = h*extension_ratio
    bbox[:,0] = c_x - bbox[:,2]/2.
    bbox[:,1] = c_y - bbox[:,3]/2.
    
    # xywh -> xyxy
    bbox[:,2] = bbox[:,2] + bbox[:,0]
    bbox[:,3] = bbox[:,3] + bbox[:,1]
    return bbox


def soft_argmax_2d(heatmap2d):
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height*width))
    heatmap2d = F.softmax(heatmap2d, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))

    accu_x = heatmap2d.sum(dim=(2))
    accu_y = heatmap2d.sum(dim=(3))

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2,3))
    accu_y = heatmap3d.sum(dim=(2,4))
    accu_z = heatmap3d.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out