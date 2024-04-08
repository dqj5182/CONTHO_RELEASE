import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import torch
import cv2
import trimesh
import pyrender
import matplotlib.pyplot as plt

from core.config import cfg, logger
from models.templates import smplh, obj_dict


def vis_results(inputs, outputs, targets, meta_info, sample_idx=0, print_idx=0):
    for k,v in outputs.items():
        if torch.is_tensor(v): outputs[k] = v[sample_idx].detach().cpu().numpy()
        else: outputs[k] = v[sample_idx]
    for k,v in targets.items():
        if torch.is_tensor(v): targets[k] = v[sample_idx].detach().cpu().numpy()
        else: targets[k] = v[sample_idx]
    for k,v in meta_info.items():
        if torch.is_tensor(v): meta_info[k] = v[sample_idx].detach().cpu().numpy()
        else: meta_info[k] = v[sample_idx]

    aid = meta_info['ann_id'].item()
    img = inputs['img'][sample_idx, :3].cpu().numpy().transpose(1,2,0)[:,:,::-1] * 255.0
    cv2.imwrite(osp.join(cfg.vis_dir, f'{aid:06d}_img.png'), img)

    smpl_mesh_cam = outputs['smpl_verts']
    cam_trans = outputs['cam_trans']
    obj_mesh = obj_dict[meta_info['obj_name']].load_template()
    obj_mesh_cam = obj_dict.transform_object(obj_mesh.vertices, outputs['obj_pose'], outputs['obj_trans'])

    smpl_mesh = trimesh.Trimesh(smpl_mesh_cam + cam_trans, smplh.faces, process=False, maintain_order=True)
    obj_mesh = trimesh.Trimesh(obj_mesh_cam + cam_trans, obj_mesh.faces, process=False, maintain_order=True)
    cam_param = {'focal': (cfg.CAMERA.focal[0]*2, cfg.CAMERA.focal[1]*2), 'princpt': cfg.CAMERA.princpt}
    rendered_img = render_mesh(img, smpl_mesh, (0.72, 0.72, 0.72), obj_mesh, (0.23, 0.28, 0.82), cam_param=cam_param)  
    cv2.imwrite(osp.join(cfg.vis_dir, f'{aid:06d}_rendered.png'), rendered_img)


def vis_results_demo(outputs, img_files, obj_name_list):
    for each_idx in range(len(img_files)):
        img_name = osp.splitext(img_files[each_idx])[0]

        smpl_mesh_cam = outputs['smpl_verts'][each_idx].detach().cpu().numpy()
        obj_mesh = obj_dict[obj_name_list[each_idx].item()].load_template()
        obj_mesh_cam = obj_dict.transform_object(obj_mesh.vertices, outputs['obj_pose'][each_idx].detach().cpu().numpy(), outputs['obj_trans'][each_idx].detach().cpu().numpy())

        smpl_colors = np.ones_like(smpl_mesh_cam) * np.array([0.72, 0.72, 0.72])
        obj_colors = np.ones_like(obj_mesh_cam) * np.array([0.88, 0.42, 0.38])

        smpl_obj_mesh_cam = np.concatenate((smpl_mesh_cam, obj_mesh_cam))
        smpl_obj_faces = np.concatenate((smplh.faces, obj_mesh.faces+len(smpl_mesh_cam)))
        smpl_obj_colors = np.concatenate((smpl_colors, obj_colors))

        if not osp.exists(osp.join(cfg.vis_dir, f'{img_name}')):
            os.makedirs(osp.join(cfg.vis_dir, f'{img_name}'))

        save_obj(smpl_mesh_cam*1000, smplh.faces, osp.join(cfg.vis_dir, f'{img_name}', 'smpl_mesh.obj'))
        save_obj(obj_mesh_cam*1000, obj_mesh.faces, osp.join(cfg.vis_dir, f'{img_name}', 'obj_mesh.obj'))
        save_obj_with_color(smpl_obj_mesh_cam*1000, smpl_obj_faces, smpl_obj_colors, osp.join(cfg.vis_dir, f'{img_name}', 'smpl_obj_mesh.obj'))


def vis_bbox(img, bbox):
    img = img.copy()
    color, thickness = (0, 255, 0), 1

    if len(bbox) == 4:
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        pos1 = (x_min, y_min)
        pos2 = (x_min, y_max)
        pos3 = (x_max, y_min)
        pos4 = (x_max, y_max)

        img = cv2.line(img, pos1, pos2, color, thickness) 
        img = cv2.line(img, pos1, pos3, color, thickness) 
        img = cv2.line(img, pos2, pos4, color, thickness) 
        img = cv2.line(img, pos3, pos4, color, thickness) 
    elif len(bbox) == 8:
        kps_line = obj_dict.skeleton
        for l in range(len(kps_line)):
            i1, i2 = kps_line[l][0], kps_line[l][1]
            
            p1 = bbox[i1,0].astype(np.int32), bbox[i1,1].astype(np.int32)
            p2 = bbox[i2,0].astype(np.int32), bbox[i2,1].astype(np.int32)
            cv2.line(img, p1, p2, color, thickness)

    return img


def vis_keypoints(img, kps, alpha=1, size=3):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if len(kps) == 1: colors = [(255,255,255)]

    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    kp_mask = np.ascontiguousarray(kp_mask, dtype=np.uint8)

    # Draw the keypoints.
    for i in range(len(kps)):
        if kps[i][-1] > 0:
            p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
            cv2.circle(kp_mask, p, radius=size, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
            
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints_with_skeleton(img, kps, kps_line, bbox=None, kp_thre=0.4, alpha=1):
    # Convert form plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_line))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    
    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)
    
    # Perfrom the drawing on a copy of the image, to allow for blending
    kp_mask = np.copy(img)

    # Draw bounding box
    if bbox is not None:
        b1 = bbox[0, 0].astype(np.int32), bbox[0, 1].astype(np.int32)
        b2 = bbox[1, 0].astype(np.int32), bbox[1, 1].astype(np.int32)
        b3 = bbox[2, 0].astype(np.int32), bbox[2, 1].astype(np.int32)
        b4 = bbox[3, 0].astype(np.int32), bbox[3, 1].astype(np.int32)

        cv2.line(kp_mask, b1, b2, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b2, b3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b3, b4, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b4, b1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # Draw the keypoints
    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        
        p1 = kps[i1,0].astype(np.int32), kps[i1,1].astype(np.int32)
        p2 = kps[i2,0].astype(np.int32), kps[i2,1].astype(np.int32)
        if kps[i1,-1] > kp_thre and kps[i2,-1] > kp_thre:
            cv2.line(
                kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[i1,-1] > kp_thre:
            cv2.circle(
                kp_mask, p1, radius=2, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[i2,-1] > kp_thre:
            cv2.circle(kp_mask, p2, radius=2, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def save_obj(v, f=None, file_path=''):
    obj_file = open(file_path, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def save_obj_with_color(v, f, c, file_name=''):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + ' ' + str(c[i][0]) + ' ' + str(c[i][1]) + ' ' + str(c[i][2]) + '\n')
    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def print_eval_history(eval_history):
    for k1 in eval_history.keys():
        message = f'{k1.upper()} | '
        for k2 in eval_history[k1].keys():
                message += f"{k2}: {eval_history[k1][k2][-1]:.2f}\t"
        logger.info(message)


def get_color(idx, num, map='coolwarm'):
    cmap = plt.get_cmap(map)
    colors = [cmap(i) for i in np.linspace(0, 1, num)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if idx % 2 == 0: return colors[idx//2]
    else: return colors[-(idx//2+1)]


def render_mesh(img, mesh1, color1, mesh2=None, color2=None, cam_param=None, return_mesh=False):
    mesh1, mesh2 = mesh1.copy(), mesh2.copy()

    if 'R' in cam_param:
        mesh1.vertices = np.dot(mesh1.vertices, cam_param['R'])
        if mesh2 is not None:
            mesh2.vertices = np.dot(mesh2.vertices, cam_param['R'])
    if 't' in cam_param:
        mesh1.vertices += cam_param['t'][None,:]
        if mesh2 is not None:
            mesh2.vertices += cam_param['t'][None,:]

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh1.apply_transform(rot)
    if mesh2 is not None:
        mesh2.apply_transform(rot)

    # light
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=160.0)
    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([-6.0, 0.0, -8.0])
    scene.add(light, pose=light_pose)

    # mesh
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(color1[0],color1[1],color1[2],1.0), smooth=False)
    mesh1 = pyrender.Mesh.from_trimesh(mesh1, material=material, smooth=False)
    scene.add(mesh1, 'mesh')

    if mesh2  is not None:
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(color2[0],color2[1],color2[2],1.0), smooth=False)
        mesh2 = pyrender.Mesh.from_trimesh(mesh2, material=material, smooth=False)
        scene.add(mesh2, 'mesh')

    # camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (depth > 0)[:, :, None]
    img = rgb[:, :, :3] * valid_mask + img * (1 - valid_mask)
    if return_mesh:
        return img.astype(np.uint8), rgb
    else:
        return img.astype(np.uint8)