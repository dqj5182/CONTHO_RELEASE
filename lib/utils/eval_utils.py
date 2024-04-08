import torch
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors

from core.config import cfg
from models.templates import smplh


def eval_chamfer_distance(pred_human, target_human, pred_object, target_object, object_faces, sample_num=6000):
    pred_human, target_human = pred_human.copy(), target_human.copy()
    pred_object, target_object, object_faces = pred_object.copy(), target_object.copy(), object_faces.copy()
    batch_size = pred_human.shape[0]

    for j in range(batch_size):
        pred_mesh = np.concatenate((pred_human[j], pred_object[j]))
        target_mesh = np.concatenate((target_human[j], target_object[j]))

        pred_mesh = rigid_align(pred_mesh, target_mesh)
        pred_human[j], pred_object[j] = pred_mesh[:len(pred_human[j]),:], pred_mesh[len(pred_human[j]):,:]
        target_human[j], target_object[j] = target_mesh[:len(target_human[j]),:], target_mesh[len(target_human[j]):,:]
    
    human_chamfer_dist = []
    for j in range(batch_size):
        pred_mesh = trimesh.Trimesh(pred_human[j], smplh.faces, process=False, maintain_order=True)
        target_mesh = trimesh.Trimesh(target_human[j], smplh.faces, process=False, maintain_order=True)

        pred_verts, target_verts = pred_mesh.sample(sample_num), target_mesh.sample(sample_num)
        dist = chamfer_distance(target_verts, pred_verts)
        human_chamfer_dist.append(dist)

    object_chamfer_dist = []
    for j in range(batch_size):
        pred_mesh = trimesh.Trimesh(pred_object[j], object_faces[j], process=False, maintain_order=True)
        target_mesh = trimesh.Trimesh(target_object[j], object_faces[j], process=False, maintain_order=True)

        pred_verts, target_verts = pred_mesh.sample(sample_num), target_mesh.sample(sample_num)
        dist = chamfer_distance(target_verts, pred_verts)
        object_chamfer_dist.append(dist)

    return np.array(human_chamfer_dist), np.array(object_chamfer_dist)


def eval_contact_score(pred_human, pred_object, target_h_contacts, metric='l2'):
    pred_human, pred_object = pred_human.copy(), pred_object.copy()
    batch_size = pred_human.shape[0]

    precision_list = []
    recall_list = []
    for j in range(batch_size):
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(pred_object[j])
        min_y_to_x = x_nn.kneighbors(pred_human[j])[0].squeeze()
        pred_contacts = (min_y_to_x < cfg.TEST.contact_thres)
        pred_idxs = np.where(pred_contacts)[0]

        target_contacts = (target_h_contacts[j] == 1).numpy()
        target_idxs = np.where(target_contacts)[0]
        true_positive = (pred_contacts * target_contacts).sum()

        if len(pred_idxs) > 0:
            precision_list.append(true_positive / len(pred_idxs))
        if len(target_idxs) > 0:
            recall_list.append(true_positive / len(target_idxs))

    return np.array(precision_list), np.array(recall_list)


def eval_contact_estimation(h_contacts, o_contacts, target_h_contacts, target_o_contacts):
    h_contacts, o_contacts = h_contacts.copy(), o_contacts.copy()
    batch_size = h_contacts.shape[0]

    precision_list = []
    recall_list = []
    for j in range(batch_size):
        pred_contacts = smplh.upsample(torch.tensor(h_contacts[j])).numpy() > 0.5
        pred_idxs = np.where(pred_contacts)[0]

        target_contacts = (target_h_contacts[j] == 1).numpy()
        target_idxs = np.where(target_contacts)[0]
        true_positive = (pred_contacts * target_contacts).sum()

        if len(pred_idxs) > 0:
            precision_list.append(true_positive / len(pred_idxs))
        if len(target_idxs) > 0:
            recall_list.append(true_positive / len(target_idxs))

    precision_list2 = []
    recall_list2 = []
    for j in range(batch_size):
        pred_contacts = o_contacts[j] > 0.5
        pred_idxs = np.where(pred_contacts)[0]

        target_contacts = (target_o_contacts[j] == 1).numpy()
        target_idxs = np.where(target_contacts)[0]
        true_positive = (pred_contacts * target_contacts).sum()

        if len(pred_idxs) > 0:
            precision_list2.append(true_positive / len(pred_idxs))
        if len(target_idxs) > 0:
            recall_list2.append(true_positive / len(target_idxs))

    return np.array(precision_list), np.array(recall_list), np.array(precision_list2), np.array(recall_list2)


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||_metric}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||_metric}}

        this is the squared root distance, while pytorch3d is the squared distance
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y) # bidirectional errors are accumulated
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


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


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2