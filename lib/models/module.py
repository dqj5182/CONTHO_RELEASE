import math
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F

from core.config import cfg

from models.resnet import PoseResNet
from models.layer import *
from models.templates import smplh
from funcs_utils import init_weights, soft_argmax_3d, soft_argmax_2d,  restore_bbox, sample_joint_features


class Hand4Whole(nn.Module):
    def __init__(self):
        super(Hand4Whole, self).__init__()
        self.backbone = PoseResNet(50, obj_in=True)
        self.box_net = BoxNet()
        self.hand_roi_net = HandRoI(PoseResNet(50))
        self.body_position_net = PositionNet('body', 50)
        self.body_rotation_net = RotationNet('body', 50)
        self.hand_position_net = PositionNet('hand', 50)
        self.hand_rotation_net = RotationNet('hand', 50)
      
    def forward(self, img, obj_ids):
        batch_size = img.shape[0]
        body_img = F.interpolate(img, cfg.MODEL.input_body_shape)
        img_feat = self.backbone(body_img, obj_id=obj_ids)

        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size = self.box_net(img_feat, body_joint_hm.detach(), body_joint_img.detach())
        # x1y1x2y2 (256x256 scale)
        lhand_bbox_ = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.MODEL.input_hand_shape[1]/cfg.MODEL.input_hand_shape[0], 2.0) # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox_ = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.MODEL.input_hand_shape[1]/cfg.MODEL.input_hand_shape[0], 2.0) # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        lhand_bbox, rhand_bbox = lhand_bbox_.detach(), rhand_bbox_.detach()
        hand_feat = self.hand_roi_net(img[:,:3], lhand_bbox, rhand_bbox) # hand_feat: flipped left hand + right hand

        _, hand_joint_img = self.hand_position_net(hand_feat) # (2N, J_P, 3)
        hand_pose = self.hand_rotation_net(hand_feat, hand_joint_img.detach())
       
        lhand_pose, rhand_pose = hand_pose[:batch_size], hand_pose[batch_size:]
        lhand_joint_img = hand_joint_img[:batch_size,:,:]
        #lhand_joint_img = torch.cat((cfg.MODEL.img_feat_shape[2] - 1 - lhand_joint_img[:,:,0:1], lhand_joint_img[:,:,1:]),2)
        lhand_joint_img_x = lhand_joint_img[:,:,0] / cfg.MODEL.img_feat_shape[2] * cfg.MODEL.input_hand_shape[1]
        lhand_joint_img_x = cfg.MODEL.input_hand_shape[1] - 1 - lhand_joint_img_x
        lhand_joint_img_x = lhand_joint_img_x / cfg.MODEL.input_hand_shape[1] * cfg.MODEL.img_feat_shape[2]
        lhand_joint_img = torch.cat((lhand_joint_img_x[:,:,None], lhand_joint_img[:,:,1:]),2)
        rhand_joint_img = hand_joint_img[batch_size:,:,:]
        lhand_feat = torch.flip(hand_feat[:batch_size,:], [3])
        rhand_feat = hand_feat[batch_size:,:]
        body_pose, shape, cam_param = self.body_rotation_net(img_feat, body_joint_img.detach(), lhand_feat, lhand_joint_img[:,smplh.pos_joint_part['L_MCP'],:].detach(), rhand_feat, rhand_joint_img[:,smplh.pos_joint_part['R_MCP'],:].detach())
        cam_trans = self.get_camera_trans(cam_param)

        pose = torch.cat((body_pose, lhand_pose, rhand_pose),1).reshape(batch_size, -1, 6)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img),1)
        
        # change hand output joint_img according to hand bbox
        for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
            joint_img[:,smplh.pos_joint_part[part_name],0] *= (bbox[:,None,2] - bbox[:,None,0]) / cfg.MODEL.input_body_shape[1] * cfg.MODEL.img_feat_shape[2] / cfg.MODEL.img_feat_shape[2]
            joint_img[:,smplh.pos_joint_part[part_name],0] += (bbox[:,None,0] / cfg.MODEL.input_body_shape[1] * cfg.MODEL.img_feat_shape[2])
            joint_img[:,smplh.pos_joint_part[part_name],1] *= (bbox[:,None,3] - bbox[:,None,1]) / cfg.MODEL.input_body_shape[0] * cfg.MODEL.img_feat_shape[1] / cfg.MODEL.img_feat_shape[1]
            joint_img[:,smplh.pos_joint_part[part_name],1] += (bbox[:,None,1] / cfg.MODEL.input_body_shape[0] * cfg.MODEL.img_feat_shape[1])

        # normalize bbox / joint img
        lhand_bbox_ , rhand_bbox_ = lhand_bbox_.reshape(batch_size, 2, 2), rhand_bbox_.reshape(batch_size, 2, 2)
        hand_bbox = torch.cat((lhand_bbox_, rhand_bbox_), 1)
        hand_bbox = hand_bbox / torch.tensor((cfg.MODEL.input_body_shape[1], cfg.MODEL.input_body_shape[0])).to(img.device) - 0.5
        joint_img[:,:,:2] = joint_img[:,:,:2] / torch.tensor((cfg.MODEL.img_feat_shape[2], cfg.MODEL.img_feat_shape[1])).to(img.device) - 0.5
        joint_img[:,:,2] = joint_img[:,:,2] / cfg.MODEL.img_feat_shape[0] - 0.5
    
        return img_feat, hand_bbox, joint_img, pose, shape, cam_trans 
        
    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.CAMERA.focal[0]*cfg.CAMERA.focal[1]*2.0*2.0/(cfg.MODEL.input_body_shape[0]*cfg.MODEL.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans


class PositionNet(nn.Module):
    def __init__(self, part, resnet_type):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = 25
            self.hm_shape = cfg.MODEL.img_feat_shape
        elif part == 'hand':
            self.joint_num = 20
            self.hm_shape = cfg.MODEL.img_feat_shape
        if resnet_type == 18:
            feat_dim = 512
        elif resnet_type == 50:
            feat_dim = 2048
        self.conv = make_conv_layers([feat_dim,self.joint_num*self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)
      
    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,self.hm_shape[0],self.hm_shape[1],self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)            
        joint_hm = F.softmax(joint_hm.contiguous().view(-1,self.joint_num,self.hm_shape[0]*self.hm_shape[1]*self.hm_shape[2]),2)
        joint_hm = joint_hm.view(-1,self.joint_num,self.hm_shape[0],self.hm_shape[1],self.hm_shape[2])
        return joint_hm, joint_coord

       
class RotationNet(nn.Module):
    def __init__(self, part, resnet_type):
        super(RotationNet, self).__init__()
        self.part = part
        if part == 'body':
            self.joint_num = 25 + 4 + 4 # body + lhand MCP joints + rhand MCP joints
        elif part == 'hand':
            self.joint_num = 20
        if resnet_type == 18:
            feat_dim = 512
        elif resnet_type == 50:
            feat_dim = 2048
        
        if part == 'body':
            self.body_conv = make_conv_layers([feat_dim,512], kernel=1, stride=1, padding=0)
            self.lhand_conv = make_conv_layers([feat_dim,512], kernel=1, stride=1, padding=0)
            self.rhand_conv = make_conv_layers([feat_dim,512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*515,6], relu_final=False)
            self.body_pose_out = make_linear_layers([self.joint_num*515,21*6], relu_final=False) # without root
            self.shape_out = make_linear_layers([feat_dim,10], relu_final=False)
            self.cam_out = make_linear_layers([feat_dim,3], relu_final=False)
        elif part == 'hand':
            self.hand_conv = make_conv_layers([feat_dim,512], kernel=1, stride=1, padding=0)
            self.hand_pose_out = make_linear_layers([self.joint_num*515, 15*6], relu_final=False)

    def forward(self, img_feat, joint_coord_img, lhand_img_feat=None, lhand_joint_coord_img=None, rhand_img_feat=None, rhand_joint_coord_img=None):
        batch_size = img_feat.shape[0]
        
        if self.part == 'body':
            # shape parameter
            shape_param = self.shape_out(img_feat.mean((2,3)))
            
            # camera parameter
            cam_param = self.cam_out(img_feat.mean((2,3)))

            # body pose parameter
            # body feature
            body_img_feat = self.body_conv(img_feat)
            body_img_feat = sample_joint_features(body_img_feat, joint_coord_img[:,:,:2])
            body_feat = torch.cat((body_img_feat, joint_coord_img),2) # batch_size, joint_num (body), 512+3
            # lhand feature
            lhand_img_feat = self.lhand_conv(lhand_img_feat)
            lhand_img_feat = sample_joint_features(lhand_img_feat, lhand_joint_coord_img[:,:,:2])
            lhand_feat = torch.cat((lhand_img_feat, lhand_joint_coord_img),2) # batch_size, joint_num (4), 512+3
            # rhand feature
            rhand_img_feat = self.rhand_conv(rhand_img_feat)
            rhand_img_feat = sample_joint_features(rhand_img_feat, rhand_joint_coord_img[:,:,:2])
            rhand_feat = torch.cat((rhand_img_feat, rhand_joint_coord_img),2) # batch_size, joint_num (4), 512+3
            # forward to fc
            feat = torch.cat((body_feat, lhand_feat, rhand_feat),1)
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            body_pose = self.body_pose_out(feat.view(batch_size,-1))
            body_pose = torch.cat((root_pose, body_pose), -1)
            return body_pose, shape_param, cam_param

        elif self.part == 'hand':
            # hand pose parameter
            img_feat = self.hand_conv(img_feat)
            img_feat_joints = sample_joint_features(img_feat, joint_coord_img[:,:,:2])
            feat = torch.cat((img_feat_joints, joint_coord_img),2) # batch_size, joint_num, 512+3
            hand_pose = self.hand_pose_out(feat.view(batch_size,-1))
            return hand_pose


class BoxNet(nn.Module):
    def __init__(self):
        super(BoxNet, self).__init__()
        self.joint_num = 25
        self.deconv = make_deconv_layers([2048+self.joint_num*cfg.MODEL.img_feat_shape[0],256,256,256])
        self.bbox_center_out = make_conv_layers([256,2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256,256,2], relu_final=False)
        self.rhand_size = make_linear_layers([256,256,2], relu_final=False)
    
    def forward(self, img_feat, joint_hm, joint_img):
        joint_hm = joint_hm.view(joint_hm.shape[0],joint_hm.shape[1]*cfg.MODEL.img_feat_shape[0],cfg.MODEL.img_feat_shape[1],cfg.MODEL.img_feat_shape[2])
        img_feat = torch.cat((img_feat, joint_hm),1)
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center_out(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center  = bbox_center[:,0,:], bbox_center[:,1,:]
        
        # bbox size
        lhand_feat = sample_joint_features(img_feat, lhand_center[:,None,:].detach())[:,0,:]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center[:,None,:].detach())[:,0,:]
        rhand_size = self.rhand_size(rhand_feat)

        lhand_center = lhand_center / cfg.MODEL.img_feat_shape[0]
        rhand_center = rhand_center / cfg.MODEL.img_feat_shape[0]
        return lhand_center, lhand_size, rhand_center, rhand_size


class HandRoI(nn.Module):
    def __init__(self, backbone):
        super(HandRoI, self).__init__()
        self.backbone = backbone
                                           
    def forward(self, img, lhand_bbox, rhand_bbox):
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:,None], lhand_bbox),1) # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:,None], rhand_bbox),1) # batch_idx, xmin, ymin, xmax, ymax
        
        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:,1] = lhand_bbox_roi[:,1] / cfg.MODEL.input_body_shape[1] * cfg.MODEL.input_img_shape[1]
        lhand_bbox_roi[:,2] = lhand_bbox_roi[:,2] / cfg.MODEL.input_body_shape[0] * cfg.MODEL.input_img_shape[0]
        lhand_bbox_roi[:,3] = lhand_bbox_roi[:,3] / cfg.MODEL.input_body_shape[1] * cfg.MODEL.input_img_shape[1]
        lhand_bbox_roi[:,4] = lhand_bbox_roi[:,4] / cfg.MODEL.input_body_shape[0] * cfg.MODEL.input_img_shape[0]
        lhand_img = torchvision.ops.roi_align(img, lhand_bbox_roi, cfg.MODEL.input_hand_shape, aligned=False)
        lhand_img = torch.flip(lhand_img, [3]) # flip to the right hand

        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:,1] = rhand_bbox_roi[:,1] / cfg.MODEL.input_body_shape[1] * cfg.MODEL.input_img_shape[1]
        rhand_bbox_roi[:,2] = rhand_bbox_roi[:,2] / cfg.MODEL.input_body_shape[0] * cfg.MODEL.input_img_shape[0]
        rhand_bbox_roi[:,3] = rhand_bbox_roi[:,3] / cfg.MODEL.input_body_shape[1] * cfg.MODEL.input_img_shape[1]
        rhand_bbox_roi[:,4] = rhand_bbox_roi[:,4] / cfg.MODEL.input_body_shape[0] * cfg.MODEL.input_img_shape[0]
        rhand_img = torchvision.ops.roi_align(img, rhand_bbox_roi, cfg.MODEL.input_hand_shape, aligned=False)
            
        hand_img = torch.cat((lhand_img, rhand_img))
        hand_feat = self.backbone(hand_img)
        return hand_feat


class ObjectRegressor(nn.Module):
    def __init__(self):
        super(ObjectRegressor, self).__init__()
        self.obj_pose_out = nn.Linear(2048, 6)
        self.obj_trans_out = nn.Linear(2048, 3)

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.mean((2,3))
        pred_obj_pose = self.obj_pose_out(x).reshape(batch_size, -1, 6)
        pred_obj_trans = self.obj_trans_out(x)
        return pred_obj_pose, pred_obj_trans