import torch
import torch.nn as nn

from models.templates import smplh


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()        
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, pred, gt, valid=None):
        gt = gt.to(pred.device)
        batch_size = gt.shape[0]
        
        if valid is not None:
            valid = valid.bool()
            pred, gt = pred[valid], gt[valid]

        return self.criterion(pred, gt)


class ParamLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ParamLoss, self).__init__()        
        if type == 'l1': self.criterion = nn.L1Loss(reduction='mean')
        elif type == 'l2': self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, param_out, param_gt, valid=None):
        param_out = param_out.reshape(param_out.shape[0], -1)
        param_gt = param_gt.reshape(param_gt.shape[0], -1).to(param_out.device)

        if valid is not None:
            valid = valid.reshape(-1).to(param_out.device)
            param_out, param_gt = param_out * valid[:,None], param_gt * valid[:,None]
        return self.criterion(param_out, param_gt)


class CoordLoss(nn.Module):
    def __init__(self, type='l1'):
        super(CoordLoss, self).__init__()
        if type == 'l1': self.criterion = nn.L1Loss(reduction='mean')
        elif type == 'l2': self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, pred, target, valid=None):
        target = target.to(pred.device)
        if valid is None:
            if pred.shape[-1] != target.shape[-1]:
                target, valid = target[...,:-1], target[...,-1]
            else:
                return self.criterion(pred, target)
        else:
            valid = valid.to(pred.device)

        pred, target = pred * valid[...,None], target * valid[...,None]
        return self.criterion(pred, target)


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = torch.LongTensor(face.astype(int))

    def forward(self, coord_out, coord_gt):
        coord_gt = coord_gt.to(coord_out.device)
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()


def get_loss():
    loss = {}
    loss['contact'] = ClsLoss()
    loss['vert'] = CoordLoss(type='l1')
    loss['edge'] = EdgeLengthLoss(smplh.faces)
    loss['param'] = ParamLoss(type='l1')
    loss['coord'] = CoordLoss(type='l1')
    loss['hand_bbox'] = CoordLoss(type='l1')  
    return loss
