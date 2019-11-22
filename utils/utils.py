from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.optim.optimizer import Optimizer
import torch
import numpy as np
import torch.nn as nn
import math
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class RegressionTransform(nn.Module):
    def __init__(self,mean=None,std_box=None,std_ldm=None):
        super(RegressionTransform, self).__init__()
        if mean is None:
            #self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std_box is None:
            #self.std_box = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            self.std_box = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std_box = std_box
        if std_ldm is None:
            #self.std_ldm = (torch.ones(1,10) * 0.1).cuda()
            self.std_ldm = (torch.ones(1,10) * 0.1)

    def forward(self,anchors,bbox_deltas,ldm_deltas,img):
        widths  = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x   = anchors[:, :, 0] + 0.5 * widths
        ctr_y   = anchors[:, :, 1] + 0.5 * heights

        # Rescale
        ldm_deltas = ldm_deltas * self.std_ldm.cuda()
        bbox_deltas = bbox_deltas * self.std_box.cuda()

        bbox_dx = bbox_deltas[:, :, 0] 
        bbox_dy = bbox_deltas[:, :, 1] 
        bbox_dw = bbox_deltas[:, :, 2]
        bbox_dh = bbox_deltas[:, :, 3]

        # get predicted boxes
        pred_ctr_x = ctr_x + bbox_dx * widths
        pred_ctr_y = ctr_y + bbox_dy * heights
        pred_w     = torch.exp(bbox_dw) * widths
        pred_h     = torch.exp(bbox_dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        # get predicted landmarks
        pt0_x = ctr_x + ldm_deltas[:,:,0] * widths
        pt0_y = ctr_y + ldm_deltas[:,:,1] * heights
        pt1_x = ctr_x + ldm_deltas[:,:,2] * widths
        pt1_y = ctr_y + ldm_deltas[:,:,3] * heights
        pt2_x = ctr_x + ldm_deltas[:,:,4] * widths
        pt2_y = ctr_y + ldm_deltas[:,:,5] * heights
        pt3_x = ctr_x + ldm_deltas[:,:,6] * widths
        pt3_y = ctr_y + ldm_deltas[:,:,7] * heights
        pt4_x = ctr_x + ldm_deltas[:,:,8] * widths
        pt4_y = ctr_y + ldm_deltas[:,:,9] * heights

        pred_landmarks = torch.stack([
            pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x,pt4_y
        ],dim=2)

        # clip bboxes and landmarks
        B,C,H,W = img.shape

        pred_boxes[:,:,::2] = torch.clamp(pred_boxes[:,:,::2], min=0, max=W)
        pred_boxes[:,:,1::2] = torch.clamp(pred_boxes[:,:,1::2], min=0, max=H)
        pred_landmarks[:,:,::2] = torch.clamp(pred_landmarks[:,:,::2], min=0, max=W)
        pred_landmarks[:,:,1::2] = torch.clamp(pred_landmarks[:,:,1::2], min=0, max=H)

        return pred_boxes, pred_landmarks


def nms(boxes,scores,iou_threshold):
    boxes = boxes.cpu().numpy()
    score = scores.cpu().numpy()

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])
        a=start_x[index]
        b=order[:-1]
        c=start_x[order[:-1]]
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < iou_threshold)
        order = order[left]

    picked_boxes = torch.Tensor(picked_boxes)
    picked_score = torch.Tensor(picked_score)
    return picked_boxes, picked_score