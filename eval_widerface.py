import utils
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import cv2
import scipy.io as sio
import os
from centerface import CenterFace
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def bbox_overlap(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def get_detections(data_batch, model, cuda = True, threshold=0.35):
    model.eval()
    with torch.no_grad():
        picked_boxes, picked_landmarks = [], []
        img_batch = data_batch['input']

        if cuda:
            img_batch = img_batch.cuda()
        outputs = model(img_batch)[0]
        heatmaps, scales, offsets = torch.clamp(outputs['hm'].sigmoid_(), min=1e-4, max=1-1e-4).detach().cpu().numpy(), outputs['wh'].detach().cpu().numpy(), outputs['reg'].detach().cpu().numpy()
        for i in range(len(img_batch)):
            heatmap, scale, offset = heatmaps[i], scales[i], offsets[i]
            dets = decode(heatmap, scale, offset, None, (640, 640), threshold=threshold)
            picked_boxes.append(dets)
        return picked_boxes

def decode(heatmap, scale, offset, landmark, size, threshold=0.1):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, :, :], scale[1, :, :]
    offset0, offset1 = offset[0, :, :], offset[1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    boxes = []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = scale0[c0[i], c1[i]]*4, scale1[c0[i], c1[i]]*4
            # s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s0 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s1 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s0, size[1]), min(y1 + s1, size[0]), s])
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :]
    return boxes

def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep
def compute_overlap(a,b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    # (N, K) ndarray of overlap between boxes and query_boxes
    return torch.from_numpy(intersection / ua)    
 

def evaluate(val_data,model,threshold=0.5):
    recall = 0.
    precision = 0.
    #for i, data in tqdm(enumerate(val_data)):
    for data in tqdm(iter(val_data)):
        annots = data['meta']['gt_det']
        picked_boxes = get_detections(data, model)
        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):          
            annot_boxes = annots[j]#[:keep_ind]
            annot_boxes = annot_boxes[annot_boxes[:,0]!=-1]
            if boxes is None and annot_boxes.shape[0] == 0:
                continue
            elif (boxes is None or len(boxes) < 1) and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes is not None and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.   
                continue       
            overlap = bbox_overlap(boxes, annot_boxes)

            # compute recall
            max_overlap, _ = torch.max(torch.FloatTensor(overlap),dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num/annot_boxes.shape[0]
            # compute precision
            max_overlap, _ = torch.max(torch.FloatTensor(overlap),dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives/boxes.shape[0]

        recall += recall_iter/len(picked_boxes)
        precision += precision_iter/len(picked_boxes)

    return recall/len(val_data),precision/len(val_data)