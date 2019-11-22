import utils
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from utils import cpu_nms as  nms
import time
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


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
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
def get_detections(img_batch, model,score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        classifications, bboxes, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        
        for i in range(batch_size):
            classification = torch.exp(classifications[i,:,:])
            bbox = bboxes[i,:,:]
            landmark = landmarks[i,:,:]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax==1
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice
            
            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]
            keep = nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
        
        return picked_boxes, picked_landmarks

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


def evaluate(val_data,retinaFace,threshold=0.5):
    recall = 0.
    precision = 0.
    #for i, data in tqdm(enumerate(val_data)):
    for data in tqdm(iter(val_data)):
        img_batch = data['img'].cuda()
        annots = data['annot'].cuda()


        picked_boxes,_ = get_detections(img_batch,retinaFace)
        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):          
            annot_boxes = annots[j]
            annot_boxes = annot_boxes[annot_boxes[:,0]!=-1]

            if boxes is None and annot_boxes.shape[0] == 0:
                continue
            elif boxes is None and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes is not None and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.   
                continue         
            
            overlap = box_iou(annot_boxes, boxes)
                 
            # compute recall
            max_overlap, _ = torch.max(overlap,dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num/annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap,dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives/boxes.shape[0]

        recall += recall_iter/len(picked_boxes)
        precision += precision_iter/len(picked_boxes)

    return recall/len(val_data),precision/len(val_data)