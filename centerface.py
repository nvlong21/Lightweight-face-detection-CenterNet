import numpy as np
import cv2
import datetime
from model.detnet25 import ShuffleNetV2
from model.centernet import EfficientNet
from model.mnet25 import CenterFace_MobileNet
import torch
from collections import OrderedDict
from torchvision import transforms as trans
import time
class CenterFace(object):
    def __init__(self, height, width, landmarks=True):
        self.landmarks = landmarks
        self.net = EfficientNet()
        self.cuda = True
        if self.cuda:
            self.net.cuda()
        checkpoint = torch.load('weights/model_epoch_fn.pt')
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        del checkpoint
        self.transform_img = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.40789654, 0.44719302, 0.47026115], [0.28863828, 0.27408164, 0.27809835])
        ])
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)

    def __call__(self, img, threshold=0.5):
        img = cv2.resize(img, (self.img_w_new, self.img_h_new))
        img = self.transform_img(img)
        img = torch.unsqueeze(img, 0)
        begin = datetime.datetime.now()
        if self.cuda:
            img = img.cuda()
        out = self.net(img)[0]
        if self.landmarks:
            heatmap, scale, offset, lms = out['hm'].detach().cpu().numpy(), out['wh'].detach().cpu().numpy(),\
             out['reg'].detach().cpu().numpy(), out['lm'].detach().cpu().numpy()
        else:
            heatmap, scale, offset = out['hm'], out['wh'], out['reg']
        end = datetime.datetime.now()
            
        print("cpu times = ", end - begin)
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)

        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] // self.scale_w, dets[:, 1:4:2]//self.scale_h  #// self.scale_w, self.scale_h 
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2]// self.scale_w , lms[:, 1:10:2] //self.scale_h#/ / self.scale_w , self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        print(heatmap.shape)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
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
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + x1)
                        lm.append(landmark[0, j * 2+1, c0[i], c1[i]] * s1 + y1)
                        
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    def nms(self, boxes, scores, nms_thresh):
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
