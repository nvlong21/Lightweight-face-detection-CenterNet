from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import time
import os
try:

    from .image import flip, color_aug
    from .image import get_affine_transform, affine_transform
    from .image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
    from .image import draw_dense_reg
except:
    from image import flip, color_aug
    from image import get_affine_transform, affine_transform
    from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
    from image import draw_dense_reg
import math
import matplotlib.pyplot as plt
def check_landmarks(lm, bbox):
    w_s = lm[[0, 2, 4, 6, 8]] - bbox[0]
    w_idx = list(np.where(w_s < 0))[0]
    h_s = lm[[1, 3, 5, 7, 9]] - bbox[1]
    h_idx = list(np.where(h_s < 0))[0]
    if (w_idx.shape[0] +  h_idx.shape[0]) == 0:
        return True
    return False

class CenterFaceData(data.Dataset):
    mean = np.array([0.408, 0.447, 0.470],
                                     dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.289, 0.274, 0.278],
                                     dtype=np.float32).reshape(1, 1, 3)
    def __init__(self,txt_path, split= "train", debug = True):
        super(CenterFaceData, self).__init__()
        self.imgs_path = []
        self.words = []
        self.batch_count = 0
        self.img_size = 640
        self.split = split
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
                line = line.rstrip() 
                if line.startswith('#'):
                        if isFirst is True:
                                isFirst = False
                        else:
                                labels_copy = labels.copy()
                                self.words.append(labels_copy)        
                                labels.clear()       
                        path = line[2:]
                        path = txt_path.replace('label.txt','images/') + path
                        self.imgs_path.append(path)

                else:
                        line = line.split(' ')
                        label = [float(x) for x in line]
                        labels.append(label)
        self.words.append(labels)
        self.default_resolution = [640, 640]
        self.max_objs = 128
        self.keep_res = False
        self.down_ratio = 4

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                                         dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.debug = debug
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                                        dtype=np.float32)
        return bbox
    def __len__(self):
        return len(self.imgs_path)    
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
                i *= 2
        return border // i

    def __getitem__(self, index):
        if os.path.exists(self.imgs_path[index]):
            img = cv2.imread(self.imgs_path[index])
        else:
            print("%s not exists"%self.imgs_path[index])
        anns = np.array(self.words[index])
        bboxes = anns[:, :4]
        bboxes = np.array([self._coco_box_to_bbox(bb) for bb in bboxes])
        lms = np.zeros((anns.shape[0], 10), dtype=np.float32)
        if self.split =="train":
            for idx, ann in enumerate(anns):
                lm = np.zeros(10, dtype=np.float32) - 1
                if ann[4]>=0:
                    for i in range(5):

                        lm[i*2] = ann[4 + 3 * i]
                        lm[i*2+1] = ann[4 + 3 * i + 1]
                lms[idx] = lm
        num_objs = min(len(anns), self.max_objs)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.default_resolution[0], self.default_resolution[1]

        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1
        
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)

        inp1 = inp.copy()
        inp = (inp.astype(np.float32) / 255.)
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        num_classes = 1
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)

        landmarks = np.zeros((self.max_objs, 10), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        lm_reg = np.zeros((self.max_objs, 10), dtype=np.float32)
        lm_ind = np.zeros((self.max_objs), dtype=np.int64)
        lm_mask = np.zeros((self.max_objs), dtype=np.uint8)

        gt_det = []
        cls_id = 0
        for k in range(num_objs):
            flag_lm =  False
            bbox = bboxes[k]
            lm = lms[k]
            bbox1 = bbox.copy()
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                if lm[0] >=0:  
                    lm[0::2] = width - lm[0::2] - 1
                    l_tmp = lm.copy()
                    lm[0:2] = l_tmp[2:4]
                    lm[2:4] = l_tmp[0:2]
                    lm[6:8] = l_tmp[8:10]
                    lm[8:10] = l_tmp[6:8]

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            if lm[0] >= 0:
                lm[:2] = affine_transform(lm[:2], trans_output)
                lm[2:4] = affine_transform(lm[2:4], trans_output)
                lm[4:6] = affine_transform(lm[4:6], trans_output)
                lm[6:8] = affine_transform(lm[6:8], trans_output)
                lm[8:10] = affine_transform(lm[8:10], trans_output)

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w >0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
            
                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = 1. * w, 1. * h

                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                if lm[0]>0 and lm[1]< output_h and lm[2] < output_w and lm[3] < output_h \
                    and lm[6] > 0 and lm[7] > 0 and lm[8] < output_w and lm[9] > 0:

                    lm_ind[k] = ct_int[1] * output_w + ct_int[0]
                    if h*w>10:
                        lm_mask[k] = 1

                    lm_temp = lm.copy()
                    lm_int = lm_temp.astype(np.int32)
                    lm_reg[k] = lm_temp - lm_int
                    lm_temp[[0, 2, 4, 6, 8]] = lm_temp[[0, 2, 4, 6, 8]] - ct_int[0]
                    lm_temp[[1, 3, 5, 7, 9]] =  lm_temp[[1, 3, 5, 7, 9]] - ct_int[1]
                    landmarks[k] = lm_temp
                gt_det.append([4*(ct[0] - w / 2), 4*(ct[1] - h / 2), 
                       4*(ct[0] + w / 2), 4*(ct[1] + h / 2)])
        # if self.debug :# and ("COCO" in str(self.imgs_path[files_index])):
        #     print(len(lms), len(bboxes))
        #     import matplotlib
        #     matplotlib.use('Agg')
        #     import matplotlib.pyplot as plt
        #     for lm, bb in zip(lms, bboxes):
        #         plt.figure(figsize=(50, 50)) 
                
        #         if bb[3] - bb[1] > 0 and bb[2] - bb[0] and np.array(np.where(lm > 0)).shape[1] ==10:
        #             cv2.circle(inp1, (int(lm[0]), int(lm[1])), 2, (255, 0, 0), -1)
        #             cv2.circle(inp1, (int(lm[2]), int(lm[3])), 2, (255, 255, 0), -1)
        #             cv2.circle(inp1, (int(lm[4]), int(lm[5])), 2, (255, 155, 155), -1)
        #             cv2.circle(inp1, (int(lm[6]), int(lm[7])), 2, (255, 0, 255), -1)
        #             cv2.circle(inp1, (int(lm[8]), int(lm[9])), 2, (65, 86, 255), -1)
        #             plt.plot(bb[[0, 2, 2, 0, 0]].T, bb[[1, 1, 3, 3, 1]].T, '.-')
        #     plt.imshow(inp1)
        #     plt.axis('off')
        #     plt.savefig('debug/_after%s'%self.imgs_path[index].split("/")[-1])
        #     time.sleep(10)

        ret = {'input': inp, 'hm': hm, 'lm':landmarks,'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'lm_ind': lm_ind, 'lm_mask': lm_mask}
        
        if not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                   np.zeros((1, 4), dtype=np.float32)
            meta = {'gt_det': gt_det}
            ret['meta'] = meta
        return ret

from torch.utils.data import Dataset, DataLoader
if __name__ == '__main__':

    data = CenterFaceData(txt_path = "/media/hdd/sources/data/data_wider/train/label.txt")
    dataloader_val = DataLoader(data, num_workers=8, batch_size=2,pin_memory=True,
            drop_last=False)
    
    for data in dataloader_val:
        print(data["hm"])