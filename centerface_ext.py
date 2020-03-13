import numpy as np
import cv2
import datetime
from model.centernet import ghost_net
import torch
from collections import OrderedDict
from torchvision import transforms as trans
import time
import torch.nn as nn
import torch
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections  


class CenterFace(object):
    mean = np.array([0.485, 0.456, 0.406],
                                     dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225],
                                     dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, height, width, landmarks=True):
        self.landmarks = landmarks
        if self.landmarks:
            self.net = ghost_net()
            self.cuda = True
            if self.cuda:
                self.net.cuda()
            checkpoint = torch.load('weight/model_epoch_145.pt')
            self.net.load_state_dict(checkpoint)
            self.net.eval()
            del checkpoint
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)

    def __call__(self, img, threshold=0.5):
        img = cv2.resize(img, (self.img_w_new, self.img_h_new))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = (img.astype(np.float32) / 255.)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img)
        img = torch.unsqueeze(img, 0)
        begin = datetime.datetime.now()
        if self.cuda:
            img = img.cuda()
        out = self.net(img)[0]
        hm_t, hm_l, hm_b, \
        hm_r, hm_c, hm_ml, \
        hm_mr, hm_n, hm_el, hm_er = torch.clamp(out['hm_t'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_l'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_b'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_r'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_c'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_ml'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_mr'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_n'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_el'].sigmoid_(), min=1e-4, max=1-1e-4), \
                                    torch.clamp(out['hm_er'].sigmoid_(), min=1e-4, max=1-1e-4)

        end = datetime.datetime.now()
        print("cpu times = ", end - begin)
        if self.landmarks:
            dets = self.decode_ext( hm_t, hm_l, hm_b, \
                                    hm_r, hm_c, hm_ml, \
                                    hm_mr, hm_n, hm_el, hm_er)
        else:
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
        dets = dets[0]
        # if len(dets) > 0:
        #     dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] // self.scale_w, dets[:, 1:4:2]//self.scale_h  #// self.scale_w, self.scale_h 
        dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2]*4 // self.scale_w, dets[:, 1:4:2]*4//self.scale_h  #// self.scale_w, self.scale_h 
        return dets.detach().cpu().numpy()
    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]

        if self.opt.fix_res:
          inp_height, inp_width = self.opt.input_h, self.opt.input_w
          c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
          s = max(height, width) * 1.0
        else:
          inp_height = (new_height | self.opt.pad) + 1
          inp_width = (new_width | self.opt.pad) + 1
          c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
          s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
          resized_image, trans_input, (inp_width, inp_height),
          flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
          images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        return images
    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        # print(heatmap.shape)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        print(len(c0))
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = scale0[c0[i], c1[i]]*4, scale1[c0[i], c1[i]]*4
                # s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                print(o0, o1)
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s0 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s1 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])

                boxes.append([x1, y1, min(x1 + s0, size[1]), min(y1 + s1, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2, c0[i], c1[i]]* s0  + x1)
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
    def decode_ext(self, t_heat, l_heat, b_heat, r_heat, ct_heat, \
                 ml_heat, mr_heat, n_heat, el_heat, er_heat,
                t_regr=None, l_regr=None, b_regr=None, r_regr=None, \
                scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, \
                num_dets=50, K=50):
        batch, cat, height, width = t_heat.size()
        # perform nms on heatmaps
        t_heat = _nms(t_heat)
        l_heat = _nms(l_heat)
        b_heat = _nms(b_heat)
        r_heat = _nms(r_heat)
          
        t_heat[t_heat > 1] = 1
        l_heat[l_heat > 1] = 1
        b_heat[b_heat > 1] = 1
        r_heat[r_heat > 1] = 1

        t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heat, K=K)
        l_scores, l_inds, l_clses, l_ys, l_xs = _topk(l_heat, K=K)
        b_scores, b_inds, b_clses, b_ys, b_xs = _topk(b_heat, K=K)
        r_scores, r_inds, r_clses, r_ys, r_xs = _topk(r_heat, K=K)
        t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
        t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
        l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
        l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
        b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
        b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
        r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
        r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

        t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
        l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
        b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
        r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

        box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
        box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()
        ct_inds = t_clses.long() * (height * width) + box_ct_ys * width + box_ct_xs
        ct_inds = ct_inds.view(batch, -1)
        ct_heat = ct_heat.view(batch, -1, 1)
        ct_scores = _gather_feat(ct_heat, ct_inds)

        t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
        l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
        b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
        r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
        ct_scores = ct_scores.view(batch, K, K, K, K)
        scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

        # reject boxes based on classes
        cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + \
                   (t_clses != r_clses)
        cls_inds = (cls_inds > 0)

        top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
        top_inds = (top_inds > 0)
        left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
        left_inds = (left_inds > 0)
        bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
        bottom_inds = (bottom_inds > 0)
        right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
        right_inds = (right_inds > 0)

        sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
                  (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
                  (ct_scores < center_thresh)
        sc_inds = (sc_inds > 0)

        scores = scores - sc_inds.float()
        scores = scores - cls_inds.float()
        scores = scores - top_inds.float()
        scores = scores - left_inds.float()
        scores = scores - bottom_inds.float()
        scores = scores - right_inds.float()

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        if t_regr is not None and l_regr is not None \
          and b_regr is not None and r_regr is not None:
            t_regr = _transpose_and_gather_feat(t_regr, t_inds)
            t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
            l_regr = _transpose_and_gather_feat(l_regr, l_inds)
            l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
            b_regr = _transpose_and_gather_feat(b_regr, b_inds)
            b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
            r_regr = _transpose_and_gather_feat(r_regr, r_inds)
            r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

            t_xs = t_xs + t_regr[..., 0]
            t_ys = t_ys + t_regr[..., 1]
            l_xs = l_xs + l_regr[..., 0]
            l_ys = l_ys + l_regr[..., 1]
            b_xs = b_xs + b_regr[..., 0]
            b_ys = b_ys + b_regr[..., 1]
            r_xs = r_xs + r_regr[..., 0]
            r_ys = r_ys + r_regr[..., 1]
        else:
            t_xs = t_xs + 0.5
            t_ys = t_ys + 0.5
            l_xs = l_xs + 0.5
            l_ys = l_ys + 0.5
            b_xs = b_xs + 0.5
            b_ys = b_ys + 0.5
            r_xs = r_xs + 0.5
            r_ys = r_ys + 0.5
          
        bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)

        clses  = t_clses.contiguous().view(batch, -1, 1)
        clses  = _gather_feat(clses, inds).float()

        t_xs = t_xs.contiguous().view(batch, -1, 1)
        t_xs = _gather_feat(t_xs, inds).float()
        t_ys = t_ys.contiguous().view(batch, -1, 1)
        t_ys = _gather_feat(t_ys, inds).float()
        l_xs = l_xs.contiguous().view(batch, -1, 1)
        l_xs = _gather_feat(l_xs, inds).float()
        l_ys = l_ys.contiguous().view(batch, -1, 1)
        l_ys = _gather_feat(l_ys, inds).float()
        b_xs = b_xs.contiguous().view(batch, -1, 1)
        b_xs = _gather_feat(b_xs, inds).float()
        b_ys = b_ys.contiguous().view(batch, -1, 1)
        b_ys = _gather_feat(b_ys, inds).float()
        r_xs = r_xs.contiguous().view(batch, -1, 1)
        r_xs = _gather_feat(r_xs, inds).float()
        r_ys = r_ys.contiguous().view(batch, -1, 1)
        r_ys = _gather_feat(r_ys, inds).float()
        
        return torch.cat([bboxes, scores], dim=2)

        detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                                b_xs, b_ys, r_xs, r_ys, clses], dim=2)


        return detections

    # def decode_ext(self, t_heat, l_heat, b_heat, r_heat, ct_heat, \
    #              ml_heat, mr_heat, n_heat, el_heat, er_heat,
    #             t_regr=None, l_regr=None, b_regr=None, r_regr=None, \
    #             scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, \
    #             num_dets=1000):
    #     t_heat = np.squeeze(t_heat)
    #     l_heat = np.squeeze(l_heat)
    #     b_heat = np.squeeze(b_heat)
    #     r_heat = np.squeeze(r_heat)
    #     ct_heat = np.squeeze(ct_heat)

    #     ml_heat, mr_heat, n_heat, el_heat, er_heat = np.squeeze(ml_heat), np.squeeze(mr_heat), np.squeeze(n_heat), \
    #                                                 np.squeeze(el_heat), np.squeeze(er_heat)

    #     scores    = (t_heat + l_heat + b_heat + r_heat ) / 4
    #     c0, c1 = np.where(scores > 0.2)
    #     print(len(c0))
    #     if self.landmarks:
    #         boxes, lms = [], []
    #     else:
    #         boxes = []
    #     if len(c0) > 0:
    #         for i in range(len(c0)):
    #             s0, s1 = scale0[c0[i], c1[i]]*4, scale1[c0[i], c1[i]]*4
    #             # s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
    #             o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
    #             print(o0, o1)
    #             s = heatmap[c0[i], c1[i]]
    #             x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s0 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s1 / 2)
    #             x1, y1 = min(x1, size[1]), min(y1, size[0])

    #             boxes.append([x1, y1, min(x1 + s0, size[1]), min(y1 + s1, size[0]), s])
    #             if self.landmarks:
    #                 lm = []
    #                 for j in range(5):
    #                     lm.append(landmark[0, j * 2, c0[i], c1[i]]* s0  + x1)
    #                     lm.append(landmark[0, j * 2+1, c0[i], c1[i]] * s1 + y1)
                        
    #                 lms.append(lm)

    #         boxes = np.asarray(boxes, dtype=np.float32)
    #         keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
    #         boxes = boxes[keep, :]
    #         if self.landmarks:
    #             lms = np.asarray(lms, dtype=np.float32)
    #             lms = lms[keep, :]
    #     if self.landmarks:
    #         return boxes, lms
    #     else:
    #         return boxes
    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep
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
