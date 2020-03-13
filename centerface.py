import numpy as np
import cv2
import datetime
from model.centernet import efficientnet_b0
import torch
from collections import OrderedDict
from torchvision import transforms as trans
import time


class CenterFace(object):
    mean = np.array([0.408, 0.447, 0.470],
                                     dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.289, 0.274, 0.278],
                                     dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, height, width, landmarks=True):
        self.landmarks = landmarks
        if self.landmarks:
            self.net = efficientnet_b0()
            self.cuda = True
            if self.cuda:
                self.net.cuda()
            checkpoint = torch.load('weight/model_epoch_150.pt')
            self.net.load_state_dict(checkpoint)
            self.net.eval()
            del checkpoint
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)

    def __call__(self, img, threshold=0.2):
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
        if self.landmarks:
            heatmap, scale, offset, lms = torch.clamp(out['hm'].sigmoid_(), min=1e-4, max=1-1e-4).detach().cpu().numpy(), out['wh'].detach().cpu().numpy(),\
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
        # output = self.model(images)[-1]
        # hm = output['hm'].sigmoid_()
        # wh = output['wh']
        # reg = output['reg'] if self.opt.reg_offset else None
        # if self.opt.flip_test:
        # hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        # wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        # reg = reg[0:1] if reg is not None else None
        # torch.cuda.synchronize()
        # forward_time = time.time()
        # dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
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
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > 0.3)
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
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + 0.5) * 4 - s0 / 2), max(0, (c0[i]  + 0.5) * 4 - s1 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])

                boxes.append([x1, y1, min(x1 + s0, size[1]), min(y1 + s1, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        # lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + x1) 
                        # lm.append(landmark[0, j * 2+1, c0[i], c1[i]] * s1 + y1)
                        lm.append((landmark[0, j * 2, c0[i], c1[i]] + c1[i] + 0.5)*4) #+ x1) 
                        lm.append((landmark[0, j * 2+1, c0[i], c1[i]] + c0[i] + 0.5)*4)
                        
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
