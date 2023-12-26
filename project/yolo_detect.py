# coding:utf-8
import cv2
import json
import time
import os
import torch
import numpy as np
from Project.camera import LoadStreams, LoadImages
import sys
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
import random

class Darknet(object):
    """docstring for Darknet"""
    def __init__(self, opt):
        self.opt = opt
        self.device = select_device(self.opt["device"])
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.opt["weights"], map_location=self.device)
        self.stride = int(self.model.stride.max()) 
        self.model.to(self.device).eval()
        # self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.names = ['face', 'smoke', 'phone', 'drink']
        # self.names = ['face', 'phone', 'smoke', 'drink']
        # print(self.names)
        if self.half: self.model.half()
        self.source = self.opt["source"]
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    
    def preprocess(self, img):
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def detect(self, dataset):

        root_dir = self.opt["result_save_path"]
        t0 = time.time()

        for path, img, img_stream, vid_cap in dataset:

            img = self.preprocess(img)
            t1 = time.time()
            pred = self.model(img, augment=self.opt["augment"])[0]  # 0.22s

            pred = pred.float()
            pred = non_max_suppression(pred, self.opt["conf_thres"], self.opt["iou_thres"])
            t2 = time.time()

            pred_boxes = []
            for i, det in enumerate(pred):
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, img_stream[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', img_stream, getattr(dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                # print(det)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls_id in det:
                        print(cls_id)
                        lbl = self.names[int(cls_id)]
                        xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        score = round(conf.tolist(), 3)
                        label = "{}: {}".format(lbl, score)
                        print(label)
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        pred_boxes.append((x1, y1, x2, y2, lbl, score))
                        
                        self.plot_one_box(xyxy, im0, color=(255, 212, 123), label=label)
                        stream_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                        img_save_path = os.path.join(root_dir, str(stream_name) + '.jpg')
                        # print(img_save_path)
                        # time.sleep(2)
                        cv2.imwrite(img_save_path,cv2.resize(im0, (800, 600)))
                        
                else:
                    print("No object in camera")     

    # Plotting functions
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    with open('path/to/config.json', 'r', encoding='utf8') as fp:
        opt = json.load(fp)
        print('[INFO] YOLOv5 Config:', opt)
    darknet = Darknet(opt)
    
    if darknet.webcam:
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)
    else:
        dataset = LoadImages(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)
    darknet.detect(dataset)
    # cv2.destroyAllWindows()
