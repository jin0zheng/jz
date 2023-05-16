import time

import torch

from ..utils import box_utils
from .data_preprocessing_fox import PredictionTransform
from ..utils.misc import Timer
import numpy as np
import torch.nn.functional as F

class Predictor:
    def __init__(self, net, size, mean, std, nms_method=None,
                 iou_threshold=0.3, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.size = size
        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        start_time = time.time()
        image = self.transform(image)  #
        # print('transform time is '+str(time.time()-start_time))
        images = image.unsqueeze(0)
        images = images.to(self.device)

        with torch.no_grad():
            for i in range(1):
                self.timer.start()
                scores, boxes = self.net.forward(images)
                # print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]

        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []

        # print(scores[:, 1].max())
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]

            if probs.size(0) == 0:
                continue

            # 只把大于pro_threshlod的box挑了出来
            subset_boxes = boxes[mask, :]

            # shape为[N,5]，大于pro_threshlod的box的坐标和可能性
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)

            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)

        # radio是保持读取图片长宽比不变的情况下，如何从predict出来的结果转化到最原始的读取图片
        if height / width > self.size[1] / self.size[0]:
            radio = height / self.size[1]
        else:
            radio = width / self.size[0]

        picked_box_probs[:, 0] *= self.size[0] * radio  # 是的 这里的self.size[0]是宽
        picked_box_probs[:, 1] *= self.size[1] * radio
        picked_box_probs[:, 2] *= self.size[0] * radio
        picked_box_probs[:, 3] *= self.size[1] * radio
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
