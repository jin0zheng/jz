import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import box_utils


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float,就是哪一类的样本少
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    # def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, reduction='mean'):
    def __init__(self, num_class, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        # self.alpha = alpha
        self.gamma = gamma
        # self.smooth = smooth
        self.reduction = reduction

        # if self.alpha is None:
        #     self.alpha = torch.ones(self.num_class, 1)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
        #     self.alpha = self.alpha / self.alpha.sum()
        # elif isinstance(self.alpha, float):
        #     alpha = torch.ones(self.num_class, 1)
        #     alpha = alpha * (1 - self.alpha)
        #     alpha[balance_index] = self.alpha
        #     self.alpha = alpha
        # else:
        #     raise TypeError('Not support alpha type')
        #
        # if self.smooth is not None:
        #     if self.smooth < 0 or self.smooth > 1.0:
        #         raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        # alpha = self.alpha
        # if alpha.device != input.device:
        #     alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        # if self.smooth:
        #     one_hot_key = torch.clamp(
        #         one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        # alpha = alpha[idx]
        # loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        loss = -1 * torch.pow((1 - pt), gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class MultiboxLoss(nn.Module):
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance, device, loss_fuction):
        # neg_pos_ratio:  the ratio between the negative examples and positive examples.
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.loss_function = loss_fuction

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.       网络输出的
            predicted_locations (batch_size, num_priors, 4): predicted locations.      网络输出的
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """

        # labels,gt_locations是读取的标签和位置，confidence和predicted_locations是网络输出的
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]  # 对每一行做softmax然后取log,然后取背景类
            # print(loss.shape,labels.shape)
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)  # [batch,num_priors]  torch.tensor

        # 从[batch_size, num_priors, num_classes] 到 [batch_size*num_priors, num_classes]
        confidence = confidence[mask, :]
        if self.loss_function == 'cross_entropy':
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask],
                                                  reduction='sum')  # 仅仅mask的地儿(所有正样本和三倍负样本)对进行交叉熵

        elif self.loss_function == 'focal_loss':
            focal_loss = FocalLoss(num_class=num_classes, gamma=2, reduction='sum')  # 仅仅mask的地儿(所有正样本和三倍负样本)对进行交叉熵
            classification_loss = focal_loss(confidence.reshape(-1, num_classes), labels[mask])

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')  # smooth_l1_loss
        # smooth_l1_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')  #l2 loss
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


#  自己根据https://blog.csdn.net/qq_33278884/article/details/91572173进行修改,没有使用alpha


# loss包括两部分，其中分类的还是和MultiboxLoss一样，关键点回归自己根据https://blog.csdn.net/z240626191s/article/details/125322139 写的
class Ciou_loss(nn.Module):
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance, device, loss_fuction):
        super(Ciou_loss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = torch.tensor(center_variance).to(device)
        self.size_variance = torch.tensor(size_variance).to(device)
        self.priors = priors.to(device)
        self.loss_function = loss_fuction

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.       网络输出的
            predicted_locations (batch_size, num_priors, 4): predicted locations.      网络输出的
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """

        # labels,gt_locations是读取的标签和位置，confidence和predicted_locations是网络输出的
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]  # 对每一行做softmax然后取-log,然后取背景类，[batch_size, num_priors]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)  # [batch,num_priors]  torch.tensor，
            # 这里输出的是标签为正样本的位置（比如数量为1）以及概率最大的前3个负样本的位置

        # 从[batch_size, num_priors, num_classes] 到 [batch_size*num_priors, num_classes]
        confidence = confidence[mask, :]
        if self.loss_function == 'cross_entropy':
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask],
                                                  reduction='sum')  # 仅仅mask的地儿(所有正样本和三倍负样本)对进行交叉熵
        elif self.loss_function == 'focal_loss':
            focal_loss = FocalLoss(num_class=num_classes, gamma=2, reduction='sum')  # 仅仅mask的地儿(所有正样本和三倍负样本)对进行交叉熵
            classification_loss = focal_loss(confidence.reshape(-1, num_classes), labels[mask])

        pos_mask = labels > 0  # 哪些不是背景类,就是哪些有手，有手的取1,[batch_size, num_priors]

        # 输出为（batchsize,num_priors,4）,其中每一行是(center_x, center_y, w, h)
        predicted_boxes = box_utils.convert_locations_to_boxes(predicted_locations, self.priors, self.center_variance,
                                                               self.size_variance)
        gt_boxes = box_utils.convert_locations_to_boxes(gt_locations, self.priors, self.center_variance,
                                                        self.size_variance)

        predicted_boxes = predicted_boxes[pos_mask, :].reshape(-1, 4)
        gt_boxes = gt_boxes[pos_mask, :].reshape(-1, 4)

        b1_xy = predicted_boxes[:, :2]  # 取预测框的中心点
        b1_wh = predicted_boxes[:, 2:]  # 取预测框的wh
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half  # 求得左上角坐标
        b1_maxes = b1_xy + b1_wh_half  # 求得右下角坐标

        b2_xy = gt_boxes[:, :2]  # 取真实框的中心点
        b2_wh = gt_boxes[:, 2:]  # 取真实框的wh
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half  # 求得左上角坐标
        b2_maxes = b2_xy + b2_wh_half  # 求得右下角坐标

        intersect_mins = torch.max(b1_mins, b2_mins)  # 相交的左上角
        intersect_max = torch.min(b1_maxes, b2_maxes)  # 相交的右下角

        intersect_wh = torch.max(intersect_max - intersect_mins, torch.zeros_like(intersect_max))  # 求得相交的w和h
        intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]  # 相交面积
        b1_area = b1_wh[:, 0] * b1_wh[:, 1]  # 预测框面积
        b2_area = b2_wh[:, 0] * b2_wh[:, 1]  # 真实框面积

        union_area = b1_area + b2_area - intersect_area  # 并集面积
        iou = intersect_area / union_area  # 计算IOU

        # CIOU = IOU - d(b,bgt)/c^2 - αv
        # 计算中心差距d
        center_distance = torch.sum(torch.pow(b1_xy - b2_xy, 2), axis=-1)  # 先求平方再相加，得到欧氏距离axis=-1是最后一个维度操作 d(b,bgt)

        # 计算包含两个框的最小框的左上角和右下角
        closebox_min = torch.min(b1_mins, b2_mins)  # 左上角
        closebox_max = torch.max(b1_maxes, b2_maxes)  # 右下角
        # closebox_wh = torch.max(closebox_max - closebox_min, torch.zeros_like(intersect_max))

        # 计算对角线的距离
        closebox_distance = torch.sum(torch.pow(closebox_max - closebox_min, 2), axis=-1)

        # 计算ciou
        v = torch.pow(torch.atan(b2_wh[:, 0] / (b2_wh[:, 1] + 1e-6)) - torch.atan(b1_wh[:, 0] / (b1_wh[:, 1] + 1e-6)),
                      2) * 4 / math.pow(math.pi, 2)
        alpha = v / (1 - iou + v)
        ciou = iou - center_distance / torch.clamp(closebox_distance, min=1e-8) - alpha * v

        loss_ciou = 1 - ciou
        ciou_loss = torch.sum(loss_ciou)
        num_pos = gt_boxes.size(0)

        return ciou_loss / num_pos, classification_loss / num_pos
