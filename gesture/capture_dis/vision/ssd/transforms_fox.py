# 主要来自于ultraface的transform文件，然后在上面进行修改

import cv2
import numpy as np
from numpy import random
import math
import torch


# 用于组合各种变化
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


# 把img图像从int转换到float32类型
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


# 把img图像减去平均值,这就直接赋值，没有用到传进来的参数
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


# 变成tensor，范围0-1  然后进行通道顺序变化
class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes.astype(np.float32), labels


# 变成tensor，范围0-1  然后进行通道顺序变化
class ToTensor_pre(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


# 把img图像除以标准差
class imgprocess(object):
    def __init__(self, std):
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= self.std
        return image.astype(np.float32), boxes, labels


# 从相对坐标到绝对坐标
class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels


# 从绝对坐标到相对坐标
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels


# resize，一目了然，但是要注意这里的size是（宽度，高度）
# 比如一个矩阵a的shape为（200，100），b=cv2.resize(a,(100,50)) b.shape为(50,100)
# 这里的输入的size就是[128,96],也就是size[0]就是输出图像的宽度！
class Resize(object):
    def __init__(self, size=(300, 300)):
        self.size = size
        # print('size:', size)

    def __call__(self, image, boxes=1, labels=None):
        height, width, _ = image.shape
        image1 = np.zeros((self.size[1], self.size[0]))
        # width_new和height_new是长宽不变条件下缩放后的宽和高
        if height / width > self.size[1] / self.size[0]:
            height_new = self.size[1]
            width_new = int(self.size[1] * width / height)
        else:
            height_new = int(self.size[0] * height / width)
            width_new = self.size[0]

        #   输入的image为三维，输出的为2维
        image = cv2.resize(image, (width_new, height_new))

        for i in range(height_new):
            for j in range(width_new):
                image1[i, j] = image[i, j]
        image1 = image1[:, :, np.newaxis]

        boxes[:, 0] = boxes[:, 0] / width * width_new
        boxes[:, 2] = boxes[:, 2] / height * height_new
        boxes[:, 1] = boxes[:, 1] / width * width_new
        boxes[:, 3] = boxes[:, 3] / height * height_new

        return image1, boxes, labels


class Resize_pre(object):
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        image1 = np.zeros((self.size[1], self.size[0]))
        # width_new和height_new是长宽不变条件下缩放后的宽和高
        if height / width > self.size[1] / self.size[0]:
            height_new = self.size[1]
            width_new = int(self.size[1] * width / height)
        else:
            height_new = int(self.size[0] * height / width)
            width_new = self.size[0]
        image = cv2.resize(image, (width_new, height_new))
        for i in range(height_new):
            for j in range(width_new):
                image1[i, j] = image[i, j]
        image1 = image1[:, :, np.newaxis]

        return image1, boxes, labels


# 整体的数据都乘以一个数
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


# 整体的数据都加减一个数
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


# 在左面和上面加入黑边，整体大小也就会变化
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


# 这个是自己写的，旋转矩阵
class Rotate(object):
    def __init__(self, angle_1=180):
        self.angle_1 = angle_1

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            return image, boxes, labels

        self.angle = np.random.uniform(-self.angle_1, self.angle_1)

        # 先进行旋转，这个就是从零点进行旋转
        matrix = np.eye(3)
        matrix[0, 0] = math.cos(self.angle / 180 * math.pi)
        matrix[1, 1] = math.cos(self.angle / 180 * math.pi)
        matrix[0, 1] = math.sin(self.angle / 180 * math.pi)
        matrix[1, 0] = -math.sin(self.angle / 180 * math.pi)

        # 然后进行平移
        translation_matrix = np.eye(3)
        width = image.shape[1]
        height = image.shape[0]
        # 看四个顶点旋转后的位置，然后取最大最小值
        vertex = np.array([[0, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1], [width - 1, 0, 1]])
        heng_zuobiao = []  # 四个顶点的横坐标
        zong_zuobiao = []
        for i in range(4):
            heng_zuobiao = heng_zuobiao + [(matrix @ vertex[i])[0]]
            zong_zuobiao = zong_zuobiao + [(matrix @ vertex[i])[1]]
        heng_min = min(heng_zuobiao)
        zong_min = min(zong_zuobiao)
        heng_changdu = max(heng_zuobiao) - min(heng_zuobiao) + 1
        zong_changdu = max(zong_zuobiao) - min(zong_zuobiao) + 1
        translation_matrix[0, 2] = -heng_min
        translation_matrix[1, 2] = -zong_min

        # 最后的旋转矩阵
        matrix = translation_matrix @ matrix
        matrix = matrix[:2]
        image = cv2.warpAffine(image, matrix, (int(heng_changdu), int(zong_changdu)),
                               borderValue=np.random.randint(0, 256))

        # boxes也会变，所以也要调整
        if boxes.any():
            boxes_new = []
            for i in range(boxes.shape[0]):
                vertex = np.array(
                    [[boxes[i, 0], boxes[i, 1], 1],
                     [boxes[i, 0], boxes[i, 3], 1],
                     [boxes[i, 2], boxes[i, 1], 1],
                     [boxes[i, 2], boxes[i, 3], 1]])  # 四个顶点原始坐标
                heng_zuobiao = []  # 变换之后四个顶点的横坐标
                zong_zuobiao = []
                for i in range(4):
                    heng_zuobiao = heng_zuobiao + [(matrix @ vertex[i])[0]]
                    zong_zuobiao = zong_zuobiao + [(matrix @ vertex[i])[1]]
                heng_min = min(heng_zuobiao)
                zong_min = min(zong_zuobiao)
                heng_max = max(heng_zuobiao)
                zong_max = max(zong_zuobiao)
                boxes_new.append([heng_min, zong_min, heng_max, zong_max])  # 这里的顺序就是xmin,ymin,xmax,ymax
            boxes_new = np.array(boxes_new)

        image = image[:, :, np.newaxis]
        return image, boxes_new, labels


# 随机镜面翻转
class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


# # 裁剪中的某个函数
# def intersect(box_a, box_b):
#     max_xy = np.minimum(box_a[:, 2:], box_b[2:])
#     min_xy = np.maximum(box_a[:, :2], box_b[:2])
#     inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
#     return inter[:, 0] * inter[:, 1]
#
# # 裁剪中的某个函数
# def jaccard_numpy(box_a, box_b):
#     """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#     is simply the intersection over union of two boxes.
#     E.g.:
#         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#     Args:
#         box_a: Multiple bounding boxes, Shape: [num_boxes,4]
#         box_b: Single bounding box, Shape: [4]
#     Return:
#         jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
#     """
#     inter = intersect(box_a, box_b)
#     area_a = ((box_a[:, 2] - box_a[:, 0]) *
#               (box_a[:, 3] - box_a[:, 1]))  # [A,B]
#     area_b = ((box_b[2] - box_b[0]) *
#               (box_b[3] - box_b[1]))  # [A,B]
#     union = area_a + area_b - inter
#     return inter / union  # [A,B]
#
# # 裁剪中的某个函数
# def object_converage_numpy(box_a, box_b):
#     """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#     is simply the intersection over union of two boxes.
#     E.g.:
#         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#     Args:
#         box_a: Multiple bounding boxes, Shape: [num_boxes,4]
#         box_b: Single bounding box, Shape: [4]
#     Return:
#         jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
#     """
#     inter = intersect(box_a, box_b)
#     area_a = ((box_a[:, 2] - box_a[:, 0]) *
#               (box_a[:, 3] - box_a[:, 1]))  # [A,B]
#     area_b = ((box_b[2] - box_b[0]) *
#               (box_b[3] - box_b[1]))  # [A,B]
#     return inter / area_a  # [A,B]
#
# # 还是这个好用一些，但是我看了一下感觉也不大好，因为这个是对有很多大大小小的人脸那种图用于裁剪，这里不大适合，因为手势都比较大
# # 而且只有一个，很可能裁剪一下，然后就没有了~
# class RandomSampleCrop(object):
#     """Crop
#     Arguments:
#         img (Image): the image being input during training
#         boxes (Tensor): the original bounding boxes in pt form
#         labels (Tensor): the class labels for each bbox
#         mode (float tuple): the min and max jaccard overlaps
#     Return:
#         (img, boxes, classes)
#             img (Image): the cropped image
#             boxes (Tensor): the adjusted bounding boxes in pt form
#             labels (Tensor): the class labels for each bbox
#     """
#
#     def __init__(self):
#         self.sample_options = (
#             # using entire original input image
#             None,
#             # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
#             (0.1, None),
#             (0.3, None),
#             (0.7, None),
#             (0.9, None),
#             # randomly sample a patch
#             (None, None),
#         )
#
#     def __call__(self, image, boxes=None, labels=None):
#         height, width, _ = image.shape
#         while True:
#             # randomly choose a mode
#             mode = random.choice(self.sample_options)
#             if mode is None:
#                 return image, boxes, labels
#
#             min_iou, max_iou = mode
#             if min_iou is None:
#                 min_iou = float('-inf')
#             if max_iou is None:
#                 max_iou = float('inf')
#
#             # max trails (50)
#             for _ in range(50):
#                 current_image = image
#
#                 w = random.uniform(0.3 * width, width)
#                 h = random.uniform(0.3 * height, height)
#
#                 # aspect ratio constraint b/t .5 & 2
#                 if h / w < 0.5 or h / w > 2:
#                     continue
#
#                 left = random.uniform(width - w)
#                 top = random.uniform(height - h)
#
#                 # convert to integer rect x1,y1,x2,y2
#                 rect = np.array([int(left), int(top), int(left + w), int(top + h)])
#
#                 # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
#                 overlap = jaccard_numpy(boxes, rect)
#
#                 # is min and max overlap constraint satisfied? if not try again
#                 if overlap.max() < min_iou or overlap.min() > max_iou:
#                     continue
#
#                 # cut the crop from the image
#                 current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
#                                 :]
#
#                 # keep overlap with gt box IF center in sampled patch
#                 centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
#
#                 # mask in all gt boxes that above and to the left of centers
#                 m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
#
#                 # mask in all gt boxes that under and to the right of centers
#                 m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
#
#                 # mask in that both m1 and m2 are true
#                 mask = m1 * m2
#
#                 # have any valid boxes? try again if not
#                 if not mask.any():
#                     continue
#
#                 # take only matching gt boxes
#                 current_boxes = boxes[mask, :].copy()
#
#                 # take only matching gt labels
#                 current_labels = labels[mask]
#
#                 # should we use the box left and top corner or the crop's
#                 current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
#                                                   rect[:2])
#                 # adjust to crop (by substracting crop's left,top)
#                 current_boxes[:, :2] -= rect[:2]
#
#                 current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
#                                                   rect[2:])
#                 # adjust to crop (by substracting crop's left,top)
#                 current_boxes[:, 2:] -= rect[:2]
#
#                 return current_image, current_boxes, current_labels
#
# # 这个应该是ultralface作者自己改的，自己改的并不好，感觉作者就不想用这个模块，所以就直接用原版的
# class RandomSampleCrop_v2(object):
#     """Crop
#     Arguments:
#         img (Image): the image being input during training
#         boxes (Tensor): the original bounding boxes in pt form
#         labels (Tensor): the class labels for each bbox
#         mode (float tuple): the min and max jaccard overlaps
#     Return:
#         (img, boxes, classes)
#             img (Image): the cropped image
#             boxes (Tensor): the adjusted bounding boxes in pt form
#             labels (Tensor): the class labels for each bbox
#     """
#
#     def __init__(self):
#         self.sample_options = (
#             # using entire original input image
#             None,
#             # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
#
#             # randomly sample a patch。  这几个都一样是几个意思？
#             (1, None),
#             (1, None),
#             (1, None),
#             (1, None),
#         )
#
#     def __call__(self, image, boxes=None, labels=None):
#         height, width, _ = image.shape
#         while True:
#             # randomly choose a mode
#             mode = random.choice(self.sample_options)
#             if mode is None:
#                 return image, boxes, labels
#
#             min_iou, max_iou = mode
#             if min_iou is None:
#                 min_iou = float('-inf')
#             if max_iou is None:
#                 max_iou = float('inf')
#
#             # max trails (50)
#             for _ in range(50):
#                 current_image = image
#
#                 w = random.uniform(0.3 * width, width)
#                 h = random.uniform(0.3 * height, height)
#
#                 # aspect ratio constraint b/t .5 & 2   特别就是这句，必须要h和w相等才可以，这不是在扯犊子吗？
#                 if h / w != 1:
#                     continue
#                 left = random.uniform(width - w)
#                 top = random.uniform(height - h)
#
#                 # convert to integer rect x1,y1,x2,y2
#                 rect = np.array([int(left), int(top), int(left + w), int(top + h)])
#
#                 # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
#                 overlap = object_converage_numpy(boxes, rect)
#
#                 # is min and max overlap constraint satisfied? if not try again
#                 if overlap.max() < min_iou or overlap.min() > max_iou:
#                     continue
#
#                 # cut the crop from the image
#                 current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
#                                 :]
#
#                 # keep overlap with gt box IF center in sampled patch
#                 centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
#
#                 # mask in all gt boxes that above and to the left of centers
#                 m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
#
#                 # mask in all gt boxes that under and to the right of centers
#                 m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
#
#                 # mask in that both m1 and m2 are true
#                 mask = m1 * m2
#
#                 # have any valid boxes? try again if not
#                 if not mask.any():
#                     continue
#
#                 # take only matching gt boxes
#                 current_boxes = boxes[mask, :].copy()
#
#                 # take only matching gt labels
#                 current_labels = labels[mask]
#
#                 # should we use the box left and top corner or the crop's
#                 current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
#                                                   rect[:2])
#                 # adjust to crop (by substracting crop's left,top)
#                 current_boxes[:, :2] -= rect[:2]
#
#                 current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
#                                                   rect[2:])
#                 # adjust to crop (by substracting crop's left,top)
#                 current_boxes[:, 2:] -= rect[:2]
#
#                 return current_image, current_boxes, current_labels
