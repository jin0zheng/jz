from vision.ssd.transforms_fox import *
import albumentations as al
import random


# 用的al的数据增强  img shape:(_,_,1)  boxes shape:(:,4) labels shape:(:)
class TrainAugmentation_al_fox:
    def __init__(self, size=[480, 640], mean=127.5, std=1.0, num_classes=None):
        self.mean = mean
        self.size = size
        self.std = std
        self.num_classes = num_classes

    def __call__(self, img, boxes, labels):

        if self.num_classes == 3:
            if random.randint(0, 1):
                _, width, _ = img.shape
                img = img[:, ::-1, :]
                zhongjianbianliang = boxes[:, 0].copy()
                boxes[:, 0] = width - boxes[:, 2]
                boxes[:, 2] = width - zhongjianbianliang
                labels = labels + 1

        # rotate = Rotate(angle_1=180)
        # img, boxes, labels = rotate(img, boxes, labels)  # 出的结果img是三维的
        boxes = boxes.reshape(-1, 4).astype(np.uint64)  # 这一句要加上，要不然可能会超出边界
        labels = labels.reshape(-1, ).tolist()

        if boxes[0, 0] < 0 or boxes[0, 1] < 0:
            print('shape:' + str(img.shape))
            print(boxes)
            return 0
        if boxes[0, 2] > img.shape[1] or boxes[0, 3] > img.shape[0]:
            print('shape:' + str(img.shape))
            print(boxes)
            return 0

        height, width, _ = img.shape
        if height / width > self.size[0] / self.size[1]:
            height_new = self.size[0]
            width_new = int(self.size[0] * width / height)
        else:
            height_new = int(self.size[1] * height / width)
            width_new = self.size[1]

        transform = al.Compose([
            al.Resize(height_new, width_new),
            al.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
            # al.Flip(p=0.5),
            al.PadIfNeeded(min_height=self.size[0], min_width=self.size[1], position='random',
                           border_mode=cv2.BORDER_CONSTANT, value=0)],
            bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        result = transform(image=img, bboxes=boxes, class_labels=labels)

        transformed_image = result['image']  # np.array (h,w,c)
        transformed_bboxes = result['bboxes']  # list len()为box的个数， 每个元素为一个元组（xmin,ymin,xmax,ymax）
        transformed_class_labels = result['class_labels']

        transformed_image = (transformed_image - self.mean) / self.std  # 首先-均值 /方差
        transformed_image = torch.from_numpy(transformed_image.astype(np.float32))  # 转化为tensor
        transformed_image = transformed_image.permute(2, 0, 1)

        transformed_bboxes = torch.tensor(transformed_bboxes)
        transformed_bboxes[:, 0] /= self.size[1]
        transformed_bboxes[:, 2] /= self.size[1]
        transformed_bboxes[:, 1] /= self.size[0]
        transformed_bboxes[:, 3] /= self.size[0]

        transformed_class_labels = torch.tensor(transformed_class_labels).long()

        # 以下三个分别为tensor (c,h,w)  tensor (n,4)  tensor (n)
        return transformed_image, transformed_bboxes, transformed_class_labels


# 根据自己的需求进行resize然后裁剪，确保各个大小的框的比例  img shape:(_,_,1)  boxes shape:(:,4) 相对于原始边长的比例 labels shape:(:)
class TrainAugmentation_al_fox1:
    def __init__(self, size=[480, 640], mean=127.5, std=1.0, num_classes=None, area_radio_range=None):
        self.mean = mean
        self.size = size
        self.std = std
        self.num_classes = num_classes
        self.area_radio_range = area_radio_range

    def __call__(self, img, boxes, labels):
        height, width, _ = img.shape
        labels = labels.reshape(-1, ).tolist()

        #########  首先进行resize，当输入大小和图像一样的时候，这部分可以删了 ##############

        if height / width > self.size[0] / self.size[1]:
            height_new = self.size[0]
            width_new = int(self.size[0] * width / height)
        else:
            height_new = int(self.size[1] * height / width)
            width_new = self.size[1]

        transform = al.Compose([
            al.Resize(height_new, width_new)],
            bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        result = transform(image=img, bboxes=boxes, class_labels=labels)
        img = result['image']
        boxes = result['bboxes']
        labels = result['class_labels']
        boxes = np.array(boxes)

        ################  根据面积的radio对整张图进行扩大或者缩小  #####################


        height, width, _ = img.shape
        area = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])  # 本来多大面积
        area_radio = np.random.random() * (self.area_radio_range[1] - self.area_radio_range[0]) + self.area_radio_range[
            0]  # 要变为占[480,640]多大比例
        side_radio = (area_radio / (area / self.size[0] / self.size[1])) ** 0.5  # 边长变为原来的多少
        # 当一条边很长，另一条边很短，这样面积虽然小，但是当扩大很多倍的时候，长边有可能超过长宽
        if (boxes[0][2] - boxes[0][0]) * side_radio > self.size[1] or (boxes[0][3] - boxes[0][1]) * side_radio > \
                self.size[0]:
            pass
        else:
            height_new, width_new = int(height * side_radio), int(width * side_radio)
            transform = al.Compose([al.Resize(height_new, width_new)],
                                   bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            result = transform(image=img, bboxes=boxes, class_labels=labels)
            img = result['image']
            boxes = result['bboxes']
            labels = result['class_labels']
            boxes = np.array(boxes)

        ##################  然后进行裁剪  #############################
        height, width, _ = img.shape
        xmin, ymin, xmax, ymax = boxes[0]
        # 由于裁剪之后的图片要小于[480, 640]，所以要是图片增大以后，xmax有可能会大于640，因此这个时候就不能从0开始裁剪，而是从xmax - self.size[1]开始
        # 这里+0.1是为了防止math.ceil(xmin+0.1)=0，这样会出错，这个的随机区间是[)
        crop_x_min = np.random.randint(max(0, xmax - self.size[1]), math.ceil(xmin + 0.1))
        crop_x_max = min(crop_x_min + self.size[1], width)
        crop_y_min = np.random.randint(max(0, ymax - self.size[0]), math.ceil(ymin + 0.1))
        crop_y_max = min(crop_y_min + self.size[0], height)
        transform = al.Compose([al.Crop(crop_x_min, crop_y_min, crop_x_max, crop_y_max),
                                al.RandomRotate90(p=1)],
                               bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        result = transform(image=img, bboxes=boxes, class_labels=labels)

        img = result['image']
        boxes = result['bboxes']
        labels = result['class_labels']
        boxes = np.array(boxes)
        labels = np.array(labels)

        if random.randint(0, 1):  # 左右翻转
            _, width, _ = img.shape
            img = img[:, ::-1, :]
            zhongjianbianliang = boxes[:, 0].copy()
            boxes[:, 0] = width - boxes[:, 2]
            boxes[:, 2] = width - zhongjianbianliang
            if self.num_classes == 3:
                labels = labels + 1

        if random.randint(0, 1):  # 上下翻转
            height, width, _ = img.shape
            img = img[::-1, :, :]
            zhongjianbianliang = boxes[:, 1].copy()
            boxes[:, 1] = height - boxes[:, 3]
            boxes[:, 3] = height - zhongjianbianliang
            if self.num_classes == 3:
                labels = labels + 1
                if labels > 2:
                    labels = 1
                labels = np.array(labels)

        # rotate = Rotate(angle_1=180)
        # img, boxes, labels = rotate(img, boxes, labels)  # 出的结果img是三维的
        boxes = boxes.reshape(-1, 4).astype(np.uint64)  # 这一句要加上，要不然可能会超出边界
        labels = labels.reshape(-1, ).tolist()

        if boxes[0, 0] < 0 or boxes[0, 1] < 0:
            print('shape:' + str(img.shape))
            print(boxes)
            return 0
        if boxes[0, 2] > img.shape[1] or boxes[0, 3] > img.shape[0]:
            print('shape:' + str(img.shape))
            print(boxes)
            return 0

        height, width, _ = img.shape

        if height > self.size[0] or width > self.size[1]:    # 因为前面有旋转，所以旋转以后长宽中的一个可能会比原来的长，这个时候需要压缩
            if height / width > self.size[0] / self.size[1]:
                height_new = self.size[0]
                width_new = int(self.size[0] * width / height)
            else:
                height_new = int(self.size[1] * height / width)
                width_new = self.size[1]

            transform = al.Compose([
                al.Resize(height_new, width_new),
                al.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
                # al.Flip(p=0.5),
                al.PadIfNeeded(min_height=self.size[0], min_width=self.size[1], position='random',
                               border_mode=cv2.BORDER_CONSTANT, value=np.random.randint(0, 256))],
                bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        else:   # 要是没有比原来的长就不用压缩了
            transform = al.Compose([
                # al.Resize(height_new, width_new),  # 这里不需要resize！！！！
                al.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
                # al.Flip(p=0.5),
                al.PadIfNeeded(min_height=self.size[0], min_width=self.size[1], position='random',
                               border_mode=cv2.BORDER_CONSTANT, value=np.random.randint(0, 256))],
                bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        result = transform(image=img, bboxes=boxes, class_labels=labels)

        transformed_image = result['image']  # np.array (h,w,c)
        transformed_bboxes = result['bboxes']  # list len()为box的个数， 每个元素为一个元组（xmin,ymin,xmax,ymax）
        transformed_class_labels = result['class_labels']

        transformed_image = (transformed_image - self.mean) / self.std  # 首先-均值 /方差
        transformed_image = torch.from_numpy(transformed_image.astype(np.float32))  # 转化为tensor
        transformed_image = transformed_image.permute(2, 0, 1)

        transformed_bboxes = torch.tensor(transformed_bboxes)
        transformed_bboxes[:, 0] /= self.size[1]
        transformed_bboxes[:, 2] /= self.size[1]
        transformed_bboxes[:, 1] /= self.size[0]
        transformed_bboxes[:, 3] /= self.size[0]

        transformed_class_labels = torch.tensor(transformed_class_labels).long()

        # 以下三个分别为tensor (c,h,w)  tensor (n,4)  tensor (n)
        return transformed_image, transformed_bboxes, transformed_class_labels


class ValAugmentation_al_fox:

    def __init__(self, size=[480, 640], mean=127.5, std=1.0, num_classes=None):
        self.mean = mean
        self.size = size
        self.std = std
        self.num_classes = num_classes

    def __call__(self, img, boxes, labels):

        if self.num_classes == 3:
            if random.randint(0, 1):
                _, width, _ = img.shape
                img = img[:, ::-1, :]
                zhongjianbianliang = boxes[:, 0].copy()
                boxes[:, 0] = width - boxes[:, 2]
                boxes[:, 2] = width - zhongjianbianliang
                labels = labels + 1

        height, width, _ = img.shape

        if height / width > self.size[0] / self.size[1]:
            height_new = self.size[0]
            width_new = int(self.size[0] * width / height)
        else:
            height_new = int(self.size[1] * height / width)
            width_new = self.size[1]

        transform = al.Compose(
            [
                al.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=1),
                al.Resize(height_new, width_new),
                al.PadIfNeeded(min_height=self.size[0], min_width=self.size[1], position='random',
                               border_mode=cv2.BORDER_CONSTANT, value=0)
            ],
            bbox_params=al.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        result = transform(image=img, bboxes=boxes, class_labels=labels)
        transformed_image = result['image']
        transformed_bboxes = result['bboxes']
        transformed_class_labels = result['class_labels']

        transformed_image = (transformed_image - self.mean) / self.std  # 首先-均值 /方差
        transformed_image = torch.from_numpy(transformed_image.astype(np.float32))  # 转化为tensor
        transformed_image = transformed_image.permute(2, 0, 1)

        transformed_bboxes = torch.tensor(transformed_bboxes)
        transformed_bboxes[:, 0] /= self.size[1]
        transformed_bboxes[:, 2] /= self.size[1]
        transformed_bboxes[:, 1] /= self.size[0]
        transformed_bboxes[:, 3] /= self.size[0]

        transformed_class_labels = torch.tensor(transformed_class_labels).long()
        return transformed_image, transformed_bboxes, transformed_class_labels


class PredictionTransform:  # 这里由于只需要对image做处理，不用处理
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            # Resize_pre(size),    # 如果是修改后的网络，这可以删掉
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor_pre()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


#  在荆虹自己写的数据增强
class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([  # 这里面每个都有输入label和box标签，但是大多都没有变
            ConvertFromInts(),  # 把输入图片从INT型转化为float型
            # Rotate(180),     # 随机旋转
            RandomMirror(),  # 镜像
            Resize(self.size),  # 进行resize,size[0]是宽度
            ToPercentCoords(),  # 这个比较有意思，就是把box从绝对值改到了相对于image的比例
            SubtractMeans(self.mean),  # 去均值
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),  # 去标准差
            ToTensor(),  # 然后进行通道顺序变化
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """

        img = img[:, :, np.newaxis]

        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            ToPercentCoords(),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        image = image[:, :, np.newaxis]
        return self.transform(image, boxes, labels)
