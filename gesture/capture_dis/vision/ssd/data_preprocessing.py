from ..transforms.transforms import *

class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([     # 这里面每个都有输入label和box标签，但是大多都没有变
            ConvertFromInts(),   # 把输入图片从INT型转化为float型
            PhotometricDistort(),   # RGB转化到HSV  然后做明亮度调节
            # RandomSampleCrop_v2(),   # 随机裁剪，这个就不用了了
            RandomMirror(),       # 镜像，label变
            ToPercentCoords(),    # 这个比较有意思，就是把box从绝对值改到了相对于image的比例
            Resize(self.size),   # 进行resize
            SubtractMeans(self.mean),   #去均值
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),   # 去标准差
            ToTensor(),  # 变到0-1  然后进行通道数变化
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
