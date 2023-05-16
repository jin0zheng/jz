from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch.nn as nn
import torch.nn.functional as F
from vision.ssd.config import fd_config as config
from vision.ssd.predictor_fox import Predictor
from vision.ssd.ssd import SSD, SSD_fox


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


# 主干网络
class Mb_Tiny_fox(nn.Module):
    def __init__(self, num_classes=2, base_channel=None):
        super(Mb_Tiny_fox, self).__init__()
        # self.base_channel = 8 * 2  原始是16
        self.base_channel = base_channel

        # self.base_channel = 8 * 2  # 原始是16

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),  # 这里就是先做1*1
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(1, self.base_channel, 2),  # 输入时候设为单通道，以下全都是按照原始输入为480*640算的，240*320  1
            conv_dw(self.base_channel, self.base_channel * 2, 1),     # 240*320 2
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 120*160 3
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),  # 120*160 4
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 60*80 5
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),  # 60*80 6
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),  # 60*80 7
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),  # 60*80 8
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 30*40 9
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),  # 30*40 10
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),  # 30*40 11
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8 12
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1)  # 10*8 13
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


#
def create_mb_tiny_fd_fox(num_classes, is_test=False, device="cuda", base_channel=None):
    base_net = Mb_Tiny_fox(num_classes, base_channel)  # 这里的这个num_classes其实没用到
    base_net_model = base_net.model  # disable dropout layer

    # 抽出哪几行来
    source_layer_indexes = config.source_layer_indexes
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=base_net.base_channel * 16, out_channels=base_net.base_channel * 4, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=base_net.base_channel * 16,
                            kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD(num_classes,   # 几个类
               base_net_model,  # 主干网络
               source_layer_indexes,  # 在提取其中的哪几层连同最后一层一起开始检测
               extras,
               classification_headers,
               regression_headers,
               is_test=is_test, config=config, device=device)


#  自己修改的网络！！！
def create_mb_tiny_fd_fox(num_classes, is_test=False,device="cuda", base_channel=None):
    base_net = Mb_Tiny_fox(num_classes, base_channel)
    base_net_model = base_net.model  # disable dropout layer,是主干网络

    # 抽出哪几行来
    source_layer_indexes = [
        8,
        11,
        13
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=base_net.base_channel * 16, out_channels=base_net.base_channel * 4, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=base_net.base_channel * 16,
                            kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD_fox(num_classes, base_net_model, source_layer_indexes,
               extras, classification_headers, regression_headers, device=device,config=config,is_test=is_test)


def create_mb_tiny_fd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None, image_size=None,
                                image_mean=None, image_std=None):
    predictor = Predictor(net, image_size, image_mean,
                          image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)

    return predictor
