import numpy as np
import torch
import math

# image_mean_test = image_mean = np.array([127, 127, 127])
# image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2

min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]  # 所有anchor的边长，都是正方形，无论网络咋便，anchor都是这么大
source_layer_indexes = [8, 11, 13]  # 从中间提取的哪几层

# 以下的参数随着输入大小的变化，也会变化，以输入图片大小为[480,640]（高度，宽度）为例，以下都为宽、高，这里注意顺序！！
image_size_w_h = [640, 480]  # 宽，高
shrinkage_list = [[8.0, 16.0, 32.0, 64.0], [8.0, 16.0, 32.0, 60.0]]  # 即每个特征层相对于原图缩小的比例，也就是在特征层上的一点投射到原图时的比例
feature_map_w_h_list = [[80, 40, 20, 10], [60, 30, 15, 8]]  # default feature map size
priors = torch.tensor([])  # 求得的先验框，shape (17640,4) 每行为 (x_center, y_center, w, h),无论是位置还是长宽都做了归一化，例如tensor([0.0063, 0.0083, 0.0156, 0.0208])



# 在define_img_size_fox中被调用的,生成先验框的
def generate_priors_fox(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True) -> torch.Tensor:
    priors = []
    priors_w_h = []
    # 有几个特征层，然后一个特征层一个特征层的计算priors,index就是第几个特征层
    for index in range(0, len(feature_map_list[0])):
        # scale_w = image_size[0] / shrinkage_list[0][index]
        # scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) * (shrinkage_list[0][index] / image_size[0])
                y_center = (j + 0.5) * (shrinkage_list[1][index] / image_size[1])

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
                    if [w * image_size[0], h * image_size[1]] not in priors_w_h:
                        priors_w_h.append([w * image_size[0], h * image_size[1]])
    print(priors_w_h)
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:  # 限制范围的上下限
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


# 根据输入的长度、kernal、padding、stride求输出长度
def compute_output(input, kernal, padding, stride):
    return math.floor((input - kernal + 2 * padding) / stride) + 1


# 在define_img_size_fox中被调用的,生成每一层的feature_map
def generate_feature_map_w_h_list_dict_fox(image_size_w_h):
    global source_layer_indexes
    feature_map = [[image_size_w_h[0]], [image_size_w_h[1]]]  # 输入的w,h
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 2))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 2))  # 第1层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第2层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 2))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 2))  # 第3层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第4层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 2))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 2))  # 第5层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第6层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第7层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第8层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 2))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 2))  # 第9层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第10层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第11层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 2))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 2))  # 第12层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 1))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 1))  # 第13层
    feature_map[0].append(compute_output(feature_map[0][-1], 3, 1, 2))
    feature_map[1].append(compute_output(feature_map[1][-1], 3, 1, 2))  # 第extra层
    return feature_map



def define_img_size_fox(img_size_h_w): # img_size_h_w为高、宽，例如[480,640]
    global image_size_w_h, feature_map_w_h_list, priors, source_layer_indexes
    image_size_w_h = [img_size_h_w[1], img_size_h_w[0]]  # 从高、宽变为宽\高

    # 几个特征层的大小
    # feature_map_w_h_list_dict = {128: [[16, 8, 4, 2], [12, 6, 3, 2]],
    #                              160: [[20, 10, 5, 3], [15, 8, 4, 2]],
    #                              320: [[40, 20, 10, 5], [30, 15, 8, 4]],
    #                              480: [[60, 30, 15, 8], [45, 23, 12, 6]],
    #                              640: [[80, 40, 20, 10], [60, 30, 15, 8]],
    #                              1280: [[160, 80, 40, 20], [120, 60, 30, 15]]}
    feature_map_all = generate_feature_map_w_h_list_dict_fox(image_size_w_h)

    feature_map_w_h_list = [[],[]]
    for index in source_layer_indexes:    # 从里面抽取的哪几层特征层
        feature_map_w_h_list[0].append(feature_map_all[0][index])
        feature_map_w_h_list[1].append(feature_map_all[1][index])
    feature_map_w_h_list[0].append(feature_map_all[0][-1])     # 再加上最后一层特征层
    feature_map_w_h_list[1].append(feature_map_all[1][-1])

    priors = []
    for i in range(0, 2):
        item_list = []
        for k in range(0, len(feature_map_w_h_list[i])):
            item_list.append(image_size_w_h[i] / feature_map_w_h_list[i][k])
        shrinkage_list.append(item_list)
    priors = generate_priors_fox(feature_map_w_h_list, shrinkage_list, image_size_w_h, min_boxes)
