# 原来的版本是输入大小是固定的，现在是任何大小都可以，修改前和修改后的都在这里面，唯一需要修改的地方就是create_mb_tiny_fd_predictor中的Predictor中的PredictionTransform
# 中的Resize_pre(size)删掉就好
import os
import time
from math import tan, pi
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn as nn
from efficientnet_lite import build_efficientnet_lite
from vision.ssd.config.fd_config import define_img_size_fox

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
saved_location = []  # 保存的坐标，用来根据前面两个计算下一个，每一个元素是[x1,y1,x2,y2]
if_save = 0  # 保存标志位
out_path = os.path.join(os.getcwd(), 'neg_collected') + "\\"
if not os.path.exists(out_path):
    os.mkdir(out_path)
name_list = os.listdir(out_path)
if_end = 0
num = 0  # 编号
name = "lin1"
print(os.path.exists(out_path + name))
if not os.path.exists(out_path + name):
    os.mkdir(out_path + '/' + name)
    os.mkdir(out_path + '/' + name + '/check')
    os.mkdir(out_path + '/' + name + '/image')
    os.mkdir(out_path + '/' + name + '/anno')

interval = 7


# interval = input('please input the interval time (usually 8)')
# interval = int(interval)

def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3, 3]).T  #
    T = np.array(extrinsics.translation)
    return (R, T)


def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                     [0, 0, 1]])


def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])


# Set up a mutex to share data between threads
from threading import Lock

frame_mutex = Lock()
frame_data = {"left": None, "right": None, "timestamp_ms": None}
# input_size = [480, 592]
# input_size = [400, 424]
input_size = [800, 848]
cla_size = [224, 224]  # (高，宽)
classify_mean = 0  # 预处理的时候的均值
classify_std = 1  # 预处理的时候的标注差
prob_threshold = 0.6
candidate_size = 1500
define_img_size_fox(input_size)
from vision.ssd.mb_tiny_fd_fox import create_mb_tiny_fd_fox, create_mb_tiny_fd_predictor  # 在这里就是用的自己的网络

# test_device = 'cuda:0'
test_device = 'cpu'
class_names = ['background', 'hand']
model_path = 'det_models/slim-Epoch-6270-Loss-1.3399791325374917.pth'
base_channel = 12  # 这里只能选择12或者16
checkpoint_folder = 'classify_models'
resume_model = 'Epoch-820-1.3119984865188599.pth'
expand_radio = 0.3  # 每条边往外扩多少

net = create_mb_tiny_fd_fox(len(class_names), is_test=True, device=test_device, base_channel=base_channel)
net.load(model_path)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device,
                                        image_size=[input_size[1], input_size[0]],
                                        image_mean=127.5, image_std=1)
model_name = 'efficientnet_lite0'
num_classes = 29
min_range = 30  # 比如说
model = build_efficientnet_lite(model_name, num_classes + 4 + 2)  # 增加了四个数字用做框的回归
model = model.to(test_device)
checkpoint = torch.load(checkpoint_folder + '/' + resume_model, map_location=torch.device('cpu'))


# checkpoint = torch.load(checkpoint_folder + '/' + resume_model, map_location=None)


def load_checkpoint(net, checkpoint):
    from collections import OrderedDict
    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        # k2 = 'module.' + k if not k.startswith('module.') else k
        k2 = k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]
    net.load_state_dict(temp, strict=True)


def load_checkpoint1(net, checkpoint):
    from collections import OrderedDict
    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        # k2 = 'module.' + k if not k.startswith('module.') else k
        k2 = k[7:] if k.startswith('module.') else k
        temp[k2] = checkpoint[k]
    net.load_state_dict(temp, strict=True)


use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
if use_gpu:
    model = nn.DataParallel(model)

print('load model')
checkpoint = torch.load(checkpoint_folder + '/' + resume_model, map_location='cpu')
load_checkpoint1(model, checkpoint)

model.eval()


def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()


pipe = rs.pipeline()
cfg = rs.config()
pipe.start(cfg, callback)
try:
    # Set up an OpenCV window to visualize the results
    min_disp = 0
    # must be divisible by 16
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp

    profiles = pipe.get_active_profile()
    streams = {"left": profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
               "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics = {"left": streams["left"].get_intrinsics(),
                  "right": streams["right"].get_intrinsics()}

    # Print information about both cameras
    print("Left camera:", intrinsics["left"])
    print("Right camera:", intrinsics["right"])

    # Translate the intrinsics from librealsense into OpenCV
    K_left = camera_matrix(intrinsics["left"])
    D_left = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics(streams["left"], streams["right"])

    stereo_fov_rad = 90 * (pi / 180)  # 90 degree desired fov
    stereo_height_px = 480  # 300x300 pixel stereo output   # 这里需要修改一下
    stereo_height_px = 800  # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px / 2 / tan(stereo_fov_rad / 2)
    R_left = np.eye(3)

    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    # print('image size, height:' + str(stereo_height_px) + ', width:' + str(stereo_width_px))
    stereo_cx = (stereo_height_px - 1) / 2 + max_disp
    stereo_cy = (stereo_height_px - 1) / 2
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0, 0, 1, 0]])

    m1type = cv2.CV_32FC1
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    undistort_rectify = {"left": (lm1, lm2)}

    while True:
        # Check if the camera has acquired any frames
        frame_mutex.acquire()
        valid = frame_data["timestamp_ms"] is not None
        frame_mutex.release()

        # If frames are ready to process
        if valid:
            # Hold the mutex only long enough to copy the stereo frames
            frame_mutex.acquire()
            frame_copy = {"left": frame_data["left"].copy()}
            ori_img = frame_copy["left"]
            frame_mutex.release()
            ori_img = cv2.resize(ori_img, (input_size[1], input_size[0]))
            img = ori_img.copy()

            ################## 从这里开始追踪 ######################
            if_exist_hand = 0  # 在大框里面有没有可能有手
            img = img[:, :, np.newaxis]
            if len(saved_location) < 2:
                predict_start_time = time.time()
                with torch.no_grad():
                    boxes, labels, probs = predictor.predict(img, candidate_size / 2, prob_threshold)
                if probs.shape[0] == 0:
                    saved_location = []
                else:
                    if_exist_hand = 1
                    index = torch.argmax(probs)
                    box = boxes[index, :].numpy()
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            else:
                if_exist_hand = 1
                previous_location0 = saved_location[0]
                previous_location1 = saved_location[1]
                x1, y1, x2, y2 = 2 * np.array(saved_location[1]) - np.array(saved_location[0])

            # 如果在大框里面可能存在手
            if if_exist_hand:
                # if x1 > input_size[1] - min_range or y1 > input_size[0] - min_range or x2 < min_range or y2 < min_range:
                #     continue
                if x2 - x1 < min_range or y2 - y1 < min_range:
                    saved_location = []
                    continue

                ############## 扩边
                # 首先计算出扩边以后的长宽，长宽比等于cla_size的长宽比
                width = int((x2 - x1) * (1 + expand_radio))
                height = int((y2 - y1) * (1 + expand_radio))
                if height / width >= cla_size[0] / cla_size[1]:
                    width = int(height / cla_size[0] * cla_size[1])
                elif height / width < cla_size[0] / cla_size[1]:
                    height = int(cla_size[0] / cla_size[1] * width)

                # 然后看看在不在图像范围内
                x1_new = int((x2 + x1) / 2 - width / 2)
                x2_new = int((x2 + x1) / 2 + width / 2)
                if x1_new < 0:
                    x1_new = 0
                    x2_new = width - 1
                elif x2_new >= input_size[1]:
                    x2_new = input_size[1] - 1
                    x1_new = x2_new - (width - 1)

                y1_new = int((y1 + y2) / 2 - height / 2)
                y2_new = int((y1 + y2) / 2 + height / 2)

                if y1_new < 0:
                    y1_new = 0
                    y2_new = height - 1
                elif y2_new >= input_size[0]:
                    y2_new = input_size[0] - 1
                    y1_new = y2_new - (height - 1)

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                crop_img = img[y1_new:(y2_new + 1), x1_new:(x2_new + 1), :]
                # cv2.imshow('111111', crop_img)

                crop_img_input = cv2.resize(crop_img, (cla_size[1], cla_size[0]))
                # cv2.imshow('222222', crop_img_input)

                crop_img_input = crop_img_input / np.std(crop_img_input) * classify_std
                crop_img_input = crop_img_input - np.mean(crop_img_input) + classify_mean
                crop_img_input1 = torch.tensor(crop_img_input).to(test_device)
                crop_img_input1 = crop_img_input1.permute(2, 0, 1).unsqueeze(0).float()
                cal_result = model(crop_img_input1)

                # 是不是手
                ifhand = cal_result[0][0]
                ifhand = nn.Sigmoid()(ifhand)
                pre_x1 = pre_y1 = pre_x2 = pre_y2 = 0  # 每一步先做一个初始化，这样保存数据的时候好找那些没有手

                if ifhand < 0.5:
                    saved_location = []
                else:
                    if len(saved_location) == 2:
                        saved_location.pop(0)

                    # 是不是右手
                    ifright = cal_result[0][1]
                    ifright = nn.Sigmoid()(ifright)
                    if ifright > 0.5:
                        out_txt = 'right '
                    else:
                        out_txt = 'left '

                    # label
                    index = torch.argmax(cal_result[0][2:-4])
                    index = index.cpu().numpy()
                    out_txt = out_txt + str(index)

                    # location
                    pre_anno = cal_result[0][-4:]
                    pre_anno = nn.ReLU6(inplace=True)(pre_anno) / 6
                    pre_x = pre_anno[0] * width
                    pre_y = pre_anno[1] * height
                    pre_w = pre_anno[2] * width
                    pre_h = pre_anno[3] * height

                    pre_x1 = int(pre_x - pre_w / 2) + x1_new
                    pre_x2 = int(pre_x + pre_w / 2) + x1_new
                    pre_y1 = int(pre_y - pre_h / 2) + y1_new
                    pre_y2 = int(pre_y + pre_h / 2) + y1_new

                    saved_location.append([pre_x1, pre_y1, pre_x2, pre_y2])

                    cv2.putText(img, out_txt, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(img, (pre_x1, pre_y1), (pre_x2, pre_y2), (255, 255, 255), 2)
                    # cv2.putText(img, str(probs[0].numpy()), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.7, (255, 0, 0), 2)  # 画上预测出来的框

            cv2.imshow('1', img)
            key = cv2.waitKey(1 + interval)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            if key == ord('s'):
                if if_save == 0:
                    if_save = 1;

                    print('start saving %s' % num)
                else:
                    if_save = 0
                    print('stop saving %s' % num)
                    num += 1

            if if_save:
                save_name = 'single_' + str(num) + "_" + str(int(time.time() * 1000))
                cv2.imwrite(out_path + '/' + name + '/check/' + save_name + '.jpg', img)
                cv2.imwrite(out_path + '/' + name + '/image/' + save_name + '.jpg', ori_img)
                np.savetxt(out_path + '/' + name + '/anno/' + save_name + '.txt',
                           np.array([pre_x1, pre_y1, pre_x2, pre_y2]), fmt='%d', delimiter=' ')


finally:
    pipe.stop()
