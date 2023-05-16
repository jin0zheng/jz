# 用来采集背景图像
import os.path
import sys
import threading
import time
from math import tan, pi
from threading import Lock
import cv2
import numpy as np
import pyrealsense2 as rs
from pynput.keyboard import Listener, Key

add_time = 100  # 增加单次采集的间隔，单位ms
if_save = 0  # 存储标志，初始设置为0
n = 0  # 手势起始编号

# 创建存储路径
path = './background_1'
if not os.path.exists(path):
    os.mkdir(path)

height = 800  # 输出的图片的高度
frame_mutex = Lock()
frame_data = {"left": None, "right": None, "timestamp_ms": None}


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


def resize_fox(image, size):  # 这里的size[0]是宽！！！  注意一下通道
    height, width, _ = image.shape
    image1 = np.zeros((size[1], size[0], 3))
    # width_new和height_new是长宽不变条件下缩放后的宽和高
    if height / width > size[1] / size[0]:
        height_new = size[1]
        width_new = int(size[1] * width / height)
    else:
        height_new = int(size[0] * height / width)
        width_new = size[0]
    image = cv2.resize(image, (width_new, height_new))

    image1[0:height_new, 0:width_new, :] = image
    return image1


pipe = rs.pipeline()
cfg = rs.config()
pipe.start(cfg, callback)


def on_release(key):
    global if_save, n
    frame_mutex.acquire()
    if key == Key.space:
        if_save = abs(if_save) - 1
        if if_save:
            print('start saving')
        else:
            print('stop saving')
    if key == Key.esc:
        print("end")
        if_save = " "
    frame_mutex.release()


def listener_():
    with Listener(on_release=on_release) as listener:
        listener.join()


thread = threading.Thread(target=listener_, args=())
thread.start()
try:
    stereo_fov_rad = 90 * (pi / 180)  # 90 degree desired fov
    stereo_height_px = 480  # 300x300 pixel stereo output   # 这里需要修改一下
    stereo_focal_px = stereo_height_px / 2 / tan(stereo_fov_rad / 2)

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
            cv2.imshow('capture_background', ori_img)
            if if_save:
                cv2.waitKey(add_time)  # 采集等待间隔
                cv2.imwrite(path + '/' + str(int(time.time() * 1000)) + '.jpg', ori_img)
            else:
                cv2.waitKey(8)
        if if_save == " ":
            break
finally:
    pipe.stop()

cv2.destroyAllWindows()
