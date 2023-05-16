# 用来采集手势图像
import os.path
import time
from threading import Lock, Thread

import cv2
import numpy as np
import pyrealsense2 as rs

out_path = 'hand_collected'
if not os.path.exists(out_path):
    os.mkdir(out_path)
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


def save1():
    pipe = rs.pipeline()
    cfg = rs.config()
    pipe.start(cfg, callback)
    global height, if_save, out_path, name, gesture_num, interval

    while True:
        frame_mutex.acquire()
        valid = frame_data["timestamp_ms"] is not None
        frame_mutex.release()

        if valid:
            frame_mutex.acquire()
            frame_copy = {"left": frame_data["left"].copy()}
            ori_img = frame_copy["left"]
            frame_mutex.release()
            # print(ori_img.shape,ori_img.dtype)
            cv2.imshow('1', ori_img)
            cv2.waitKey(1)

            if if_save and (gesture_num != None):
                cv2.imwrite(
                    out_path + '/' + name + '/single_' + str(gesture_num) + '_' + str(int(time.time() * 1000)) + '.jpg',
                    ori_img)
            if if_end:
                break

    cv2.destroyAllWindows()


if_save = 0
if_end = 0
gesture_num = None
name_exist = 1
while name_exist:
    name = input('please input name:')
    if name in os.listdir(out_path):
        print('the name has already existed')
    else:
        name_exist = 0

t = Thread(target=save1, args=())
t.start()

os.mkdir(out_path + '/' + name)
interval = input('please input the interval time (usually 8)')
interval = int(interval)

while True:
    if if_save == 0:
        shuru = input('press s to start saving, press q to quit')
        if shuru == 's':
            if_save = 1
            shuru = input('input the gesture num')
            gesture_num = int(shuru)
        if shuru == 'q':
            if_end = 1
            break

    if if_save == 1:
        shuru = input('press s to stop saving, press q to quit')
        if shuru == 's':
            if_save = 0
            gesture_num = None
        if shuru == 'q':
            if_end = 1
            break
