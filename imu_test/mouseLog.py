import datetime
import time
from pynput import mouse
import os
import keyboard

count = 0
tag = ''
os.chdir(r"C:\Users\Administrator\Desktop\adb")
os.popen("adb start-server")
a = os.popen("adb shell date +%s%N")
t = time.time()
s = a.read()[:-6]
s = int(s) / 10000
print('timeoffset', t - s)
os.chdir('D:\jz_script\imu_test\logs')
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
with open('mouse_log%s.txt' % date, 'a') as f:
    f.write(f'数据采集日期: {date}\n')
    f.write(f'timeoffset {t - s}\n')


# 定义鼠标事件处理函数
def on_mouse_event(x, y, button, pressed):
    global count, tag
    # 按下鼠标按钮时记录日志
    if pressed:
        if str(button) == "Button.left":
            timestamp = time.time()
            if count % 2 == 0:
                print(f"{int(count / 2)} 开始采集 {timestamp}")
                tag += f'{int(count / 2)} {timestamp} start\n'
                count += 1
            else:
                print(f'{int(count / 2)} 结束采集 {timestamp} \n')
                tag += f'{int(count / 2)} {timestamp} end\n'
                with open('mouse_log%s.txt' % date, 'a') as f:
                    f.write(tag)
                count += 1
                tag = ''
        elif str(button) == "Button.right":
            if count % 2 == 1:
                count -= 1
            tag = ''
            print('撤回成功\n')
        elif str(button) == "Button.middle":
            count = 0
            tag = ''
            print('个数清零成功\n')


def keystroke_start(event):
    global count, tag
    keystroke = event.name
    if keystroke == 'caps lock':
        # 创建鼠标监听器
        with mouse.Listener(on_move=None, on_click=on_mouse_event, on_scroll=None) as listener:
            listener.join()
            listener.daemon = False


keyboard.on_press(keystroke_start)
keyboard.wait("q")  # Wait for user to press 'q' to quit
