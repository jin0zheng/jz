import time
from pynput.keyboard import Controller, Listener, Key
import numpy as np
import threading

n = 0
t = time.time()
print("每次标注的平均时间:%.2fs" % np.average(np.loadtxt("1.txt")))
print("最近2000次标注的时间:%.2fs" % np.average(np.loadtxt("1.txt")[-2000:]))
def on_release(key):
    global t, n

    if key == Key.caps_lock:
        Controller().tap(Key.delete)
        Controller().tap('d')
        n += 1
        t2 = time.time() - t
        if 0.1 < t2 < 60:
            open('1.txt', "a+").write("%.5f\n" % t2)
        print(n, t2)
        t = time.time()


def listener_():
    with Listener(on_press=on_release) as listener:
        listener.join()
        listener.setDaemon(True)


if __name__ == '__main__':
    thread = threading.Thread(target=listener_, args=())
    thread.start()
