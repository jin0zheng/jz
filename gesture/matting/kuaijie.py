import time
from pynput.keyboard import Controller, Key
import pynput.mouse as pm
from pynput.mouse import Button

n = 0


def on_click(x, y, button, pressed):
    if not pressed and button == Button.left:
        time.sleep(0.1)
        Controller().tap(Key.delete)
        global n
        n += 1
        print(n)
    return False
while 1:
    with pm.Listener(on_click=on_click) as pmlistener:
        pmlistener.join()
# deff sfdsdfsdf ewfsdsdf