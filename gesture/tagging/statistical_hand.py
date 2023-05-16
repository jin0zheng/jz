import os

path = r'D:\jz_script\gesture\capture_hand\wubaolin\img'
dir1 = {str(i): 0 for i in range(29)}
s = 1
n = 0
for i in os.listdir(path):
    n += 1
    num = i.split("_")[1]
    if s:
        # if int(num) in [18, 25] and s:
        if n % 2 == 0:
            os.remove(path + "\\" + i)
        # print(i)
for i in os.listdir(path):
    num = i.split("_")[1]
    if num in dir1.keys():
        dir1[num] += 1

for i in dir1.items():
    print(i)

print('count', len(os.listdir(path)))
