import re
import os

path = r'D:\script\gesture\background_2\shiyan\anno'
os.chdir(path)
new_path = "anno"
if not os.path.exists(new_path):
    os.mkdir(new_path)

for i in os.listdir(path):
    if i.endswith(".xml"):
        a = open(i, encoding="utf-8").read().split('bndbox')[1]
        pattern = re.compile('\d+')
        coordinates = " ".join(re.findall(pattern, a))
        new_file = os.path.join(path, new_path, i.replace("xml", "txt"))
        op = open(new_file, 'w').write(coordinates)

print("end")
