import os

anno_file_path = r'single_0_1681117230042.xml'
s = open(anno_file_path).read()
# print(s)
path = "D:\jz_script\gesture\capture_hand\jinzheng1"
img_path = os.path.join(path, "img")
anno_path = os.path.join(path, "anno")
anno1_path = os.path.join(path, "anno1")

for i in os.listdir(anno1_path):
    s = open(anno_file_path).read()
    name = i.split(".")[0]
    coodrings = open(os.path.join(anno1_path,i)).read().split()
    s = s.replace("507",coodrings[0])
    s = s.replace("379", coodrings[1])
    s = s.replace("592", coodrings[2])
    s = s.replace("451", coodrings[3])
    s = s.replace("single_0_1681117230042",name)

    open(os.path.join(anno_path,i.replace("txt","xml")),"w").write(s)
