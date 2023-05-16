import os

path_target = r'C:\Users\Administrator\Desktop\target'
path_ori = r"C:\Users\Administrator\Desktop\original"


for i in os.listdir(path_target):
    if i.startswith('51'): # 10 11 20 21
        try:
            print(i.split("-")[0],end="\t")
            s = open(os.path.join(path_ori, i.replace('target', 'original'))).readlines()[2].strip()
            print(s,end="\t")
            s = open(os.path.join(path_target, i)).readlines()[2].strip()
            print(s)
        except:
            print(i)