import os
import numpy as np
path = r'D:\matting_data\synthesis_2\shang_classify'
path1 = os.path.join(path,'txt')
path2 = os.path.join(path,'img')
txt = np.array([i.split(".")[0] for i in os.listdir(path1)])
jpe = np.array([i.split(".")[0] for i in os.listdir(path2)])
txt_ = np.setdiff1d(txt, jpe)
jpe_ = np.setdiff1d(jpe, txt)
print("txt_:多出的txt", len(txt_))
print("jpe_:没标的图片", len(jpe_))
input("请确认是否删除")
for i in txt_:
    os.remove(os.path.join(path1, i + ".txt"))
for i in jpe_:
    os.remove(os.path.join(path2, i + ".jpg"))
print("清理完成")
