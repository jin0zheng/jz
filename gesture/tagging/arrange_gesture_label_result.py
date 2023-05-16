import os
import numpy as np
import re


def doing(path):
    input("请确认\t%s\t中的数据已经备份" % path)
    path_anno = os.path.join(path, 'anno')
    path_img = os.path.join(path, 'img')
    ann = np.array([i.split(".")[0] for i in os.listdir(path_anno)])
    for i in ann:  # 清理有问题的标注
        s = open(path_anno + "\\" + i + ".xml", encoding='utf-8').read()
        # print(s)
        tag_num = len(s.split("/xmin")) - 1
        # print(tag_num)
        if tag_num == 0:
            os.remove(path_anno + "\\" + i + ".xml")
        elif tag_num == 1:
            a = s.split('bndbox')[1]
            pattern = re.compile('\d+')
            x1, y1, x2, y2 = [int(k) for k in re.findall(pattern, a)]
            if x2 - x1 < 20 or y2 - y1 < 20:
                os.remove(os.path.join(path_anno, i + ".xml"))
        else:
            # print("请检查文件%s" % i)
            os.remove(os.path.join(path_anno, i + ".xml"))

    ann = np.array([i.split(".")[0] for i in os.listdir(path_anno)])
    jpe = np.array([i.split(".")[0] for i in os.listdir(path_img)])
    ann_ = np.setdiff1d(ann, jpe)
    jpe_ = np.setdiff1d(jpe, ann)

    print("ann_:多出的标注:", len(ann_))
    print("jpe_:没标的图片:", len(jpe_))
    for i in ann_:  # 清理多余的标注
        os.remove(os.path.join(path_anno, i + ".xml"))
    for i in jpe_:  # 清理多余的jpg
        os.remove(os.path.join(path_img, i + ".jpg"))
    print("清理完成")


if __name__ == '__main__':
    path = r'D:\jz_script\yolov7-pytorch-master\VOCdevkit\VOC2007'
    doing(path)
