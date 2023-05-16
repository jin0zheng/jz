import os
path = r'D:\jz_script\gesture\capture_hand\wubaolin'
os.chdir(path)
for i in os.listdir(path):
    # num = i.split('_')[1]
    # if num == '29':
    #     os.rename(i,i.replace('_29_','_3_'))
    # if num == '14':
    #     os.rename(i,i.replace('_14_','_15z_'))
    # if num == '15':
    #     os.rename(i,i.replace('_15_','_14z_'))


    os.rename(i, i.replace('z', ''))
