import os.path
import shutil

input_name = 'hulitian_classify'  # 只需要修改这个
input_img_path = 'D:/matting_data/synthesis_1_img'  # 手部的img数据用来
input_txt_path = 'D:/matting_data/synthesis_1_txt'  # 手部的txt数据

output = 'D:/matting_data/synthesis_2'

if not os.path.exists(output):
    os.mkdir(output)

if os.path.exists(output + '/' + input_name):
    print(input_name + ' already done')
    print('delete ' + input_name)
    shutil.rmtree(output + '/' + input_name)

os.mkdir(output + '/' + input_name)

# input_img_path+'/'+input_name   : synthesis_1_img/hulitian_classify
folder_list = os.listdir(input_img_path + '/' + input_name)

for folder_name in folder_list:
    # input_img_path+'/'+input_name+'/' + folder_name :synthesis_1_img/hulitian_classify/-10
    img_list = os.listdir(input_img_path + '/' + input_name + '/' + folder_name)
    for img_name in img_list:
        shutil.copy(input_txt_path + '/' + input_name + '/' + folder_name + '/' + img_name[:-3] + 'txt',
                    output + '/' + input_name + '/' + img_name[:-3] + 'txt')

print('done')