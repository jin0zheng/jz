import torch

print(torch.__version__)
print(torch.cuda.is_available())  # cuda是否可用
print(torch.cuda.device_count())  # 返回GPU的数量
print(torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
