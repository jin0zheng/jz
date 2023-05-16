import numpy as np
import quaternion
import pandas as pd
op = open("../imu/left-swipe_positive_0.txt")
index = op.readline().strip().split('\t')
print(index)
datas = np.loadtxt(op)
datas = datas.transpose()
s = []
for i in range(datas.shape[0]):
    if i > 1:
        s.extend(savgol_filter(datas[i:i + 1, :], 60 + 1, 3))
    else:
        s.extend(datas[i:i + 1, :])
z = pd.Series(s, index)
z.PhoneTime = z.PhoneTime - z.PhoneTime[0]

orientation = quaternion.one

for i in range(len(time)):
    dt = time[i] - time[i - 1]

    # 计算角速度
    gyro = np.array([GYRO_X[i], GYRO_Y[i], GYRO_Z[i]])

    # 将角速度转换为四元数表示法，并将其与旧的方向乘起来得到新方向
    q = quaternion.from_rotation_vector(gyro * dt)
    orientation = q * orientation

    # 标准化方向
    orientation.normalize()

    # 使用新的方向更新位置
    # 以及你想要执行的任何其他操作

# 将方向转换为旋转矩阵对象
rotation = orientation.rotation_matrix

# 从旋转矩阵中提取角度信息（弧度）
roll, pitch, yaw = rotation.as_euler('xyz')

# 将角度转换为度数
roll = np.degrees(roll)
pitch = np.degrees(pitch)
yaw = np.degrees(yaw)

# 计算位置上的累积变化
# 取决于你想要执行的操作和数据的