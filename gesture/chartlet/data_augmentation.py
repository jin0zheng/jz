import math
import numpy as np
import cv2


def hand_resize(image, hand_length_range):  # 对手的有效数值（非255）进行缩放，range为最长边范围
    length_change = np.random.randint(hand_length_range[0], hand_length_range[1] + 1)
    image_h, image_w = image.shape
    image_h_min, image_w_min, image_h_max, image_w_max = 100000, 100000, 0, 0
    for h in range(image_h):
        for w in range(image_w):
            if image[h, w] != 255:
                image_h_min = min(image_h_min, h)
                image_w_min = min(image_w_min, w)
                image_h_max = max(image_h_max, h)
                image_w_max = max(image_w_max, w)
    image = image[image_h_min:(image_h_max + 1), image_w_min:(image_w_max + 1)]  # 取出有效范围
    h_before, w_before = image.shape

    if h_before < w_before:
        w_after = length_change
        h_after = int(h_before / w_before * length_change)
    else:
        h_after = length_change
        w_after = int(w_before / h_before * length_change)

    image = cv2.resize(image, (w_after, h_after), interpolation=cv2.INTER_NEAREST)

    return image


def rotate_total(image, angle_1):  # 原图大小不变，旋转后的完全包含原图，所以会增大一些
    angle = np.random.uniform(-angle_1, angle_1)
    # 先进行旋转，这个就是从零点进行旋转
    matrix = np.eye(3)
    matrix[0, 0] = math.cos(angle / 180 * math.pi)
    matrix[1, 1] = math.cos(angle / 180 * math.pi)
    matrix[0, 1] = math.sin(angle / 180 * math.pi)
    matrix[1, 0] = -math.sin(angle / 180 * math.pi)

    # 然后进行平移
    translation_matrix = np.eye(3)
    width = image.shape[1]
    height = image.shape[0]
    # 看四个顶点旋转后的位置，然后取最大最小值
    vertex = np.array([[0, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1], [width - 1, 0, 1]])
    heng_zuobiao = []  # 四个顶点的横坐标
    zong_zuobiao = []
    for i in range(4):
        heng_zuobiao = heng_zuobiao + [(matrix @ vertex[i])[0]]
        zong_zuobiao = zong_zuobiao + [(matrix @ vertex[i])[1]]
    heng_min = min(heng_zuobiao)
    zong_min = min(zong_zuobiao)
    heng_changdu = max(heng_zuobiao) - min(heng_zuobiao) + 1
    zong_changdu = max(zong_zuobiao) - min(zong_zuobiao) + 1
    translation_matrix[0, 2] = -heng_min
    translation_matrix[1, 2] = -zong_min

    # 最后的旋转矩阵
    matrix = translation_matrix @ matrix
    matrix = matrix[:2]
    image = cv2.warpAffine(image, matrix, (int(heng_changdu), int(zong_changdu)),
                           borderValue=255, flags=cv2.INTER_NEAREST)

    return image


def rotate_part(image, angle):  # 原图大小不变，旋转后的图跟原图大小一样，所以原图会有一些在外面
    angle = np.random.uniform(-angle, angle)

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=np.random.randint(0, 256))

    return image


def background_resize(background, background_height_range, background_width_range, final_size):
    if np.random.random() > 0.01:
        height_new = (np.random.random() * (background_height_range[1] - background_height_range[0]) + \
                      background_height_range[0]) * background.shape[0]
        height_new = int(height_new)
        width_new = np.random.random() * (background_width_range[1] - background_width_range[0]) + \
                    background_width_range[0] * background.shape[0]
        width_new = int(width_new)
        x_new = np.random.randint(0, background.shape[1] - width_new)
        y_new = np.random.randint(0, background.shape[0] - height_new)

        background_crop = background[y_new:(y_new + height_new), x_new:(x_new + width_new)]
        return cv2.resize(background_crop, (final_size[1], final_size[0]))
    else:
        return background


def hand_mean_std(hand, hand_mean_range, hand_std_range):
    if np.random.random() > 0.01:
        hand_data = []  # 用来装手的数值
        for i in range(hand.shape[0]):
            for j in range(hand.shape[1]):
                if hand[i, j] != 255:
                    hand_data.append(hand[i, j])
        hand_data = np.array(hand_data)
        mean_old = np.mean(hand_data)
        mean_new = mean_old * (np.random.random() * (hand_mean_range[1] - hand_mean_range[0]) + hand_mean_range[0])
        std_old = np.std(hand_data)
        std_new = std_old * (np.random.random() * (hand_std_range[1] - hand_std_range[0]) + hand_std_range[0])
        # print(np.std(hand_data),np.mean(hand_data))

        hand_data = hand_data * std_new / std_old
        hand_data = hand_data - np.mean(hand_data) + mean_new
        # print(np.std(hand_data), np.mean(hand_data))

        hand_data = np.clip(hand_data, 0, 254)

        final_hand = np.ones(hand.shape) * 255
        num = 0
        for i in range(hand.shape[0]):
            for j in range(hand.shape[1]):
                if hand[i, j] != 255:
                    final_hand[i, j] = hand_data[num]
                    num += 1
        return final_hand.astype(np.uint8)
    return hand


def background_mean_std(background, background_mean_range, background_std_range):
    if np.random.random() > 0.01:
        mean_old = np.mean(background)
        mean_new = mean_old * (
                np.random.random() * (background_mean_range[1] - background_mean_range[0]) + background_mean_range[
            0])
        std_old = np.std(background)
        std_new = std_old * (
                np.random.random() * (background_std_range[1] - background_std_range[0]) + background_std_range[0])

        background = background * std_new / std_old
        background = background - np.mean(background) + mean_new
        background = np.clip(background, 0, 255)

        return background.astype(np.uint8)
    return background
