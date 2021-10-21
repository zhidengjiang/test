import cv2
import math
import numpy as np


# import time
# a=100#半长轴
# b=50#半短轴

# cv2.imshow('background',background)
# cv2.waitKey(0)
# drawline(background,element,point)

def aod_cal(background, fetalhead_element, right_point, left_point):
    # 旋转之后的坐标
    a = fetalhead_element[1][1] / 2
    b = fetalhead_element[1][0] / 2
    angel = math.pi * (90-fetalhead_element[2]) / 180
    x0 = fetalhead_element[0][0]
    y0 = fetalhead_element[0][1]


    x_pingyi_right = right_point[0] - x0
    y_pingyi_right = right_point[1] - y0
    x_pingyi_left = left_point[0] - x0
    y_pingyi_left = left_point[1] - y0
    k3 = (right_point[1]- left_point[1]) / (right_point[0]- left_point[0])
    angel0 = math.atan(k3)
    # angel0 = math.pi / 3
    transform0 = np.array([[math.cos(angel), -math.sin(angel)],
                           [math.sin(angel), math.cos(angel)]])
    transform1 = np.array([[math.cos(angel), math.sin(angel)],
                           [-math.sin(angel), math.cos(angel)]])
    transform_1 = np.array([[math.cos(angel), math.sin(angel), x0],
                            [-math.sin(angel), math.cos(angel), y0]])
    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])

    # x_xuanzhuan_right, y_xuanzhuan_right = Transmat1 @ np.array([x_pingyi_right, y_pingyi_right])
    x_xuanzhuan_right, y_xuanzhuan_right = transform0 @ np.array([x_pingyi_right, y_pingyi_right])
    # x_m,y_m = transform_1 @ np.array([x_xuanzhuan_right, y_xuanzhuan_right, 1])
    x_xuanzhuan_left, y_xuanzhuan_left = transform0 @ np.array([x_pingyi_left, y_pingyi_left])

    # 未旋转之前的坐标
    x1 = right_point[0]
    y1 = right_point[1]
    x2 = left_point[0]
    y2 = left_point[1]
    x3 = 0.
    y3 = 0.

    # if y2 == y1:
    #     x3 = x_xuanzhuan_right
    #     y3 = np.sqrt(1 - np.power(x3, 2) / np.power(a, 2)) * b  # 新建坐标系下的坐标
    #     x3, y3 = transform_1 @ np.array([x3, y3, 1])
    #     # y3 = x3 * math.sin(-angel) + y3 * math.cos(angel) + y0
    #     # x3 = x1
    #     pass
    #     # d = y3 - y_xuanzhuan_right
    #     # return d
    # else:
    k = -angel - (math.atan((right_point[1] - left_point[1]) / (right_point[0] - left_point[0])))
    if angel + angel0 != math.pi / 2:
        k1 = math.tan(angel + angel0)
    # k0 = (y_xuanzhuan_right - y_xuanzhuan_left) / (x_xuanzhuan_right - x_xuanzhuan_left)
        k1 = -1 / k1
        m = np.power(b, 2) + np.power(a, 2) * np.power(k1, 2)
        n = 2 * np.power(a, 2) * (k1 * y_xuanzhuan_right - np.power(k1, 2) * x_xuanzhuan_right)
        q = np.power(a, 2) * (
                               np.power(y_xuanzhuan_right, 2) +
                               np.power(k1, 2) * np.power(x_xuanzhuan_right,2) -
                               2 * k1 * x_xuanzhuan_right * y_xuanzhuan_right -
                               np.power(b, 2)
                              )
        flag = np.power(n, 2) - 4 * m * q
        if (np.power(n, 2) - 4 * m * q) == 0:
            x3 = - n / 2 * m
            y3 = k1 * x3 - k1 * x_xuanzhuan_right + y_xuanzhuan_right
            # d = np.sqrt(np.power(y3 - y_xuanzhuan_right, 2) + np.power(x3 - x_xuanzhuan_right, 2))
            x3, y3 = transform_1 @ np.array([x3, y3, 1])  # 原坐标系下的坐标
        if (np.power(n, 2) - 4 * m * q) > 0:
            x_1 = (-n + np.sqrt(np.power(n, 2) - 4 * m * q)) / 2 / m
            y_1 = k1 * x_1 - k1 * x_xuanzhuan_right + y_xuanzhuan_right
            # x_1_1, y_1_1 = transform1 @ np.array([x_1, y_1])  # 原坐标系下的坐标

            x_1_1, y_1_1 = transform_1 @ np.array([x_1, y_1, 1])
            x_2 = (-n - np.sqrt(np.power(n, 2) - 4 * m * q)) / 2 / m
            y_2 = k1 * x_2 - k1 * x_xuanzhuan_right + y_xuanzhuan_right
            # x_2_1, y_2_1 = transform1 @ np.array([x_2, y_2])  # 原坐标系下的坐标
            x_2_1, y_2_1 = transform_1 @ np.array([x_2, y_2, 1])
            if y_2_1 < y_1_1:
                y3 = y_2_1
                x3 = x_2_1
                # d = np.sqrt(np.power(y3 - y_xuanzhuan_right, 2) + np.power(x3 - x_xuanzhuan_right, 2))
            else:
                y3 = y_1_1
                x3 = x_1_1
                # d = np.sqrt(np.power(y3 - y_xuanzhuan_right, 2) + np.power(x3 - x_xuanzhuan_right, 2))
        else:
            print("there is a mistake")
    else:
        x3 = x_xuanzhuan_right
        y3 = - b * np.sqrt(1-np.power(x3, 2) / np.power(a, 2))
        x3, y3 = transform_1 @ np.array([x3, y3, 1])

    cv2.circle(background, (int(x3), int(y3)), 3, (255, 0, 0), 3)
    # x_3, y_3 = transform1 @ np.array([x3, y3])
    # cv2.line(background, right_point, (int(x_3), int(y_3)), (255, 255, 255), 3)

    return x3, y3
    # print('AOD',round(aod,2),'（度）')


# cv2.imshow('img',background)
# cv2.waitKey()
if __name__ == "__main__":
    background = np.zeros((600, 600, 3), np.uint8)
    fetalhead_element = ((400, 200), (100, 300), 60)
    right_point = (500, 100)
    left_point = (300, 100)
    cv2.ellipse(background, fetalhead_element, (0, 255, 0), 2)  # 椭圆
    # left_bottom = (400, 100)
    # cv2.circle(background, left_bottom, 3, (0, 0, 255), 3)
    cv2.circle(background, left_point, 3, (255, 0, 0), 3)  # 左端点
    cv2.circle(background, right_point, 3, (255, 0, 0), 3)  # 右端点
    cv2.line(background, right_point, left_point, (255, 255, 255), 3)  # CAOP
    x3, y3 = aod_cal(background, fetalhead_element, right_point, left_point)
    # cv.ellipse(background, ellipse2, (0, 255, 0), 2)
    print('x3:%.6f,y3:%.6f' % (x3, y3))

    cv2.imshow('img', background)
    cv2.waitKey()
