import cv2
import math
import numpy as np
import time


# a=100#半长轴
# b=50#半短轴

#
def drawline(background, element_, element_1):
    element = (element_[0], (element_[1][1], element_[1][0]), element_[2] - 90)
    element1 = (element_1[0], (element_1[1][1], element_1[1][0]), element_1[2] - 90)
    #    background = np.zeros((600,600,3),np.uint8)
    #    element1=((100,100),(150,60),-40)
    cv2.ellipse(background, element1, (255, 0, 0), 2)
    [d11, d12] = [element1[0][0] - element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] - element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    [d21, d22] = [element1[0][0] + element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] + element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    # [n1,n2] = [element1[0][0]-element1[1][1]/2*math.sin(element1[2]*0.01745),
    #     element1[0][1]+element1[1][1]/2*math.cos(element1[2]*0.01745)]
    # [m1,m2] = [element1[0][0]+element1[1][1]/2*math.sin(element1[2]*0.01745),
    #     element1[0][1]-element1[1][1]/2*math.cos(element1[2]*0.01745)]
    cv2.line(background, (round(d11), round(d12)), (round(d21), round(d22)), (255, 0, 255), 2)

    #    element=((400,300),(400,250),40)
    cv2.ellipse(background, element, (0, 0, 255), 2)
    [d31, d32] = [element[0][0] + element[1][0] / 2 * math.cos(element[2] * 0.01745),
                  element[0][1] + element[1][0] / 2 * math.sin(element[2] * 0.01745)]
    # cv2.imshow('img',background)
    # cv2.waitKey()
    #    tic = time.time()
    a = element[1][0] / 2
    b = element[1][1] / 2
    angel = 2 * math.pi * element[2] / 360
    dp21 = d21 - element[0][0]
    dp22 = d22 - element[0][1]
    dp11 = d11 - element[0][0]
    dp12 = d12 - element[0][1]
    dp2 = np.array([[dp21], [dp22]])
    dp1 = np.array([[dp11], [dp12]])
    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])
    dpz1 = Transmat1 @ dp1
    dpz2 = Transmat1 @ dp2
    dpz21 = dpz2[0][0]
    dpz22 = dpz2[1][0]
    dpz11 = dpz1[0][0]
    dpz12 = dpz1[1][0]
    if (dpz22 - dpz12) / -(dpz21 - dpz11) != 0:
        xielv_hsd = -(dpz21 - dpz11) / ((dpz22 - dpz12))
        bias_hsd = dpz22 - xielv_hsd * dpz21
        try:
            jiaox1 = (-2 * xielv_hsd * bias_hsd / b ** 2 + math.sqrt(4 * xielv_hsd ** 2 * bias_hsd ** 2 / b ** 4
                                                                     - 4 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2) * (
                                                                             bias_hsd ** 2 / b ** 2 - 1))) / (
                             2 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2))
            jiaox2 = (-2 * xielv_hsd * bias_hsd / b ** 2 - math.sqrt(4 * xielv_hsd ** 2 * bias_hsd ** 2 / b ** 4
                                                                     - 4 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2) * (
                                                                             bias_hsd ** 2 / b ** 2 - 1))) / (
                             2 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2))

            jiaoy1 = jiaox1 * xielv_hsd + bias_hsd
            jiaoy2 = jiaox2 * xielv_hsd + bias_hsd
        except:
            jiaox1 = dpz21
            jiaox2 = dpz21
            jiaoy1 = dpz22
            jiaoy2 = dpz22
        hs_jiaox = -bias_hsd / xielv_hsd
        theta = math.atan(-1 / xielv_hsd)
    else:
        if dpz21 < -a:
            jiaox1 = dpz21
            jiaox2 = dpz21
            jiaoy1 = dpz22
            jiaoy2 = dpz22
        else:
            jiaox1 = dpz21
            jiaox2 = dpz21
            jiaoy1 = b * math.sqrt(1 - dpz21 ** 2 / a ** 2)
            jiaoy2 = -b * math.sqrt(1 - dpz21 ** 2 / a ** 2)
        hs_jiaox = dpz21
        theta = 0
    if math.sqrt((dpz21 - jiaox1) ** 2 + (dpz22 - jiaoy1) ** 2) <= math.sqrt(
            (dpz21 - jiaox2) ** 2 + (dpz22 - jiaoy2) ** 2):
        jiaoy = jiaoy1
        jiaox = jiaox1
    else:
        jiaoy = jiaoy2
        jiaox = jiaox2
    jiaopz = np.array([[jiaox], [jiaoy]])
    jiaop = list(Transmat2 @ jiaopz)
    jiao1 = jiaop[0][0] + element[0][0]
    jiao2 = jiaop[1][0] + element[0][1]
    cv2.line(background, (round(d21), round(d22)), (int(round(jiao1)), int(round(jiao2))), (0, 255, 255), 2)

    hs_ds = 30 / math.cos(theta)
    hsx = hs_jiaox + hs_ds
    hsy = 0
    hspz = np.array([[hsx], [hsy]])
    hsp = list(Transmat2 @ hspz)
    hs1 = hsp[0][0] + element[0][0]
    hs2 = hsp[1][0] + element[0][1]
    cv2.line(background, (round(d31), round(d32)), (int(round(hs1)), int(round(hs2))), (255, 255, 255), 2)
    # cv2.imshow('img',background)
    # cv2.waitKey()
    if dpz21 ** 2 - a ** 2 == 0:
        dpz21 += 1

    xielv_aod = (dpz21 * dpz22 - math.sqrt(b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2)) / (
            dpz21 ** 2 - a ** 2)
    # xielv2 = (x3*y3+np.sqrt(b**2*x3**2+a**2*y3**2-a**2*b**2))/(x3**2-a**2)
    bias_aod = dpz22 - xielv_aod * dpz21
    # qiepz1=(-2*xielv_aod*bias_aod/b**2-math.sqrt(4*xielv_aod**2*bias_aod**2/b**4
    #                                               -4*(1/a**2+xielv_aod**2/b**2)*(bias_aod**2/b**2-1)))/(2*(1/a**2+xielv_aod**2/b**2))
    qiepz1 = (-2 * xielv_aod * bias_aod / b ** 2) / (2 * (1 / a ** 2 + xielv_aod ** 2 / b ** 2))
    qiepz2 = qiepz1 * xielv_aod + bias_aod
    qiepz = np.array([[qiepz1], [qiepz2]])
    qiep = list(Transmat2 @ qiepz)
    qie1 = qiep[0][0] + element[0][0]
    qie2 = qiep[1][0] + element[0][1]

    ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)
    ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)
    ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)

    aod = math.acos((ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)) / math.pi * 180  ##余弦定理
    hsd = math.sqrt((d21 - jiao1) ** 2 + (d22 - jiao2) ** 2)
    hs = math.sqrt((d31 - hs1) ** 2 + (d32 - hs2) ** 2)

    #    file_handle = open('data1.txt',mode='w')
    #    file_handle.write('hello \n')
    #    file_handle.close()
    #    toc = time.time()
    #    print(toc-tic,'秒')
    #    print('AOD',round(aod,2),'（度）')
    #    print('HSD',round(hsd,2),'（像素）')
    #    print('HS',round(hs,2),'（像素）')

    cv2.line(background, (round(d21), round(d22)), (int(qie1), int(qie2)), (255, 0, 255), 2)
    return background, aod, hsd, hs


# drawline_AOD(img_result, ellipse2, ellipse2, out_cor2, out_cor1)
def drawline_AOD(background, element_, element_1, endpoint, endpoint_l):
    element = (element_[0], (element_[1][1], element_[1][0]), element_[2] - 90)
    element1 = (element_1[0], (element_1[1][1], element_1[1][0]), element_1[2] - 90)
    #    background = np.zeros((600,600,3),np.uint8)
    #    element1=((100,100),(150,60),-40)

    # [d11, d12] = [element1[0][0] - element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
    #               element1[0][1] - element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    # [d21, d22] = [element1[0][0] + element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
    #               element1[0][1] + element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    [d11, d12] = [endpoint_l[1], endpoint_l[0]]  # left （x，y）

    [d21, d22] = [endpoint[1], endpoint[0]]  # right (x,y)
    # [n1,n2] = [element1[0][0]-element1[1][1]/2*math.sin(element1[2]*0.01745),
    #     element1[0][1]+element1[1][1]/2*math.cos(element1[2]*0.01745)]
    # [m1,m2] = [element1[0][0]+element1[1][1]/2*math.sin(element1[2]*0.01745),
    #     element1[0][1]-element1[1][1]/2*math.cos(element1[2]*0.01745)]
    # cv2.line(background, (round(d11), round(d12)), (round(d21), round(d22)), (255, 0, 255), 2)

    [d31, d32] = [element[0][0] + element[1][0] / 2 * math.cos(element[2] * 0.01745),
                  element[0][1] + element[1][0] / 2 * math.sin(element[2] * 0.01745)]
    # cv2.imshow('img',background)
    # cv2.waitKey()
    #    tic = time.time()
    a = element[1][0] / 2  # 长轴
    b = element[1][1] / 2  # 短轴
    angel = 2 * math.pi * element[2] / 360
    dp21 = d21 - element[0][0]
    dp22 = d22 - element[0][1]
    dp11 = d11 - element[0][0]
    dp12 = d12 - element[0][1]
    dp2 = np.array([[dp21], [dp22]])
    dp1 = np.array([[dp11], [dp12]])
    transform0 = np.array([[math.cos(angel), -math.sin(angel)],
                           [math.sin(angel), math.cos(angel)]])
    transform1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                           [math.sin(-angel), math.cos(-angel)]])
    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])
    d = 0.
    dpz1 = Transmat1 @ dp1
    dpz2 = Transmat1 @ dp2
    dpz21 = dpz2[0][0]  # x_xuanzhuan_right
    dpz22 = dpz2[1][0]  # y_xuanzhuan_right
    dpz11 = dpz1[0][0]  # x_xuanzhuan_left
    dpz12 = dpz1[1][0]  # y_xuanzhuan_left

    x_xuanzhuan_right = dpz21
    y_xuanzhuan_right = dpz22
    x_xuanzhuan_left = dpz11
    y_xuanzhuan_left = dpz12
    x3 = 0.
    y3 = 0.
    if y_xuanzhuan_right == y_xuanzhuan_left:
        x3 = d21
        y3 = np.sqrt(1 - np.power(b, 2) / np.power(a, 2) * np.power(x3, 2)) * b  # 原坐标系下的坐标
    else:
        k0 = (y_xuanzhuan_right - y_xuanzhuan_left) / (x_xuanzhuan_right - x_xuanzhuan_left)
        k1 = -1 / k0
        m = np.power(b, 2) + np.power(a, 2) * np.power(k1, 2)
        n = 2 * np.power(a, 2) * (k1 * y_xuanzhuan_right - np.power(k1, 2) * np.power(x_xuanzhuan_right, 2))
        q = np.power(a, 2) * (np.power(y_xuanzhuan_right, 2) + k1 * np.power(x_xuanzhuan_right,
                                                                             2) - 2 * k1 * x_xuanzhuan_right * y_xuanzhuan_right - np.power(
            b, 2))
        if (np.power(n, 2) - 4 * m * q) == 0:
            x3 = - n / 2 * m
            y3 = k1 * x3 - k1 * x_xuanzhuan_right + y_xuanzhuan_right  # 新建坐标系下的坐标
            x3, y3 = Transmat2 @ np.array([x3, y3])  # 原坐标系下的坐标
        if (np.power(n, 2) - 4 * m * q) > 0:
            x_1 = (-n + np.sqrt(np.power(n, 2) - 4 * m * q)) / 2 * m
            y_1 = k1 * x_1 - k1 * x_xuanzhuan_right + y_xuanzhuan_right
            x_1_1, y_1_1 = Transmat2 @ np.array([x_1, y_1])  # 算出来的是原坐标系下的坐标
            x_2 = (-n - np.sqrt(np.power(n, 2) - 4 * m * q)) / 2 * m
            y_2 = k1 * x_2 - k1 * x_xuanzhuan_right + y_xuanzhuan_right
            x_2_1, y_2_1 = Transmat2 @ np.array([x_2, y_2])  # 算出来的是原坐标系下的坐标
            if y_2_1 > y_1_1:
                y3 = y_2_1
                x3 = x_2_1
                # d = np.sqrt(np.power(y3 - y_xuanzhuan_right, 2) + np.power(x3 - x_xuanzhuan_right, 2))
            else:
                y3 = y_1_1
                x3 = x_1_1
        else:
            print("there is a mistake")
    d = np.sqrt(np.power(y3 - d22, 2) + np.power(x3 - d21, 2))
    cv2.line(background, (np.round(d21), np.round(d22)), (int(x3), int(y3)), (0, 0, 0), 3)

    if (dpz22 - dpz12) / -(dpz21 - dpz11) != 0:
        xielv_hsd = -(dpz21 - dpz11) / ((dpz22 - dpz12))
        bias_hsd = dpz22 - xielv_hsd * dpz21
        try:
            jiaox1 = (-2 * xielv_hsd * bias_hsd / b ** 2 + math.sqrt(4 * xielv_hsd ** 2 * bias_hsd ** 2 / b ** 4
                                                                     - 4 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2) * (
                                                                             bias_hsd ** 2 / b ** 2 - 1))) / (
                             2 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2))
            jiaox2 = (-2 * xielv_hsd * bias_hsd / b ** 2 - math.sqrt(4 * xielv_hsd ** 2 * bias_hsd ** 2 / b ** 4
                                                                     - 4 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2) * (
                                                                             bias_hsd ** 2 / b ** 2 - 1))) / (
                             2 * (1 / a ** 2 + xielv_hsd ** 2 / b ** 2))

            jiaoy1 = jiaox1 * xielv_hsd + bias_hsd
            jiaoy2 = jiaox2 * xielv_hsd + bias_hsd
        except:
            jiaox1 = dpz21
            jiaox2 = dpz21
            jiaoy1 = dpz22
            jiaoy2 = dpz22
        hs_jiaox = -bias_hsd / xielv_hsd
        theta = math.atan(-1 / xielv_hsd)
    else:
        if dpz21 < -a:
            jiaox1 = dpz21
            jiaox2 = dpz21
            jiaoy1 = dpz22
            jiaoy2 = dpz22
        else:
            jiaox1 = dpz21
            jiaox2 = dpz21
            jiaoy1 = b * math.sqrt(1 - dpz21 ** 2 / a ** 2)
            jiaoy2 = -b * math.sqrt(1 - dpz21 ** 2 / a ** 2)
        hs_jiaox = dpz21
        theta = 0
    if math.sqrt((dpz21 - jiaox1) ** 2 + (dpz22 - jiaoy1) ** 2) <= math.sqrt(
            (dpz21 - jiaox2) ** 2 + (dpz22 - jiaoy2) ** 2):
        jiaoy = jiaoy1
        jiaox = jiaox1
    else:
        jiaoy = jiaoy2
        jiaox = jiaox2
    jiaopz = np.array([[jiaox], [jiaoy]])
    jiaop = list(Transmat2 @ jiaopz)
    jiao1 = jiaop[0][0] + element[0][0]
    jiao2 = jiaop[1][0] + element[0][1]
    # cv2.line(background, (round(d21), round(d22)), (int(round(jiao1)), int(round(jiao2))), (0, 255, 255), 2)

    hs_ds = 30 / math.cos(theta)
    hsx = hs_jiaox + hs_ds
    hsy = 0
    hspz = np.array([[hsx], [hsy]])
    hsp = list(Transmat2 @ hspz)
    hs1 = hsp[0][0] + element[0][0]
    hs2 = hsp[1][0] + element[0][1]
    # cv2.line(background, (round(d31), round(d32)), (int(round(hs1)), int(round(hs2))), (255, 255, 255), 2)
    # cv2.imshow('img',background)
    # cv2.waitKey()
    if dpz21 ** 2 - a ** 2 == 0:
        dpz21 += 1
    if (b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2) >= 0:
        xielv_aod = (dpz21 * dpz22 - math.sqrt(b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2)) / (
                dpz21 ** 2 - a ** 2)
    else:
        xielv_aod = 0  ##加入判断，保证sqrt内的数大于等于0
    # xielv2 = (x3*y3+np.sqrt(b**2*x3**2+a**2*y3**2-a**2*b**2))/(x3**2-a**2)
    bias_aod = dpz22 - xielv_aod * dpz21
    # qiepz1=(-2*xielv_aod*bias_aod/b**2-math.sqrt(4*xielv_aod**2*bias_aod**2/b**4
    #                                               -4*(1/a**2+xielv_aod**2/b**2)*(bias_aod**2/b**2-1)))/(2*(1/a**2+xielv_aod**2/b**2))
    qiepz1 = (-2 * xielv_aod * bias_aod / b ** 2) / (2 * (1 / a ** 2 + xielv_aod ** 2 / b ** 2))
    qiepz2 = qiepz1 * xielv_aod + bias_aod
    qiepz = np.array([[qiepz1], [qiepz2]])
    qiep = list(Transmat2 @ qiepz)
    qie1 = qiep[0][0] + element[0][0]
    qie2 = qiep[1][0] + element[0][1]

    ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)
    ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)
    ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)

    aod = math.acos((ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)) / math.pi * 180  ##余弦定理
    #    file_handle = open('data1.txt',mode='w')
    #    file_handle.write('hello \n')
    #    file_handle.close()
    #    toc = time.time()
    #    print(toc-tic,'秒')
    #    print('AOD',round(aod,2),'（度）')
    #    print('HSD',round(hsd,2),'（像素）')
    #    print('HS',round(hs,2),'（像素）')
    cv2.line(background, (round(d21), round(d22)), (int(qie1), int(qie2)), (0, 0, 255), 2)  # 红色
    cv2.line(background, (np.round(d21), np.round(d22)), (np.round(d11), np.round(d12)), (0, 0, 255), 2)
    return background, aod, d


# cv2.imshow('img',background)
# cv2.waitKey()
if __name__ == "__main__":
    background = np.zeros((600, 600, 3), np.uint8)
    element1 = ((100, 100), (60, 250), 45)
    element = ((400, 300), (250, 400), 45)

    img, a, b, c = drawline(background, element, element1)
    cv2.imshow('img', img)
    cv2.waitKey()
    # background1 = np.zeros((600,600,3),np.uint8)
    # cv2.ellipse(background1, element, (255,0,0), 2)
    # cv2.ellipse(background1, element1, (255,0,0), 2)
    # cv2.imshow('img',background1)
    # cv2.waitKey()
