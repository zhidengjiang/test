
import cv2


def img_find_data(img, col_index):
    for j in range(img.shape[0]):
        if img[j, col_index] >= 9:
            return j

depth_list = [[11, 18, 21, 18, 13],
              [16, 21, 24, 22, 16],
              [18, 23, 26, 24, 18],
              [22,28,30,29,24],
              [27, 30, 33, 31, 27],
              [29, 33, 35, 33, 29],
              [33, 37, 38, 37, 33],
              [35, 40, 41, 40, 37],
              [41, 45, 46, 45, 42],
              [47, 50, 51, 50, 47],
              [55, 58, 60, 58, 55],
              [63, 65, 66, 64, 64],
              [76, 78, 78, 78, 75],
              [95, 94, 94, 94, 93],
              [119, 118, 119, 120, 118],]

depth_cm = [21,19,17,15,14,13,12,11,10,9,8,7,6,5,4]

col_index = [270, 290, 310, 330, 350]  #可不用

def cal_img_depth(img, depth_list, depth_cm):

    col_depth_270 = img_find_data(img, 270)
    col_depth_290 = img_find_data(img, 290)
    col_depth_310 = img_find_data(img, 310)
    end_pixel_310 = 469
    col_depth_330 = img_find_data(img, 330)
    col_depth_350 = img_find_data(img, 350)


    for i in range(len(depth_cm)):
        col_sum = 0
        if abs(col_depth_270 - depth_list[i][0]) <= 1:
            col_sum += 1
        if abs(col_depth_290 - depth_list[i][1]) <= 1:
            col_sum += 1
        if abs(col_depth_310 - depth_list[i][2]) <= 1:
            col_sum += 1
        if abs(col_depth_330 - depth_list[i][3]) <= 1:
            col_sum += 1
        if abs(col_depth_350 - depth_list[i][4]) <= 1:
            col_sum += 1
        if col_sum >= 4:
            eachpixel2cm = depth_cm[i]*1.0/(end_pixel_310 - depth_list[i][2])  #计算每个像素的实际距离大小
            return depth_cm[i], eachpixel2cm

    return 0, 0


if __name__ == '__main__':
    img_path = r'C:\Users\Administrator\Desktop\最后结果整理\分割定位结果图_new\ori_image/ATD_0091.png'
    img = cv2.imread(img_path, 0)
    depth, eachpixel2cm = cal_img_depth(img, depth_list=depth_list, depth_cm=depth_cm)   #计算某一图片对应的深度
    print('该图像超声扫描深度：', depth, 'cm')
    print('每格像素对应的实际距离：', eachpixel2cm, 'cm')
