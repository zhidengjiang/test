# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 23:53:00 2021

@author: zrd
"""

import os
import json
import numpy as np
import csv
import cv2 as cv


def show_point(dir_json, list_json, dir_img, dir_out_plot):
    for cnt, json_name in enumerate(list_json):
        if json_name.endswith('.json'):
            path_json = dir_json + json_name
            img_name = json_name.split('.')[0] + '.png'
            path_img = dir_img + img_name
            with open(path_json, 'r', encoding='gb18030') as path_json:
                jsonx = json.load(path_json)
                tmp = []
                tmp.append(''.join(json_name))
                for shape in jsonx['shapes']:
                    if shape['label'] == "p_l":
                        out_cor1 = np.array(shape['points'], dtype=int).squeeze(0)
                        out_cor1 = out_cor1[::-1]
                    elif shape['label'] == "p_r":
                        out_cor2 = np.array(shape['points'], dtype=int).squeeze(0)
                        out_cor2 = out_cor2[::-1]
            path_json.close()
            img_raw = cv.imread(path_img)
            cv.line(img_raw, (out_cor1[1] - 4, out_cor1[0]), (out_cor1[1] + 4, out_cor1[0]), (0, 255, 0), 2)
            cv.line(img_raw, (out_cor1[1], out_cor1[0] - 4), (out_cor1[1], out_cor1[0] + 4), (0, 255, 0), 2)

            cv.line(img_raw, (out_cor2[1] - 4, out_cor2[0]), (out_cor2[1] + 4, out_cor2[0]), (0, 255, 0), 2)
            cv.line(img_raw, (out_cor2[1], out_cor2[0] - 4), (out_cor2[1], out_cor2[0] + 4), (0, 255, 0), 2)

            cv.line(img_raw, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)
            # cv.line(img_raw, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)
            cv.imwrite(dir_out_plot + img_name, img_raw)



# dir_json = r'F:\ZDJ\our_data\zhujiang\dataset2\select__json/'  # json路径
# # dir_csv = r'C:\Users\JF\Desktop\aopGT.csv'  # 存取的csv路径
# dir_img = r'F:\ZDJ\our_data\zhujiang\dataset2\select_img/'
# dir_out_plot = r'F:\ZDJ\our_data\zhujiang\dataset2\select_plot_landmark/'
dir_json = r'F:\ZDJ\our_data\zhujiang\dataset2\select_json/'  # json路径
# dir_csv = r'C:\Users\JF\Desktop\aopGT.csv'  # 存取的csv路径
dir_img = r'F:\ZDJ\our_data\zhujiang\dataset2\select_img/'
dir_out_plot = r'F:\ZDJ\our_data\zhujiang\dataset2\select_plot_landmark/'
if not os.path.exists(dir_out_plot):
    os.makedirs(dir_out_plot)
list_json = os.listdir(dir_json)
# strxy = []
# json2csv(dir_csv, list_json, strxy)
show_point(dir_json, list_json, dir_img, dir_out_plot)
