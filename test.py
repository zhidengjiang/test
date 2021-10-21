import os
import json
import numpy as np
import csv

def json2csv(dir_csv, list_json, strxy):
    with open(dir_csv, 'w+',newline='') as fcsv:
        writer = csv.writer(fcsv)    
        writer.writerow(["filename","pos1_x","pos1_y","pos2_x","pos2_y"])
        for cnt, json_name in enumerate(list_json):
            if json_name.endswith('.json'):
                path_json = dir_json + json_name
                with open(path_json, 'r', encoding='gb18030') as path_json:
                    jsonx = json.load(path_json)
                    tmp = []
                    tmp.append(''.join(json_name))
                    for shape in jsonx['shapes']:
                        xy = np.array(shape['points'])
                        for m, n in xy:
                            tmp.append(''.join(str(m)))
                            tmp.append(''.join(str(n)))
                    strxy.append(tmp)
                path_json.close()
        writer.writerows(strxy)
    fcsv.close()    

dir_json = r'H:\ZJYY\deal\dataset3\select_image/'  # json路径
dir_csv = r'H:\ZJYY\deal\dataset3\aopGT_Z.csv'  # 存取的csv路径
if not os.path.exists(dir_csv):
    os.makedirs(dir_csv)
list_json = os.listdir(dir_json)
strxy = []
json2csv(dir_csv, list_json, strxy)

