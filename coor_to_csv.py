import os
import json
import numpy as np
import csv


def json2csv(path_json, path_csv, strxy, json_name):
    with open(path_json, 'r', encoding='gb18030', newline='') as path_json:
        jsonx = json.load(path_json)
        temp = []
        temp.append(str(json_name))
        with open(path_csv, 'w+', newline='', encoding='utf-8') as fcsv:
            write = csv.writer(fcsv)
            write.writerow(["filename", "position1","position2","position3","position4"])
            for shape in jsonx['shapes']:
                xy = np.array(shape['points'])
                # label=str(shape['label'])
                for m, n in xy:
                    # strxy.append(str(m) + ',' + str(n))
                    temp.append(str(m))
                    temp.append(str(n))
                #    strxy+=str(m)+','+str(n)+','
                # strxy+=label
            # fcsv.writelines(','.join(strxy) + '\n')
            # fcsv.writelines("\n")
            # json_name = str(json_name)
            #     write = csv.writer(fcsv, dialect='excel')
            # write.writerow(["filename", "position1","position2","position3","position4"])
            strxy.append(temp)
            write.writerow(strxy)
            fcsv.close()
        strxy.clear()
        path_json.close()


            # fcsv.writelines("\n")
                # fcsv.close()


dir_json = r'H:\ZJYY\deal\dataset3\select_coor/'  # json路径
dir_csv = r'H:\ZJYY\deal\dataset3\aopGT.csv'  # 存取的csv路径
if not os.path.exists(dir_csv):
    os.makedirs(dir_csv)
list_json = os.listdir(dir_json)
strxy = []
for cnt, json_name in enumerate(list_json):
    # print('cnt=%d,name=%s' % (cnt, json_name))
    path_json = dir_json + json_name
    # path_csv = dir_csv + json_name.replace('.json', '.csv')
    # print(path_json, path_csv)
    # path_csv = dir_csv
    # strxy.append(str(json_name))
    json2csv(path_json, dir_csv, strxy, json_name)
