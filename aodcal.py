import cv2
import math
import numpy as np
# import time
#a=100#半长轴
#b=50#半短轴

# cv2.imshow('background',background)
# cv2.waitKey(0)
# drawline(background,element,point)

def aod_cal(background, fetalhead_element, right_point, left_point):

    b=fetalhead_element[1][0]/2
    a=fetalhead_element[1][1]/2
    angel = math.pi*fetalhead_element[2]/180
    x0 = fetalhead_element[0][1]
    y0 = fetalhead_element[0][0]
    x_pingyi = right_point[1] - x0
    y_pingyi = right_point[0] - y0
    transform0=np.array([[math.cos(angel),-math.sin(angel)],
                        [math.sin(angel), math.cos(angel)]])
    x_xuanzhuan, y_xuanzhuan = transform0 @ np.array([x_pingyi, y_pingyi])

    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])

    k1 = (x_xuanzhuan*y_xuanzhuan + (x_xuanzhuan**2*y_xuanzhuan**2-(x_xuanzhuan**2-a**2)*(y_xuanzhuan**2 - b**2))**(1/2))/(x_xuanzhuan**2 - a**2)
    k2 = (x_xuanzhuan*y_xuanzhuan - (x_xuanzhuan**2*y_xuanzhuan**2-(x_xuanzhuan**2-a**2)*(y_xuanzhuan**2 - b**2))**(1/2))/(x_xuanzhuan**2 - a**2)
    bias1 = y_xuanzhuan - k1*x_xuanzhuan
    bias2 = y_xuanzhuan - k2*x_xuanzhuan
    
    ## 方程1 与椭圆交点
    x1 = -a**2*k1*bias1/(b**2 + a**2*k1**2)   
    y1 = k1*x1 + bias1
    
    ## 方程2 与椭圆交点
    x2 = -a**2*k2*bias2/(b**2 + a**2*k2**2)   
    y2 = k2*x2 + bias2
    
    
    transform1=np.array([[math.cos(-angel),-math.sin(-angel)],
                        [math.sin(-angel), math.cos(-angel)]])
    x1_xuanzhuan, y1_xuanzhuan = transform1 @ np.array([x1, y1])
    x2_xuanzhuan, y2_xuanzhuan = transform1 @ np.array([x2, y2])
    
    x1_pingyi = x1_xuanzhuan + x0
    y1_pingyi = y1_xuanzhuan + y0
    x2_pingyi = x2_xuanzhuan + x0
    y2_pingyi = y2_xuanzhuan + y0
    
    # cv2.circle(background, (int(y1_pingyi), int(x1_pingyi)), 3, (0, 255, 0), 3)
    # cv2.circle(background, (int(y2_pingyi), int(x2_pingyi)), 3, (0, 0, 255), 3)

    
    if y1_pingyi > y0:
        x_rightqie = x1_pingyi
        y_rightqie = y1_pingyi
    else:
        x_rightqie = x2_pingyi
        y_rightqie = y2_pingyi

    cv2.circle(background, (int(y_rightqie), int(x_rightqie)), 3, (0, 0, 255), 3)
        
    dist1 = math.sqrt((right_point[1]-left_point[1])**2+(right_point[0]-left_point[0])**2) #耻骨联合左右端点距离
    dist2 = math.sqrt((right_point[1]-x_rightqie)**2+(right_point[0]-y_rightqie)**2) #耻骨联合右端点到胎头右侧切点的距离
    dist3 = math.sqrt((x_rightqie-left_point[1])**2+(y_rightqie-left_point[0])**2) #耻骨联合左端点到胎头右侧切点的距离
    
    aod = math.acos((dist1**2+dist2**2-dist3**2)/(2*dist1*dist2))/math.pi*180   ##余弦定理
    cv2.line(background, right_point,  (int(y_rightqie), int(x_rightqie)), (255, 255, 255), 3)
    return aod
    # print('AOD',round(aod,2),'（度）')


#cv2.imshow('img',background)
#cv2.waitKey()
if __name__=="__main__":
    background = np.zeros((600,600,3),np.uint8)
    fetalhead_element=((300,400),(100,250),80)
    right_point = (500,100)
    left_point = (300,100)
    cv2.circle(background, left_point, 3, (255, 0, 0), 3)
    cv2.circle(background, right_point, 3, (255, 0, 0), 3)
    cv2.line(background, right_point, left_point, (255, 255, 255), 3)
    aod_cal(background, fetalhead_element, right_point, left_point)
    cv2.imshow('img',background)
    cv2.waitKey()
    


