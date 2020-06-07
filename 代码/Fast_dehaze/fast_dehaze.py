# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time

def run_fast_dehaze(frame):
#step 1将图片变成浮点型
    img = frame.copy()
    H = img.astype(np.float32)/255
#    H = img/255
#step 2
    M_line = np.min(H,axis=2)#step 2
#step 3
#    radius=max(50,max(img.shape[0:2])//20)
#    print(M_line.dtype,H.shape)
    radius = (max(img.shape[0:2]))//100
    M_ave = cv2.blur(M_line,(2*radius+1,2*radius+1))
#step4
    m_ave = np.mean(M_line)
#step 5
    p = float(1-m_ave+0.9)
    p=5#e为调节的参数
    coeff = min(p*m_ave,0.9)
#    L_x=np.zeros_like(img[:,:,0])
    L_x = np.minimum(coeff*M_ave,M_line)
    A = 1/2*(np.max(H)+np.max(M_ave))
    L_x = cv2.merge((L_x,L_x,L_x))
#    cv2.imshow('Mave',M_ave)
#    cv2.imshow('l(x)',L_x)
#    cv2.waitKey()H
    ONE = np.ones_like(img)
    dehaze_img = np.zeros_like(img)
    dehaze_img = (H-L_x)/(ONE-L_x/A)
#    return dehaze_img
    dehaze_img *= 255
    dehaze_img = np.clip(dehaze_img,0,255)
    return dehaze_img.astype(np.uint8)

if __name__=='__main__':
    img_path='./haze images/42.png'
    dehaze_img=fast_dehaze(img_path)
    cv2.namedWindow('dehaze',0)
    cv2.imshow('dehaze',dehaze_img)
    cv2.waitKey(0)
#    L_line[x] = std::min(coeff * M_ave_line[x], float(M_line[x]));
    