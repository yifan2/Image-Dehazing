# -*- coding: utf-8 -*-

import cv2
import numpy as np

def MinFilter_berman(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def eas_Berman(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  
    Dark_Channel = MinFilter_berman(V1, 7)
    bins = 2000
    ht = np.histogram(Dark_Channel, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(Dark_Channel.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[Dark_Channel >= ht[1][lmax]].max()
#    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return A
