# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:22:08 2019

@author: yifan
"""

import os
import cv2
import numpy as np
import math
from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import copy
from guide_berman import Guided_Berman
from airlight_eas import eas_Berman,MinFilter_berman


"""
构造KD树
"""
def get_KDtree(txt_path):
    data = np.loadtxt(txt_path)
    return KDTree(data)
    

"""
生成聚类结果
"""
def kd_tree(trans):
    h,w = trans.shape[0:2]
    tree = get_KDtree('TR1000.txt')
    trans = np.reshape(trans,(h*w,-1))
    dist,ind = tree.query(trans,k=1)
    labels = np.reshape(ind,(h,w))
    return labels

"""
功能：透射率图初始估计
参数说明：
img:
"""
def tran_eas(img,dist_from_airlight,labels,A,K=1000):
    transmission_estimation = copy.deepcopy(dist_from_airlight)
    dist_from_airlight = np.abs(dist_from_airlight)
    trans = np.zeros_like(dist_from_airlight)
    for i in range(K):
        mask = np.where(labels==i)
        if mask[0].shape[0]>0:
            max_rad = max(dist_from_airlight[mask])
            transmission_estimation[mask] /= max_rad
    trans_min = 0.1
    transmission_estimation = np.minimum(np.maximum(transmission_estimation, trans_min),1)
    trans_lower_bound = 1-np.min(img/A,axis=2)
    transmission_estimation = np.maximum(transmission_estimation,trans_lower_bound)
#    print(img.dtype,transmission_estimation.dtype,img.shape,transmission_estimation.shape)
    gimfiltR = 60 #引导滤波时半径的大小
    eps = 10**-3 #引导滤波时epsilon的值
    guided_filter = Guided_Berman(img, gimfiltR, eps)
    dst = guided_filter.filter(transmission_estimation)
#    dst = cv2.ximgproc.guidedFilter(guide=img, src=transmission_estimation, radius=16, eps=50, dDepth=-1)
    return dst
"""
参数解释：
img_path:为输入图像路径
save_path:为复原图像保存路径，默认为空
"""
def berman_dehaze(haze_img,save_path=None):
#    haze_img = cv2.imread(img_path)
    show_img = copy.deepcopy(haze_img)/float(255)
    print('show_img',show_img.dtype)
    airlight = eas_Berman(haze_img, r=81, eps=0.001, w=0.98, maxV1=0.8)/255
    h,w = haze_img.shape[0:2]
    h,w,channel = haze_img.shape
#    haze_img = haze_img.astype(np.float32)/255
    haze_img = haze_img/255
    dist_from_airlight = np.zeros_like(haze_img)
    radius = np.zeros((h,w),dtype = np.float64)
    for id in range(channel):
        dist_from_airlight[:,:,id] = haze_img[:,:,id]-airlight
        radius += np.power(dist_from_airlight[:,:,id],2)
    radius = np.sqrt(radius+0.000001)
    I_A = copy.deepcopy(radius)
    radius = cv2.merge((radius,radius,radius))
    points = dist_from_airlight/radius
    labels = kd_tree(points)
    t = tran_eas(show_img,I_A,labels,airlight)
    t = cv2.merge((t,t,t))
#    t = t.astype(np.float64)
    J = (dist_from_airlight/t+airlight)*255
#    J *= 255
    J = np.clip(J,0,255)
    J = J.astype(np.uint8)
#    print((J*255).dtype)
#    J = (255*J).astype(np.uint8)
#    print(save_path)
#    while True:
#        cv2.imshow('J',J)
#        if cv2.waitKey(10) == 27:
#            cv2.destroyAllWindows()
#            break
    if save_path:
        cv2.imwrite(save_path,J*255)
    return J

def run():
    dir_name = 'Images'
    save_dir = 'Dehazed'
    imgs = os.listdir(dir_name)
    start = 1
    for img_name in imgs:
        if img_name.endswith('png') or img_name.endswith('jpg'):
            save_path = os.path.join(save_dir,img_name)
            input_path = os.path.join(dir_name,img_name)
            haze_img = cv2.imread(input_path)
            berman_dehaze(haze_img,save_path=save_path)
            print('I have completed %s'%img_name)
            start+=1
            if start>=1:
                break
if __name__=='__main__':
#    run()
    dir_name = 'Images'
    save_dir = 'Dehazed'
    imgs = os.listdir(dir_name)
    start = 1
    for img_name in imgs:
        if img_name.endswith('png') or img_name.endswith('jpg'):
            save_path = os.path.join(save_dir,img_name)
            input_path = os.path.join(dir_name,img_name)
            haze_img = cv2.imread(input_path)
            dehazed_img = berman_dehaze(haze_img,save_path=save_path)
            print('I have completed %s'%img_name)
            start+=1
            if start>=1:
                break
    after_img = cv2.imread('Dehazed/AM_Bing_211.png')
    dehazed_img = np.clip(dehazed_img,0,255)
#    dehazed_img += 255
    show_img = dehazed_img.astype(np.uint8)
    while True:
        cv2.imshow('show',show_img)
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break
    
        
        
    
        
        
    
    
    
        