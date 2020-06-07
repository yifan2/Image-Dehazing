# -*- coding: utf-8 -*-

import numpy as np
import cv2

def save_img(img_list,save_name):
    save=np.hstack(tuple(img_list))
    cv2.imwrite(save_name,save)
    name=['W','A','sw','svm','B','V']
    for i in range(len(name)):
        cv2.imwrite(name[i]+'.jpg',img_list[i]*255)
    
def Atmoveil_east(img_path):
    img=cv2.imread(img_path)
    img=img.astype('float32')/255
    W=np.min(img,axis=2)
    sv=17
    A=cv2.medianBlur(W,sv)
    sw=np.abs(W-A)
    swm=cv2.medianBlur(sw,sv)
    B=A-swm
    p=0.9
    V=np.minimum(p*B,W)
    save_img([W,A,sw,swm,B,V],'no_haze_test.png')
#    cv2.imshow('A',median_w)
#    cv2.imshow('svm',swm)
#    cv2.imshow('B',B)
#    cv2.waitKey(0)

if __name__=='__main__':
#    path='./haze images/41.png'
    path='no_haze.jpg'
    Atmoveil_east(path)