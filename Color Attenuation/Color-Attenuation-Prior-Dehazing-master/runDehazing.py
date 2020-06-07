# -*- coding:utf-8 -*-
import sys,os
import cv2
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

from GuidedFilter import GuidedFilter

def calDepthMap(I, r):
    #计算深度图

    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:,:,1] / 255.0
    v = hsvI[:,:,2] / 255.0
    #cv2.imshow("hsvI",hsvI)
    #cv2.waitKey()

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))

    output =  0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    outputPixel = output
    output = scipy.ndimage.filters.minimum_filter(output,(r,r))
    outputRegion = output
    cv2.imwrite("data/vsFeature.jpg", outputRegion*255 )
    #cv2.imshow("outputRegion",outputRegion)
    #cv2.waitKey()
    return outputRegion, outputPixel

def estA(img, Jdark):
    #估计大气环境光A的值


    h,w,c = img.shape
    if img.dtype == np.uint8:
        img = np.float32(img) / 255
    
    # Compute number for 0.1% brightest pixels
    n_bright = int(np.ceil(0.001*h*w))
    #  Loc contains the location of the sorted pixels
    reshaped_Jdark = Jdark.reshape(1,-1)
    Y = np.sort(reshaped_Jdark) 
    Loc = np.argsort(reshaped_Jdark)
    
    # column-stacked version of I
    Ics = img.reshape(1, h*w, 3)
    ix = img.copy()
    dx = Jdark.reshape(1,-1)
    
    # init a matrix to store candidate airlight pixels
    Acand = np.zeros((1, n_bright, 3), dtype=np.float32)
    # init matrix to store largest norm arilight
    Amag = np.zeros((1, n_bright, 1), dtype=np.float32)
    
    # Compute magnitudes of RGB vectors of A
    for i in xrange(n_bright):
        x = Loc[0,h*w-1-i]
        ix[x/w, x%w, 0] = 0
        ix[x/w, x%w, 1] = 0
        ix[x/w, x%w, 2] = 1
        
        Acand[0, i, :] = Ics[0, Loc[0, h*w-1-i], :]
        Amag[0, i] = np.linalg.norm(Acand[0,i,:])
    
    # Sort A magnitudes
    reshaped_Amag = Amag.reshape(1,-1)
    Y2 = np.sort(reshaped_Amag) 
    Loc2 = np.argsort(reshaped_Amag)
    # A now stores the best estimate of the airlight
    if len(Y2) > 20:
        A = Acand[0, Loc2[0, n_bright-19:n_bright],:]
    else:
        A = Acand[0, Loc2[0,n_bright-len(Y2):n_bright],:]
    
    # finds the max of the 20 brightest pixels in original image
    print A

    #cv2.imshow("brightest",ix)
    #cv2.waitKey()
    cv2.imwrite("data/position_of_the_atmospheric_light.png", ix*255)
    
    return A

if __name__ == "__main__":
    #参数设置
    inputImagePath = "data/input.png" #输入图片路径
    r = 15 #最小值滤波时的框的大小
    beta = 1.0 #散射系数
    gimfiltR = 60 #引导滤波时半径的大小
    eps = 10**-3 #引导滤波时epsilon的值


    I = cv2.imread(inputImagePath)
    dR,dP = calDepthMap(I, r)
    guided_filter = GuidedFilter(I, gimfiltR, eps)
    refineDR = guided_filter.filter(dR)
    tR = np.exp(-beta * refineDR)
    tP = np.exp(-beta * dP)

    cv2.imwrite("data/originalDepthMap.png", dR*255)
    cv2.imwrite("data/refineDepthMap.png", refineDR*255)
    cv2.imwrite("data/transmission.png", tR*255)

    a = estA(I, dR)

    if I.dtype == np.uint8:
        I = np.float32(I) / 255

    h,w,c = I.shape
    J = np.zeros((h, w, c), dtype=np.float32)

    J[:,:,0] = I[:,:,0] - a[0,0]
    J[:,:,1] = I[:,:,1] - a[0,1]
    J[:,:,2] = I[:,:,2] - a[0,2]

    t = tR
    t0, t1 = 0.05, 1
    t = t.clip(t0, t1)

    J[:, :, 0] = J[:, :, 0]  / t
    J[:, :, 1] = J[:, :, 1]  / t
    J[:, :, 2] = J[:, :, 2]  / t

    J[:, :, 0] = J[:, :, 0]  + a[0, 0]
    J[:, :, 1] = J[:, :, 1]  + a[0, 1]
    J[:, :, 2] = J[:, :, 2]  + a[0, 2]

    cv2.imwrite("data/"+str(r)+"_beta"+str(beta)+".png", J*255)
