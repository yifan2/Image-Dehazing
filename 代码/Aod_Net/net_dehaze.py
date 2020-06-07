import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
#from skimage.measure import compare_ssim
#from skimage.measure import compare_psnr
import cv2
import os

def get_imglist(img_path):
    img_list=os.listdir(img_path)
    return [os.path.join(os.path.dirname(img_path),img_list[i]) for i in range(len(img_list))]

def ssim_score(imageA,imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    grayA=grayA.astype('float')
    grayB=grayB.astype('float')
#    print(grayA.dtype,grayB.dtype)
    score = compare_ssim(grayA, grayB)
#    diff = (diff * 255).astype("uint8")
#    print(score)
    return score

def psnr_score(imageA,imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score = compare_psnr(grayA, grayB)
    return score

def dehaze_image(data_hazy):
#    print(image_path)
#    data_hazy = Image.open(image_path)
#    data_hazy = (np.asarray(data_hazy)/255.0)
    with torch.no_grad():
        data_hazy = data_hazy/255
        data_hazy = torch.from_numpy(data_hazy).float()
        data_hazy = data_hazy.permute(2,1,0)
        data_hazy = data_hazy.cuda().unsqueeze(0)
        dehaze_net = net.dehaze_net().cuda()
        current_path = os.path.abspath(__file__)
        father_path = os.path.dirname(current_path)
        model_path = os.path.join(father_path,'snapshots/dehazer.pth')
        print(model_path)
        dehaze_net.load_state_dict(torch.load(model_path))
        clean_image = dehaze_net(data_hazy)
    #    torch_data.numpy()
        clean_image = clean_image.cpu().numpy()
        print(type(clean_image))
        clean_image = np.squeeze(clean_image, axis=0)
        clean_image = clean_image.transpose((2,1,0))*255
        clean_image = np.clip(clean_image,0,255)
    return clean_image.astype(np.uint8)
#    torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "result_baidu/" + os.path.basename(image_path))
    
    
    
if __name__ == '__main__':
    dehaze_video('biguiyuan.mp4','aod-net.avi')