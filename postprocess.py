#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:58:07 2020

@author: user
"""

import cv2
import os
import numpy as np
import copy
from PIL import Image
imgPath='./1_0_1200/1_0_1200_roi_224/20161208_584953b2ecdd81_roi.jpg'
remaskPath='./1_0_1200/1_0_1200_roi_224_results/20161208_584953b2ecdd81_roi.jpg'
save_path='./3.jpg'

def open_close(resmask, save_path):    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    resmask = cv2.morphologyEx(resmask, cv2.MORPH_CLOSE, kernel,iterations=1)    
    resmask = cv2.morphologyEx(resmask, cv2.MORPH_OPEN, kernel,iterations=1) 

#    cv2.imwrite(save_path, resmask)
    return resmask

def intcalConnectRegion(imgPath, remaskPath, save_path):
    img=cv2.imread(imgPath)
    img=open_close(img, save_path)
    oriheight = img.shape[0]
    oriwidth = img.shape[1]        
    resmask=cv2.imread(remaskPath)
    src = copy.deepcopy(resmask)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    dst = np.zeros((src.shape[0], src.shape[1]))
    ret, binary = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)
    image, contours, hirachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    c_max = []
    max_cnt = 0
    maxArea = 0.0
    for j in range(len(contours)):
        cnt = contours[j]
        tmpArea = abs(cv2.contourArea(contours[j]))
        if tmpArea > maxArea:
            if (maxArea != 0):
                c_min = []
                c_min.append(max_cnt)
                cv2.drawContours(src, c_min, -1, (0, 250, 0), -1)
            maxArea = tmpArea
            max_i = j
            max_cnt = cnt
        else:
            c_min = []
            c_min.append(cnt)
            cv2.drawContours(src, c_min, -1, (0, 250, 0), -1)
            
        c_max = max_cnt
#        cv2.drawContours(dst, contours[max_i], -1, (255, 255, 255))
#        cv2.drawContours(src, c_max, -1, (255, 255, 255), -1)

        for hang in range(src.shape[0]):
            for lie in range(src.shape[1]):
                if hang == 0 or lie == 0 or hang == src.shape[0] - 1 or lie == src.shape[1] - 1:
                    src[hang][lie] = 0

        ret2, binary2 = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)
        image, contours2, hirachy = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        num = len(contours2)
        if num > 0:
            for i in range(num):
                cnt = []
                cnt.append(contours2[i])
                cv2.drawContours(src, cnt, -1, (255, 255, 255), -1)

        resmask = src
        edgeRes = dst
        if resmask.shape[0]!=oriheight or resmask.shape[1]!=oriwidth:
            resmask = cv2.resize(resmask, (resmask.shape[1], resmask.shape[0]), interpolation=cv2.INTER_CUBIC)
        if edgeRes.shape[0]!=oriheight or edgeRes.shape[1]!=oriwidth:
            edgeRes = cv2.resize(edgeRes, (edgeRes.shape[1], edgeRes.shape[0]), interpolation=cv2.INTER_CUBIC)
#    resmask=open_close(resmask, save_path)
    cv2.imwrite(save_path, resmask)
    return resmask

#resmask=intcalConnectRegion(imgPath, remaskPath, save_path)









#resmask01=open_close(resmask, save_path)

if __name__ == '__main__':
    dir_imgPath='./-1_0_1200/-1_0_1200_roi_224/'
    dir_remaskPath='./-1_0_1200/-1_0_1200_roi_224_results/'
    dir_save_path='./-1_0_1200/-1_0_1200_roi_224_results_pro/'
    if not os.path.lexists(dir_save_path):
        os.mkdir(dir_save_path)    
    for indir in os.listdir(dir_imgPath): 
        imgPath=dir_imgPath+indir
        remaskPath=dir_remaskPath+indir
        save_path=dir_save_path+indir
        resmask=intcalConnectRegion(imgPath, remaskPath, save_path)
#        resmask01=open_close(resmask, save_path)
    dir_imgPath1='./1_0_1200/1_0_1200_roi_224/'
    dir_remaskPath1='./1_0_1200/1_0_1200_roi_224_results/'
    dir_save_path1='./1_0_1200/1_0_1200_roi_224_results_pro/'
    if not os.path.lexists(dir_save_path1):
        os.mkdir(dir_save_path1)    
    for indir in os.listdir(dir_imgPath1): 
        imgPath1=dir_imgPath1+indir
        remaskPath1=dir_remaskPath1+indir
        save_path1=dir_save_path1+indir
        resmask=intcalConnectRegion(imgPath1, remaskPath1, save_path1)
    
    
    
    
    
    
    
    