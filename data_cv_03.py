# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:54:44 2018

@author: Huang-PC
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import platform
import matplotlib.pyplot as plt
from skimage import io
from keras.utils import np_utils

def create_train_data(n):
    i = 0
    print('-' * 30)
    print('creating test image')
    print('-' * 30)
    count = 0
    img_count=0
    img_type='jpg'
    npy_path='./npydata'
    train_benign_path="./sample_random/"+str(n)+"/good/image"
    train_benign_mask_path="./sample_random/"+str(n)+"/good/label"
    train_malignant_path="./sample_random/"+str(n)+"/bad/image"
    train_malignant_mask_path="./sample_random/"+str(n)+"/bad/label"

    
    for indir in os.listdir(train_benign_path):
        img_count+=1

    for indir in os.listdir(train_malignant_path):
        img_count+=1
        
    print(img_count)
    imgdatas = np.ndarray((img_count, 224, 224, 1), dtype=np.uint8)
    imglabels = np.ndarray((img_count, 224, 224, 1), dtype=np.uint8)
    categorylabels = np.ndarray((img_count,1), dtype=np.uint8)
    name=[]
    
    for indir in os.listdir(train_benign_path):
        trainPath = os.path.join(train_benign_path, indir)
        labelPath = os.path.join(train_benign_mask_path, indir)
        img = load_img(trainPath, grayscale=True)
        label = load_img(labelPath, grayscale=True)
        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        categorylabels[i] = 0
        name.append(indir)
#        if i % 100 == 0:
#            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print(trainPath)
    
    for indir in os.listdir(train_malignant_path):
        trainPath = os.path.join(train_malignant_path, indir)
        labelPath = os.path.join(train_malignant_mask_path, indir)
        img = load_img(trainPath, grayscale=True)
        label = load_img(labelPath, grayscale=True)
        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        categorylabels[i] = 1
        name.append(indir)
#        if i % 100 == 0:
#            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print(trainPath)
#    np.save(npy_path + '/' + str(n) + '_augimgs_train.npy', imgdatas)
    np.save(npy_path + '/' + str(n) + '_imgs_train.npy', imgdatas)
    np.save(npy_path + '/' + str(n) + '_imgs_mask_train.npy', imglabels)
    np.save(npy_path + '/' + str(n) + '_category_train.npy', categorylabels)
    np.save(npy_path + '/' + str(n) + '_name_train.npy', name)
    print('Saving to .npy files done.')

def create_test_data(n):
    i = 0
    print('-' * 30)
    print('creating test image')
    print('-' * 30)
    count = 0
    img_count=0
    img_type='jpg'
    npy_path='./npydata'
    test_benign_path="./sample_random/"+str(n)+"/good/image"
    test_benign_mask_path="./sample_random/"+str(n)+"/good/label"
    test_malignant_path="./sample_random/"+str(n)+"/bad/image"
    test_malignant_mask_path="./sample_random/"+str(n)+"/bad/label"

    
    for indir in os.listdir(test_benign_path):
        img_count+=1

    for indir in os.listdir(test_malignant_path):
        img_count+=1
        
    print(img_count)
    imgdatas = np.ndarray((img_count, 224, 224, 1), dtype=np.uint8)
    imglabels = np.ndarray((img_count, 224, 224, 1), dtype=np.uint8)
    categorylabels = np.ndarray((img_count,1), dtype=np.uint8)
    name=[]
    
    for indir in os.listdir(test_benign_path):
        trainPath = os.path.join(test_benign_path, indir)
        labelPath = os.path.join(test_benign_mask_path, indir)
        img = load_img(trainPath, grayscale=True)
        label = load_img(labelPath, grayscale=True)
        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        categorylabels[i] = 0
        name.append(indir)
#        if i % 100 == 0:
#            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print(trainPath)
    
    for indir in os.listdir(test_malignant_path):
        trainPath = os.path.join(test_malignant_path, indir)
        labelPath = os.path.join(test_malignant_mask_path, indir)
        img = load_img(trainPath, grayscale=True)
        label = load_img(labelPath, grayscale=True)
        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        categorylabels[i] = 1
        name.append(indir)
#        if i % 100 == 0:
#            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print(trainPath)
#    np.save(npy_path + '/' + str(n) + '_augimgs_train.npy', imgdatas)
    np.save(npy_path + '/' + str(n) + '_imgs_test.npy', imgdatas)
    np.save(npy_path + '/' + str(n) + '_imgs_mask_test.npy', imglabels)
    np.save(npy_path + '/' + str(n) + '_category_test.npy', categorylabels)
    np.save(npy_path + '/' + str(n) + '_name_test.npy', name)
    print('Saving to .npy files done.')
    

def generate03():
    if not os.path.lexists('./npydata'):
        os.mkdir('./npydata')
    for n in range(10):
        print('n==',n)  
        create_train_data(n)
        create_test_data(n)
    
generate03()    

