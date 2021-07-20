#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:57:58 2020
@author: user
"""


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Flatten, Dense, concatenate, GlobalAveragePooling2D, Lambda, Reshape, Activation, multiply
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from keras.optimizers import *
from keras.applications import resnet50, xception
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as keras
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

npy_path='./npydata'

def load_onedir_train_data(n):
    
    imgs_train=np.load(npy_path + '/' + str(n) + '_imgs_train.npy')
    imgs_mask_train=np.load(npy_path + '/' + str(n) + '_imgs_mask_train.npy')

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

#    imgs_train /= 255
#    mean = imgs_train.mean(axis=0)
##    mean = imgs_train.mean()
#    imgs_train -= mean
#
#    imgs_mask_train /= 255
#    imgs_mask_train[imgs_mask_train > 0.5] = 1
#    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    
    category_train=np.load(npy_path + '/' + str(n) + '_category_train.npy')    
    category_train_ohe = np_utils.to_categorical(category_train, 2)    

    return imgs_train, imgs_mask_train, category_train_ohe

def load_onedir_test_data(m):
    
    imgs_test=np.load(npy_path + '/' + str(m) + '_imgs_test.npy')
    imgs_mask_test=np.load(npy_path + '/' + str(m) + '_imgs_mask_test.npy')
    
    imgs_test = imgs_test.astype('float32')    
    imgs_mask_test = imgs_mask_test.astype('float32')    
    
#    imgs_test /= 255
#    mean_test = imgs_test.mean(axis=0)
##    mean_test = imgs_test.mean()
#    imgs_test -= mean_test
#
#    imgs_mask_test /= 255
#    imgs_mask_test[imgs_mask_test > 0.5] = 1
#    imgs_mask_test[imgs_mask_test <= 0.5] = 0

    category_test = np.load(npy_path + '/' + str(m) + '_category_test.npy')
    category_test_ohe = np_utils.to_categorical(category_test, 2)

    return imgs_test, imgs_mask_test, category_test_ohe    

names = globals() 

def load_train_data(k):    
    str_imgs_train=[]
    str_imgs_mask_train=[]
    str_category_train_ohe=[]
    for n in range(10):
        if n!=k:
            print('n==',n)   
            names['imgs_train' + str(n)], names['imgs_mask_train' + str(n)], names['category_train_ohe' + str(n)]=load_onedir_train_data(n)
            print(names['imgs_train' + str(n)].shape, names['imgs_mask_train' + str(n)].shape, names['category_train_ohe' + str(n)].shape)
            str_imgs_train.append('imgs_train' + str(n))
            str_imgs_mask_train.append('imgs_mask_train' + str(n))
            str_category_train_ohe.append('category_train_ohe' + str(n))
    imgs_train= np.concatenate((names[str_imgs_train[0]], names[str_imgs_train[1]], names[str_imgs_train[2]], names[str_imgs_train[3]], names[str_imgs_train[4]],
                                names[str_imgs_train[5]], names[str_imgs_train[6]], names[str_imgs_train[7]], names[str_imgs_train[8]]), axis = 0)
    imgs_mask_train= np.concatenate((names[str_imgs_mask_train[0]], names[str_imgs_mask_train[1]], names[str_imgs_mask_train[2]], names[str_imgs_mask_train[3]], names[str_imgs_mask_train[4]],
                                names[str_imgs_mask_train[5]], names[str_imgs_mask_train[6]], names[str_imgs_mask_train[7]], names[str_imgs_mask_train[8]]), axis = 0)
    category_train_ohe= np.concatenate((names[str_category_train_ohe[0]], names[str_category_train_ohe[1]], names[str_category_train_ohe[2]], names[str_category_train_ohe[3]], names[str_category_train_ohe[4]],
                                names[str_category_train_ohe[5]], names[str_category_train_ohe[6]], names[str_category_train_ohe[7]], names[str_category_train_ohe[8]]), axis = 0)
    imgs_train /= 255
    mean = imgs_train.mean(axis=0)
#    mean = imgs_train.mean()
    imgs_train -= mean

    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    return imgs_train, imgs_mask_train, category_train_ohe, mean

def load_test_data(k):  
#    names['imgs_test' + str(k)], names['imgs_mask_test' + str(k)], names['category_test_ohe' + str(k)]=load_onedir_test_data(k)
    imgs_test, imgs_mask_test, category_test_ohe=load_onedir_test_data(k)
    imgs_test /= 255


    imgs_mask_test /= 255
    imgs_mask_test[imgs_mask_test > 0.5] = 1
    imgs_mask_test[imgs_mask_test <= 0.5] = 0
    return imgs_test, imgs_mask_test, category_test_ohe
    
#a=3
#imgs_train, imgs_mask_train, category_train_ohe = load_train_data(a)  
#imgs_test, imgs_mask_test, category_test_ohe = load_test_data(a) 
#print(imgs_train.shape, imgs_mask_train.shape, category_train_ohe.shape)  
#print(imgs_test.shape, imgs_mask_test.shape, category_test_ohe.shape)   
  
method35_softmax_semodule_acc=[]     
method35_softmax_semodule_specificity=[]
method35_softmax_semodule_sensitivity=[]
method35_softmax_semodule_F1=[]
method35_softmax_semodule_roc_auc=[]


   
    
def slice(x,index):         
    return x[:,:,:,index]     
    
def train_attention_aggregation(a):
    model00='./model35/method35_00'+str(a)+'.hdf5'
    model01='./model35/method35_01'+str(a)+'.hdf5'
    imgs_train, imgs_mask_train, category_train_ohe, mean_train = load_train_data(a)  
    imgs_test, imgs_mask_test, category_test_ohe = load_test_data(a) 
    imgs_test=imgs_test-mean_train
    print(imgs_train.shape, imgs_mask_train.shape, category_train_ohe.shape)  
    print(imgs_test.shape, imgs_mask_test.shape, category_test_ohe.shape) 
    imgs_train_2= np.concatenate((imgs_train,imgs_mask_train), axis = 3)
    #print(imgs_train_3.shape)
    img_test_2= np.concatenate((imgs_test,imgs_mask_test), axis = 3)
    #print(img_test_3.shape)

    inputs = Input((224,224,2),name='inputs')
    inputs_00 = Lambda(slice,output_shape=(224,224,1),arguments={'index':0})(inputs) 
    inputs_01 = Lambda(slice,output_shape=(224,224,1),arguments={'index':1})(inputs)
    
    inputs_00 = Reshape((224,224,1))(inputs_00)
    inputs_01 = Reshape((224,224,1))(inputs_01)
    print(inputs.shape, inputs_00.shape)
    print('--------')
#    inputs_model00=concatenate([inputs_00,inputs_00,inputs_00], axis = 3)
    inputs_model00=inputs_00
    inputs_model01=concatenate([inputs_00,inputs_00,inputs_01], axis = 3)
    print(inputs_model00.shape, inputs_model01.shape)    
    load_model00 = load_model(model00)
    load_model01 = load_model(model01)    
#    print(load_model01.summary())
    model00=Model(inputs=load_model00.input, outputs=load_model00.get_layer('activation_49').output)
    model01=Model(inputs=load_model01.input, outputs=load_model01.get_layer('activation_49').output)

#    x_00 = model00.layers[-1].output
#    x_01 = model01.layers[-1].output
    x_00=model00(inputs_model00)
    x_01=model01(inputs_model01)
    
    x = concatenate([x_00,x_01], axis = 3)
    
    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=4096 // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=4096)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape(([4096]))(excitation)

    scale = multiply([x, excitation])
    scale = GlobalAveragePooling2D()(scale)
    
    prediction = Dense(2, activation='softmax', name='predictions')(scale)    

    model = Model(inputs=inputs, outputs=prediction)
    
    for layer in model00.layers:
        layer.trainable = False 
    for layer in model01.layers:
        layer.trainable = False 
        
    print(model.summary())
    
    aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                             horizontal_flip=True, fill_mode='nearest')
    model_checkpoint = ModelCheckpoint('./model35_7x7/softmax_semodule_method'+str(a)+'.hdf5', monitor='loss',verbose=2, save_best_only=True, mode='min')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit_generator(aug.flow(imgs_train_2, category_train_ohe, batch_size=30), validation_data=(img_test_2, category_test_ohe), steps_per_epoch=60, epochs=40, verbose=2, shuffle=True, callbacks=[model_checkpoint])
    
    loss,accuracy = model.evaluate(img_test_2, category_test_ohe)
    
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    method35_softmax_semodule_acc.append(accuracy)
    K.clear_session()     


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

if __name__ == '__main__':

#    for a in range(10):
#        print('a==', a)
#        method22_00(a)
#    print('method22_00_acc==',method22_00_acc)
#    print('averagenum_method22_00_acc==',averagenum(method22_00_acc))   
#
#    for a in range(10):
#        print('a==', a)
#        method22_01(a)    
#    print('method22_01_acc==',method22_01_acc)
#    print('averagenum_method22_01_acc==',averagenum(method22_01_acc)) 
    
    for a in range(10):
#    a=8
        print('a==', a)
        train_attention_aggregation(a)    
    print('method35_softmax_semodule_acc==',method35_softmax_semodule_acc)
    print('averagenum_method35_softmax_semodule_acc==',averagenum(method35_softmax_semodule_acc))  


