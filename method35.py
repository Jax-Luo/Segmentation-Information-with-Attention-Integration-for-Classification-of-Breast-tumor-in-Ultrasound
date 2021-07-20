#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:41:09 2020

@author: user
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Flatten, Dense, concatenate, GlobalAveragePooling2D, Lambda, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from keras.optimizers import *
from keras.applications import resnet50
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
  
method35_00_acc=[]  
method35_01_acc=[] 
method35_02_acc=[] 
method35_03_acc=[] 
method35_04_acc=[]     
method35_acc=[]  


def train_resnet_ori(a):
    imgs_train, imgs_mask_train, category_train_ohe, mean_train = load_train_data(a)  
    imgs_test, imgs_mask_test, category_test_ohe = load_test_data(a) 
    imgs_test=imgs_test-mean_train
    print(imgs_train.shape, imgs_mask_train.shape, category_train_ohe.shape, mean_train.shape)  
    print(imgs_test.shape, imgs_mask_test.shape, category_test_ohe.shape) 
#    imgs_train_3= np.concatenate((imgs_train,imgs_train,imgs_train), axis = 3)
    #print(imgs_train_3.shape)
#    img_test_3= np.concatenate((imgs_test,imgs_test,imgs_test), axis = 3)
    #print(img_test_3.shape)

    inputs = Input((224,224,1),name='inputs')
    x_3channel=concatenate([inputs,inputs,inputs], axis = 3)
    
    base_model01 = resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=x_3channel)    
    model_resnet50 = Model(inputs=inputs,outputs=base_model01.output)    
    model_resnet50.compile(loss='categorical_crossentropy',optimizer='adam')
    resnet50_x = model_resnet50.layers[-2].output
    x = GlobalAveragePooling2D()(resnet50_x)
    prediction = Dense(2, activation='softmax', name='predictions')(x)  
    
#    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
#    x=base_model(x_3channel)
#    x=Flatten()(x)
#    prediction=Dense(2,activation='softmax')(x)

    
    model = Model(inputs=inputs, outputs=prediction)
#    model.summary()
#    print("layer nums:", len(model.layers))
    for layer in base_model01.layers:
        layer.trainable = True    

    aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                             horizontal_flip=True, fill_mode='nearest')
    model_checkpoint = ModelCheckpoint('./model35/method35_00'+str(a)+'.hdf5', monitor='loss',verbose=2, save_best_only=True, mode='min')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit_generator(aug.flow(imgs_train, category_train_ohe, batch_size=50), validation_data=(imgs_test, category_test_ohe), steps_per_epoch=35, epochs=150, verbose=2, shuffle=True, callbacks=[model_checkpoint])
    
    loss,accuracy = model.evaluate(imgs_test, category_test_ohe)
    
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    method35_00_acc.append(accuracy)
    K.clear_session()
 
def train_resnet_sei(a):
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
    inputs_model01=concatenate([inputs_00,inputs_00,inputs_01], axis = 3)
    
#    base_model01 = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
#    x=base_model01(inputs_model01)
#    x=Flatten()(x)
#    prediction=Dense(2,activation='softmax')(x)

    base_model01 = resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs_model01)    
    model_resnet50 = Model(inputs=inputs,outputs=base_model01.output)    
    model_resnet50.compile(loss='categorical_crossentropy',optimizer='adam')
    resnet50_x = model_resnet50.layers[-2].output
    x = GlobalAveragePooling2D()(resnet50_x)
    prediction = Dense(2, activation='softmax', name='predictions')(x)  

    
    model = Model(inputs=inputs, outputs=prediction)
#    model.summary()
#    print("layer nums:", len(model.layers))
    for layer in base_model01.layers:
        layer.trainable = True    

    aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                             horizontal_flip=True, fill_mode='nearest')
    model_checkpoint = ModelCheckpoint('./model35/method35_01'+str(a)+'.hdf5', monitor='loss',verbose=2, save_best_only=True, mode='min')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit_generator(aug.flow(imgs_train_2, category_train_ohe, batch_size=50), validation_data=(img_test_2, category_test_ohe), steps_per_epoch=35, epochs=150, verbose=2, shuffle=True, callbacks=[model_checkpoint])
    
    loss,accuracy = model.evaluate(img_test_2, category_test_ohe)
    
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    method35_01_acc.append(accuracy)
    K.clear_session()    

def method35_02(a):
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
    inputs_model01=concatenate([inputs_00,inputs_01,inputs_01], axis = 3)
    
    base_model01 = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x=base_model01(inputs_model01)
    x=Flatten()(x)
    prediction=Dense(2,activation='softmax')(x)

    
    model = Model(inputs=inputs, outputs=prediction)
#    model.summary()
#    print("layer nums:", len(model.layers))
    for layer in base_model01.layers:
        layer.trainable = True    

    aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                             horizontal_flip=True, fill_mode='nearest')
    model_checkpoint = ModelCheckpoint('./model35/method35_02'+str(a)+'.hdf5', monitor='loss',verbose=2, save_best_only=True, mode='min')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit_generator(aug.flow(imgs_train_2, category_train_ohe, batch_size=50), validation_data=(img_test_2, category_test_ohe), steps_per_epoch=35, epochs=150, verbose=2, shuffle=True, callbacks=[model_checkpoint])
    
    loss,accuracy = model.evaluate(img_test_2, category_test_ohe)
    
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    method35_02_acc.append(accuracy)
    K.clear_session()

def method35_03(a):
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
    inputs_02 = Lambda(lambda x:-x+1)(inputs_01)
    
    inputs_00 = Reshape((224,224,1))(inputs_00)
    inputs_01 = Reshape((224,224,1))(inputs_01)
    inputs_02 = Reshape((224,224,1))(inputs_02)
    
    inputs_model01=concatenate([inputs_00,inputs_01,inputs_02], axis = 3)
    
    base_model01 = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x=base_model01(inputs_model01)
    x=Flatten()(x)
    prediction=Dense(2,activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=prediction)
#    model.summary()
#    print("layer nums:", len(model.layers))
    for layer in base_model01.layers:
        layer.trainable = True    

    aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                             horizontal_flip=True, fill_mode='nearest')
    model_checkpoint = ModelCheckpoint('./model35/method35_03'+str(a)+'.hdf5', monitor='loss',verbose=2, save_best_only=True, mode='min')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit_generator(aug.flow(imgs_train_2, category_train_ohe, batch_size=50), validation_data=(img_test_2, category_test_ohe), steps_per_epoch=35, epochs=150, verbose=2, shuffle=True, callbacks=[model_checkpoint])
    
    loss,accuracy = model.evaluate(img_test_2, category_test_ohe)
    
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    method35_03_acc.append(accuracy)
    K.clear_session()
    

    
def slice(x,index):         
    return x[:,:,:,index]     
    
    

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

if __name__ == '__main__':

    for a in range(10):
        print('a==', a)
        train_resnet_ori(a)
    print('method35_00_acc==',method35_00_acc)
    print('averagenum_method35_00_acc==',averagenum(method35_00_acc))   

    for a in range(10):
        print('a==', a)
        train_resnet_sei(a)    
    print('method35_01_acc==',method35_01_acc)
    print('averagenum_method35_01_acc==',averagenum(method35_01_acc)) 

#    for a in range(10):
#        print('a==', a)
#        method35_02(a)    
#    print('method35_02_acc==',method35_02_acc)
#    print('averagenum_method35_02_acc==',averagenum(method35_02_acc)) 
#
#    for a in range(10):
#        print('a==', a)
#        method35_03(a)    
#    print('method35_03_acc==',method35_03_acc)
#    print('averagenum_method35_03_acc==',averagenum(method35_03_acc)) 
#
#    for a in range(10):
#        print('a==', a)
#        method35_04(a)    
#    print('method35_04_acc==',method35_04_acc)
#    print('averagenum_method35_04_acc==',averagenum(method35_04_acc))     
#    






















