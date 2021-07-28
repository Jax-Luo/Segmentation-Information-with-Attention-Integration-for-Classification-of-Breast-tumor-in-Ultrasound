# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:56:00 2018

@author: Huang-PC
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import load_model
from sklearn import svm
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
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from keras.utils import np_utils
from scipy import interp
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

npy_path='./npydata'

def load_onedir_train_data(n):
    
    imgs_train=np.load(npy_path + '/' + str(n) + '_imgs_train.npy')
    imgs_mask_train=np.load(npy_path + '/' + str(n) + '_imgs_mask_train.npy')

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

#    imgs_train /= 255
#    mean = imgs_train.mean(axis=0)
#    imgs_train -= mean
#
#    imgs_mask_train /= 255
#    imgs_mask_train[imgs_mask_train > 0.5] = 1
#    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    
    category_train=np.load(npy_path + '/' + str(n) + '_category_train.npy')    
    category_train_ohe = np_utils.to_categorical(category_train, 2)    

    return imgs_train, imgs_mask_train, category_train, category_train_ohe

def load_onedir_test_data(m):
    
    imgs_test=np.load(npy_path + '/' + str(m) + '_imgs_test.npy')
    imgs_mask_test=np.load(npy_path + '/' + str(m) + '_imgs_mask_test.npy')
    
    imgs_test = imgs_test.astype('float32')    
    imgs_mask_test = imgs_mask_test.astype('float32')    
    
#    imgs_test /= 255
#    mean_test = imgs_test.mean(axis=0)
#    imgs_test -= mean_test
#
#    imgs_mask_test /= 255
#    imgs_mask_test[imgs_mask_test > 0.5] = 1
#    imgs_mask_test[imgs_mask_test <= 0.5] = 0

    category_test = np.load(npy_path + '/' + str(m) + '_category_test.npy')
    category_test_ohe = np_utils.to_categorical(category_test, 2)

    return imgs_test, imgs_mask_test, category_test, category_test_ohe    

names = globals() 

def load_train_data(k):    
    str_imgs_train=[]
    str_imgs_mask_train=[]
    str_category_train=[]
    str_category_train_ohe=[]
    for n in range(10):
        if n!=k:
            print('n==',n)   
            names['imgs_train' + str(n)], names['imgs_mask_train' + str(n)], names['category_train' + str(n)], names['category_train_ohe' + str(n)]=load_onedir_train_data(n)
            print(names['imgs_train' + str(n)].shape, names['imgs_mask_train' + str(n)].shape, names['category_train' + str(n)].shape, names['category_train_ohe' + str(n)].shape)
            str_imgs_train.append('imgs_train' + str(n))
            str_imgs_mask_train.append('imgs_mask_train' + str(n))
            str_category_train.append('category_train' + str(n))
            str_category_train_ohe.append('category_train_ohe' + str(n))
    imgs_train= np.concatenate((names[str_imgs_train[0]], names[str_imgs_train[1]], names[str_imgs_train[2]], names[str_imgs_train[3]], names[str_imgs_train[4]],
                                names[str_imgs_train[5]], names[str_imgs_train[6]], names[str_imgs_train[7]], names[str_imgs_train[8]]), axis = 0)
    imgs_mask_train= np.concatenate((names[str_imgs_mask_train[0]], names[str_imgs_mask_train[1]], names[str_imgs_mask_train[2]], names[str_imgs_mask_train[3]], names[str_imgs_mask_train[4]],
                                names[str_imgs_mask_train[5]], names[str_imgs_mask_train[6]], names[str_imgs_mask_train[7]], names[str_imgs_mask_train[8]]), axis = 0)
    category_train= np.concatenate((names[str_category_train[0]], names[str_category_train[1]], names[str_category_train[2]], names[str_category_train[3]], names[str_category_train[4]],
                                names[str_category_train[5]], names[str_category_train[6]], names[str_category_train[7]], names[str_category_train[8]]), axis = 0)
    category_train_ohe= np.concatenate((names[str_category_train_ohe[0]], names[str_category_train_ohe[1]], names[str_category_train_ohe[2]], names[str_category_train_ohe[3]], names[str_category_train_ohe[4]],
                                names[str_category_train_ohe[5]], names[str_category_train_ohe[6]], names[str_category_train_ohe[7]], names[str_category_train_ohe[8]]), axis = 0)
    imgs_train /= 255
    mean = imgs_train.mean(axis=0)
#    mean = imgs_train.mean()
    imgs_train -= mean

    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    return imgs_train, imgs_mask_train, category_train, category_train_ohe, mean

def load_test_data(k):  
#    names['imgs_test' + str(k)], names['imgs_mask_test' + str(k)], names['category_test_ohe' + str(k)]=load_onedir_test_data(k)
    imgs_test, imgs_mask_test, category_test, category_test_ohe=load_onedir_test_data(k)
    imgs_test /= 255


    imgs_mask_test /= 255
    imgs_mask_test[imgs_mask_test > 0.5] = 1
    imgs_mask_test[imgs_mask_test <= 0.5] = 0
    return imgs_test, imgs_mask_test, category_test, category_test_ohe
    
#a=3
#imgs_train, imgs_mask_train, category_train_ohe = load_train_data(a)  
#imgs_test, imgs_mask_test, category_test_ohe = load_test_data(a) 
#print(imgs_train.shape, imgs_mask_train.shape, category_train_ohe.shape)  
#print(imgs_test.shape, imgs_mask_test.shape, category_test_ohe.shape)   
  
     
method_acc=[]     
method_specificity=[]
method_sensitivity=[]
method_F1=[]
method_roc_auc=[]
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

def calculate_metric01(model,imgs_test,category_test,category_test_ohe, p_ts=0.5):
    load_model01 = load_model(model)
#    imgs_test= np.concatenate((imgs_test,imgs_test,imgs_test), axis = 3)
    predicted = load_model01.predict(imgs_test)
    predicted01= predicted[:,1]

    
    predicted_1d = np.argmax(predicted, axis=1)
    loss,accuracy = load_model01.evaluate(imgs_test, category_test_ohe)

    print("\nPredicted01d softmax vector is: ")
    print(predicted_1d)

    
    print("\nclassification_report is: ")
    print(classification_report(category_test, predicted_1d))   
        
    false_positive_rate, true_positive_rate, thresholds = roc_curve(category_test, predicted01)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    tprs.append(interp(mean_fpr,false_positive_rate,true_positive_rate))
    tprs[-1][0]=0.0
    aucs.append(roc_auc)

    print(loss,accuracy)
    TP=0
    FP=0
    TN=0
    FN=0
    for m in range(len(category_test)):
        if category_test[m]==1:
            if predicted_1d[m]==1:
                TP+=1
        if category_test[m]==1:
            if predicted_1d[m]==0:
                FP+=1    
        if category_test[m]==0:
            if predicted_1d[m]==0:
                TN+=1
        if category_test[m]==0:
            if predicted_1d[m]==1:
                FN+=1       
    print(TP,FP,TN,FN)    
        
      
    specificity=TN/(TN+FP)
    sensitivity=TP/(TP+FN)  
    accuracy=(TP+TN)/(TP+TN+FN+FP)    
    F1=(TP+TP)/(TP+TP+FN+FP)   
    print(specificity,sensitivity,accuracy,F1)
    method_acc.append(accuracy)  
    method_specificity.append(specificity)
    method_sensitivity.append(sensitivity)
    method_F1.append(F1)
    method_roc_auc.append(roc_auc)
    K.clear_session() 
    
def calculate_metric02(model,imgs_test,category_test,category_test_ohe, p_ts=0.5):
    imgs_train_2= np.concatenate((imgs_train,imgs_mask_train), axis = 3)
    #print(imgs_train_3.shape)
    img_test_2= np.concatenate((imgs_test,imgs_mask_test), axis = 3)

    load_model01 = load_model(model)
#    imgs_test= np.concatenate((imgs_test,imgs_test,imgs_test), axis = 3)
    predicted = load_model01.predict(img_test_2)
    predicted01= predicted[:,1]

    
    predicted_1d = np.argmax(predicted, axis=1)
    loss,accuracy = load_model01.evaluate(img_test_2, category_test_ohe)

    print("\nPredicted01d softmax vector is: ")
    print(predicted_1d)

    
    print("\nclassification_report is: ")
    print(classification_report(category_test, predicted_1d))   
        
    false_positive_rate, true_positive_rate, thresholds = roc_curve(category_test, predicted01)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    tprs.append(interp(mean_fpr,false_positive_rate,true_positive_rate))
    tprs[-1][0]=0.0
    aucs.append(roc_auc)

    print(loss,accuracy)
    TP=0
    FP=0
    TN=0
    FN=0
    for m in range(len(category_test)):
        if category_test[m]==1:
            if predicted_1d[m]==1:
                TP+=1
        if category_test[m]==1:
            if predicted_1d[m]==0:
                FP+=1    
        if category_test[m]==0:
            if predicted_1d[m]==0:
                TN+=1
        if category_test[m]==0:
            if predicted_1d[m]==1:
                FN+=1       
    print(TP,FP,TN,FN)    
        
      
    specificity=TN/(TN+FP)
    sensitivity=TP/(TP+FN)  
    accuracy=(TP+TN)/(TP+TN+FN+FP)    
    F1=(TP+TP)/(TP+TP+FN+FP)   
    print(specificity,sensitivity,accuracy,F1)
    method_acc.append(accuracy)  
    method_specificity.append(specificity)
    method_sensitivity.append(sensitivity)
    method_F1.append(F1)
    method_roc_auc.append(roc_auc)
    K.clear_session() 
    
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

if __name__ == '__main__':

    for a in range(10):
        imgs_train, imgs_mask_train, category_train, category_train_ohe, mean_train = load_train_data(a)  
        imgs_test, imgs_mask_test, category_test, category_test_ohe = load_test_data(a) 
        imgs_test=imgs_test-mean_train
        print(imgs_train.shape, imgs_mask_train.shape, category_train_ohe.shape)  
        print(imgs_test.shape, imgs_mask_test.shape, category_test_ohe.shape)  
        img_test_2= np.concatenate((imgs_test,imgs_mask_test), axis = 3)
        model_name='./model35_7x7/softmax_semodule_method'+str(a)+'.hdf5'
        print('a==', a)
        print('model_name==',model_name)
#        calculate_metric01(model_name,imgs_test,category_test,category_test_ohe,0.5)   
        calculate_metric02(model_name,imgs_test,category_test,category_test_ohe,0.5) 
        
    plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
    std_auc=np.std(tprs,axis=0)
    plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.4f)'%mean_auc,lw=2,alpha=.8)
    std_tpr=np.std(tprs,axis=0)
    tprs_upper=np.minimum(mean_tpr+std_tpr,1)
    tprs_lower=np.maximum(mean_tpr-std_tpr,0)
    plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

#        calculate_metric01(model_name,img_test_2,category_test,category_test_ohe,0.5) 
#        calculate_metric01(method1_name,imgs_test,category_test,category_test_ohe,0.99)     
#        calculate_metric01(model_name,imgs_test,category_test,category_test_ohe,0.5)  
#        calculate_metric02(model_name,imgs_train, imgs_mask_train, category_train, category_train_ohe, imgs_test, imgs_mask_test, category_test_ohe, category_test,0.99)   

#    print('accuracy_01==',accuracy_01)
#    print('averagenum_accuracy_01==',averagenum(accuracy_01))

    print('method_acc==',method_acc)
    print('averagenum_method_acc==',averagenum(method_acc))
    
    print('method_specificity==',method_specificity)
    print('averagenum_method_specificity==',averagenum(method_specificity))

    print('method_sensitivity==',method_sensitivity)
    print('averagenum_method_sensitivity==',averagenum(method_sensitivity))

    print('method_F1==',method_F1)
    print('averagenum_method_F1==',averagenum(method_F1))

    print('method_roc_auc==',method_roc_auc)
    print('averagenum_method_roc_auc==',averagenum(method_roc_auc))



















