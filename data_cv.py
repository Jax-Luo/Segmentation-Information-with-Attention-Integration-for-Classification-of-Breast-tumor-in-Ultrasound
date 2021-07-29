# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:58:13 2019

@author: Huang-PC
"""

import os, random, shutil
import skimage.io as io

def countfile(fileDir):
    x=0
    for filename in os.listdir(fileDir):
        print (filename)
        x+=1
    return x

def count_picknumber(imgDir):
    pathDir02=os.listdir(imgDir)
    filenumber=len(pathDir02)
    rate=0.1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    return picknumber
    
def generate_backups(imgDir,labelDir,sec_imgDir_0,sec_labelDir_0):
    pathDir = os.listdir(imgDir)    #取图片的原始路径

    if not os.path.lexists(sec_imgDir_0):
        os.mkdir(sec_imgDir_0)
    if not os.path.lexists(sec_labelDir_0):
        os.mkdir(sec_labelDir_0)

    x=0        
    for filename in pathDir:
        print (filename)
        x+=1
        shutil.copyfile(imgDir+filename, sec_imgDir_0+filename)
        shutil.copyfile(labelDir+filename, sec_labelDir_0+filename)
    print(x)    


def moveFile(sec_imgDir_0,sec_labelDir_0,sec_imgDir_n,sec_labelDir_n, picknumber):
    
    if not os.path.lexists(sec_imgDir_n):
        os.mkdir(sec_imgDir_n)
    if not os.path.lexists(sec_labelDir_n):
        os.mkdir(sec_labelDir_n)

    pathDir02=os.listdir(sec_imgDir_0)
#    filenumber=len(pathDir02)
#    rate=0.1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    print('picknumber=',picknumber)
    sample = random.sample(pathDir02, picknumber)  #随机选取picknumber数量的样本图片
    print (sample)

    for name in sample:
        shutil.move(sec_imgDir_0+name, sec_imgDir_n+name)
        shutil.move(sec_labelDir_0+name, sec_labelDir_n+name)
    return

def pre_generate_dir():
    if not os.path.lexists('./sample_random/'):
        os.mkdir('./sample_random/')

    if not os.path.lexists('./sample_random/0/'):
        os.mkdir('./sample_random/0/')
    if not os.path.lexists('./sample_random/1/'):
        os.mkdir('./sample_random/1/')
    if not os.path.lexists('./sample_random/2/'):
        os.mkdir('./sample_random/2/')    
    if not os.path.lexists('./sample_random/3/'):
        os.mkdir('./sample_random/3/')   
    if not os.path.lexists('./sample_random/4/'):
        os.mkdir('./sample_random/4/')  
    if not os.path.lexists('./sample_random/5/'):
        os.mkdir('./sample_random/5/')   
    if not os.path.lexists('./sample_random/6/'):
        os.mkdir('./sample_random/6/')   
    if not os.path.lexists('./sample_random/7/'):
        os.mkdir('./sample_random/7/')   
    if not os.path.lexists('./sample_random/8/'):
        os.mkdir('./sample_random/8/')   
    if not os.path.lexists('./sample_random/9/'):
        os.mkdir('./sample_random/9/')   
        
def generate_sec_dir(a):

    if not os.path.lexists('./sample_random/'+a+'/bad/'):
        os.mkdir('./sample_random/'+a+'/bad/')    
    if not os.path.lexists('./sample_random/'+a+'/good/'):
        os.mkdir('./sample_random/'+a+'/good/')   
        
    sec_imgDir_bad='./sample_random/'+a+'/bad/'+'image/'
    sec_labelDir_bad='./sample_random/'+a+'/bad/'+'label/'        
    sec_imgDir_good='./sample_random/'+a+'/good/'+'image/'
    sec_labelDir_good='./sample_random/'+a+'/good/'+'label/' 
        
    if not os.path.lexists(sec_imgDir_bad):
        os.mkdir(sec_imgDir_bad) 
    if not os.path.lexists(sec_labelDir_bad):
        os.mkdir(sec_labelDir_bad)       
         
    if not os.path.lexists(sec_imgDir_good):
        os.mkdir(sec_imgDir_good) 
    if not os.path.lexists(sec_labelDir_good):
        os.mkdir(sec_labelDir_good)         
        
#pre_generate_dir()       
#generate_sec_dir('0')        
        
def generate_cv():
    pre_generate_dir()
    
    imgDir_1= "./1_0_1200/1_0_1200_roi_224/"
    labelDir_1= "./1_0_1200/1_0_1200_roi_224_results_pro/"
    sec_imgDir_bad_0='./sample_random/0/bad/image/'
    sec_labelDir_bad_0='./sample_random/0/bad/label/'
    if not os.path.lexists('./sample_random/0/bad/'):
        os.mkdir('./sample_random/0/bad/')    
    if not os.path.lexists(sec_imgDir_bad_0):
        os.mkdir(sec_imgDir_bad_0) 
    if not os.path.lexists(sec_labelDir_bad_0):
        os.mkdir(sec_labelDir_bad_0)     

    imgDir_0= "./-1_0_1200/-1_0_1200_roi_224/"
    labelDir_0= "./-1_0_1200/-1_0_1200_roi_224_results_pro/"
    sec_imgDir_good_0='./sample_random/0/good/image/'
    sec_labelDir_good_0='./sample_random/0/good/label/'
    if not os.path.lexists('./sample_random/0/good/'):
        os.mkdir('./sample_random/0/good/')
    if not os.path.lexists(sec_imgDir_good_0):
        os.mkdir(sec_imgDir_good_0)
    if not os.path.lexists(sec_labelDir_good_0):
        os.mkdir(sec_labelDir_good_0)
    
    picknumber_1=count_picknumber(imgDir_1)
    print('dddddddd=',picknumber_1)
    generate_backups(imgDir_1,labelDir_1,sec_imgDir_bad_0,sec_labelDir_bad_0)
    for n in range(9):
        sec_imgDir_bad_n='./sample_random/'+str(n+1)+'/bad/'+'image/'
        sec_labelDir_bad_n='./sample_random/'+str(n+1)+'/bad/'+'label/'
        if not os.path.lexists('./sample_random/'+str(n+1)+'/bad/'):
            os.mkdir('./sample_random/'+str(n+1)+'/bad/') 
        moveFile(sec_imgDir_bad_0,sec_labelDir_bad_0,sec_imgDir_bad_n,sec_labelDir_bad_n,picknumber_1)    

    picknumber_0=count_picknumber(imgDir_0)
    generate_backups(imgDir_0,labelDir_0,sec_imgDir_good_0,sec_labelDir_good_0)
    for n in range(9):
        sec_imgDir_good_n='./sample_random/'+str(n+1)+'/good/'+'/image/'
        sec_labelDir_good_n='./sample_random/'+str(n+1)+'/good/'+'/label/'
        if not os.path.lexists('./sample_random/'+str(n+1)+'/good/'):
            os.mkdir('./sample_random/'+str(n+1)+'/good/') 
        moveFile(sec_imgDir_good_0,sec_labelDir_good_0,sec_imgDir_good_n,sec_labelDir_good_n,picknumber_0)

if __name__ == '__main__':    
    generate_cv()    



