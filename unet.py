# -*- coding:utf-8 -*-

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class myUnet(object):
	def __init__(self, img_rows = 224, img_cols = 224):
		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):
        
#		mydata = dataProcess(self.img_rows, self.img_cols, aug_merge_path="./aug_merge", aug_train_path="./aug_train",
#                         aug_label_path="./aug_label", test_path = "./data/test", npy_path="./npydata")
#		imgs_train, imgs_mask_train = mydata.load_train_data()
#		imgs_test = mydata.load_test_data()
#		return imgs_train, imgs_mask_train, imgs_test
		npy_path='./npydata'
		imgs_train = np.load(npy_path + '/augimgs_train.npy')
		imgs_mask_train = np.load(npy_path + '/augimgs_mask_train.npy')
		imgs_test = np.load(npy_path + '/imgs_test.npy')
		imgs_mask_test = np.load(npy_path + '/imgs_mask_test.npy')
		print("imgs_mask_train: ------")
    
		imgs_train = imgs_train.astype('float32')
		imgs_test = imgs_test.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_mask_test = imgs_mask_test.astype('float32')
		imgs_train /= 255
		imgs_test /= 255
		mean = imgs_train.mean(axis=0)
		mean_test = imgs_test.mean(axis=0)
		imgs_train -= mean
		imgs_test -= mean_test
    
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
    
		imgs_mask_test /= 255
		imgs_mask_test[imgs_mask_test > 0.5] = 1
		imgs_mask_test[imgs_mask_test <= 0.5] = 0
    
		category_train = np.load(npy_path + '/augcategory_train.npy')
		category_train_ohe = np_utils.to_categorical(category_train, 2)
    
		category_test = np.load(npy_path + '/category_test.npy')
		category_test_ohe = np_utils.to_categorical(category_test, 2)
		print(category_test[8])
		category_test_ohe = np_utils.to_categorical(category_test, 2)
    
		return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test

	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(filters=64, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print("conv1 shape:",conv1.shape)
		print("weight_1 shape:", )
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print("conv1 shape:",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		print("conv4 shape:",conv4.shape)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		print("conv4 shape:", conv4.shape)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		print("pool4 shape:",pool4.shape)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		print("conv5 shape: ", conv5.shape)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		print("conv5 shape: ", conv5.shape)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		print("up6 shape: ",up6.shape)
		merge6 = concatenate([drop4,up6], axis = 3)  # 合并输出
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		print('conv6 shape: ', conv6.shape)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		print('conv6 shape: ', conv6.shape)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		print('up7 shape: ',up7.shape)
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		print('conv7 shape: ', up7.shape)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		print("conv7 shape: ", conv7.shape)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		print('up8 shape: ', up8.shape)
		merge8 = concatenate([conv2,up8],axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		print("conv8 shape: ",conv8.shape)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		print("conv8 shape: ", conv8.shape)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		print("up9 shape: ",up9.shape)
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		print("conv9 shape: ", conv9.shape)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print("conv9 shape: ", conv9.shape)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print("conv9 shape: ", conv9.shape)

		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		print("conv10 shape: ", conv10.shape)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

		return model

	def train(self):
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=2, save_best_only=True)
		print('Fitting model...')
		#model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, verbose=1,validation_split=0.2,
				  #shuffle=True, callbacks=[model_checkpoint])
		#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=5, verbose=1,validation_split=0.2,
				  #shuffle=True, callbacks=[model_checkpoint])
		model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=20, verbose=2,validation_split=0.2,
				  shuffle=True, callbacks=[model_checkpoint])
     
# 对一张图片的语义分割
def predict_img(model, img_path="./data/test/38.bmp", save_path="./results/38.bmp"):
    """
    :param img_path: 需要进行分割的图片
    :param save_path: 保存的地址和名
    """
    img = load_img(img_path, grayscale=True)  # 读入图片，并把它转换为灰度图
    imgdata = img_to_array(img)  # 将图片转换成数组
    
    imgs_test = imgdata.astype('float32')
    imgs_test /= 255
    mean = imgs_test.mean(axis=0)
    imgs_test -= mean

    img_test = np.zeros((1, imgs_test.shape[0], imgs_test.shape[1], 1))
    img_test[0] = imgs_test
    imgs_mask_test = model.predict(img_test, batch_size=1, verbose=1)
    imgs_mask_test[imgs_mask_test > 0.5] = 1
    imgs_mask_test[imgs_mask_test <= 0.5] = 0
    img = array_to_img(imgs_mask_test[0])
    img.save(save_path)


def load_train_data():
    print('-' * 30)
    print('load train images...')
    print('-' * 30)
    npy_path='./npydata'
    imgs_train = np.load(npy_path + '/augimgs_train.npy')
    imgs_mask_train = np.load(npy_path + '/augimgs_mask_train.npy')
    imgs_test = np.load(npy_path + '/imgs_test.npy')
    imgs_mask_test = np.load(npy_path + '/imgs_mask_test.npy')
    print("imgs_mask_train: ------")

    imgs_train = imgs_train.astype('float32')
    imgs_test = imgs_test.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_train /= 255
    imgs_test /= 255
    mean = imgs_train.mean(axis=0)
    mean_test = imgs_test.mean(axis=0)
    imgs_train -= mean
    imgs_test -= mean_test
    
    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    
    imgs_mask_test /= 255
    imgs_mask_test[imgs_mask_test > 0.5] = 1
    imgs_mask_test[imgs_mask_test <= 0.5] = 0
    
    category_train = np.load(npy_path + '/augcategory_train.npy')
    category_train_ohe = np_utils.to_categorical(category_train, 2)
    
    category_test = np.load(npy_path + '/category_test.npy')
    category_test_ohe = np_utils.to_categorical(category_test, 2)
    print(category_test[8])
    category_test_ohe = np_utils.to_categorical(category_test, 2)
    
    return imgs_train, imgs_mask_train, category_train_ohe, imgs_test, imgs_mask_test, category_test_ohe

    

if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
#	imgs_train,imgs_mask_train,category_train_ohe,imgs_test,imgs_mask_test,category_test_ohe=load_train_data()
#	testPath = "./1_0_1200_roi_resize_20190912/1_0_1200_roi_resize/"
	testPath = "./-1_0_1200_roi_resize_20190912/-1_0_1200_roi_resize/"
	savePath = testPath[:-1]+'_results/'
	if not os.path.lexists(savePath):
		os.mkdir(savePath)
	allpaths = os.listdir(testPath)
	myunet = myUnet()
	model = myunet.get_unet()  # 构造原有的网络
	model.load_weights('unet.hdf5')  # 输入训练好的权重
	for i in range(len(allpaths)):
		predict_img(model, testPath+allpaths[i], savePath+allpaths[i])
#loss,accuracy = model.evaluate(imgs_train, imgs_mask_train)
#
#print('\ntest loss',loss)
#print('accuracy',accuracy)

