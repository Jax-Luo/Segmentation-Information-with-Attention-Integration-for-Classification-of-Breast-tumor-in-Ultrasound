A novel segmentation-to-classification scheme by adding the segmentation-based attention (SBA) information to the deep convolution network (DCNN) for breast tumors classification.

This work is being submitted to the journal **pattern recognition** which is a very good journal to learn AI.

> this is a very userful implementation of breast tumors classification based on tensorflow and keras, the model is very clear.


## Requirements
Basically, this code supports and python3.6.4, the following package should installed:
* tensorflow 1.9.0 
* keras 2.1.4
* scipy
* cv2


## Dataset
TCIs are resized into a unified 224 × 224 and are one-channel gray images, and the format is JPG. Breast tumor ultrasound classification data have 1702 images. We provide sample images and can download on here:https://pan.baidu.com/s/1g_sst_VyCrRkmx-KHz94UQ password：c096. 


## Usage

Firstly train the segmentation network and get the segmentation results. 
```
python unet.py
python postprocess.py
```

Secondly generate 'npy' data file.
```
python data_cv.py
python data_cv_03.py
```

Thirdly fine-tune the feature networks.
```
python model_feature_network.py
```

Finally train the feature aggregation network.
```
python attention_aggregation.py
```


## For testing

TCIs are used as the input of the model to predict the benign and malignant tumors.
```
python test_model.py
```


## Result
Accuracy (90.78%), Sensitivity (91.18%), Specificity (90.44%), F1-score (91.46%), and AUC (0.9549) for breast tumor classification.


**That's all, help you enjoy!**
