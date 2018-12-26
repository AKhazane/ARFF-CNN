from functools import partial
import keras
from keras.layers import *
#from keras.layers import Merge
from keras.engine import Model
from keras.optimizers import Adam
#from keras.utils import multi_gpu_model
#from helper import create_convolution_block, concatenate
from metrics import dice_coefficient
# import numpy as np
import tensorflow as tf 
import nibabel as nib
import pdb


def unet(inputShape=(1,1,256,320,256)):
       
    # paddedShape = (data_ch.shape[1]+2, data_ch.shape[2]+2, data_ch.shape[3]+2, data_ch.shape[4])

    #initial padding
    #    pdb.set_trace()
        #pdb.set_trace() 
#        pdb.set_trace()
        inputs = Input(batch_shape=inputShape)

        conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(inputs)
        #print "conv1 shape:",conv1.shape
        conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv1)
        #print "conv1 shape:",conv1.shape
        pool1 = MaxPooling3D(pool_size=2, data_format='channels_first')(conv1)
        #print "pool1 shape:",pool1.shape

        conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool1)
        #print "conv2 shape:",conv2.shape
        conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv2)
        #print "conv2 shape:",conv2.shape
        pool2 = MaxPooling3D(pool_size=2, data_format='channels_first')(conv2)
        #print "pool2 shape:",pool2.shape

        conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool2)
        #print "conv3 shape:",conv3.shape
        conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv3)
        #print "conv3 shape:",conv3.shape
        pool3 = MaxPooling3D(pool_size=2, data_format='channels_first')(conv3)
        #print "pool3 shape:",pool3.shape

        conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool3)
        conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=2, data_format='channels_first')(drop4)

        conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool4)
        conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size=2, data_format='channels_first')(drop5))
        merge6 = concatenate([drop4,up6], axis = 1)
        conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge6)
        conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv6)
#        drop6 = Dropout(0.5)(conv6) 

        up7 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size=2, data_format='channels_first')(conv6))
        merge7 = concatenate([conv3,up7], axis = 1)
        conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge7)
        conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv7)
#        drop7 = Dropout(0.5)(conv)

        up8 = Conv3D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size=2, data_format='channels_first')(conv7))
        merge8 = concatenate([conv2,up8],  axis = 1)
        conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge8)
        conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv8)
#        drop8 = Dropout(0.5)(conv8) 

        up9 = Conv3D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size=2, data_format='channels_first')(conv8))
        merge9 = concatenate([conv1,up9],  axis = 1)
        conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge9)
        conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv9)
        conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv9)
        conv10 = Conv3D(1, 1, activation = 'sigmoid', data_format='channels_first')(conv9)

        model = Model(input = inputs, output = conv10)
 #       model = multi_gpu_model(model, gpus=2) 
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['mse', 'accuracy', dice_coefficient])
#        model.compile(optimizer = keras.optimizers.SGD(lr = 1e-12, decay=1e-6, nesterov=True, momentum=0.9), loss = 'mse', metrics = ['mse', dice_coefficient])
 #       model = multi_gpu_model(model, gpus=[0,1], cpu_merge=True, cpu_relocation=False)        
        return model
