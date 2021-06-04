# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:18:33 2020

@author: eych
"""

import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
import numpy as np

from keras.layers import Input
from keras.models import Model
#from keras import backend as K
from keras import layers

#######
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Lambda,Add,Reshape
from keras.layers import MaxPooling2D,Dropout


#######
from keras.layers import AlphaDropout
from keras.layers import concatenate
from keras.layers import Conv2DTranspose
from keras.layers import ConvLSTM2D
##
from keras.layers import AtrousConvolution2D


import os
import sys

print(sys.executable)


from convModule import *



def DeepForestM1(input_size,nb_labels):

    img_input = Input(shape=input_size)
    N = input_size[0]
    bn_axis = 3

    x = conv_block(img_input, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = Conv2D(64, 7, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')


    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    resn_x16 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') #(None, 16, 16, 256)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16_up)
    drop3 = Dropout(0.5)(resn_x16_up) # (None, 64, 64, 256)

    x = conv_block(resn_x16, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    resn_x8 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') #(None, 8,8, 256)

    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x8)
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6)
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6)
    drop4 = Dropout(0.5)(up6) #(None, 8, 8, 1024)
    #up6 = BatchNormalization(axis=3)(drop4)
    #up6 = Activation('relu')(up6) #(None, 64, 64, 256)
    
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop4) #(None, 64, 64, 256)
    
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    conv9 = Conv2D(nb_labels, 1, activation = 'softmax')(conv6)

    inputs=img_input
    model = Model(inputs, conv9)
    return model 




def DeepForestM2(input_size,nb_labels):

    img_input = Input(shape=input_size)
    N = input_size[0]

    '''if K.image_data_format() == 'channels_first':
    channel_axis = 1
    input_shape = (nb_rows, nb_cols, channels)
    if K.image_data_format() == 'channels_last':
    channel_axis = 3'''
    bn_axis = 3

    x = conv_block(img_input, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = Conv2D(64, 7, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    resn_x32 = identity_block(x, 3, [128,128,512], stage=3, block='d')
    resn_x32_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x32)
    drop2 = Dropout(0.5)(resn_x32_up) # (None, 64, 64, 256)


    x = conv_block(resn_x32, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    resn_x16 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') #(None, 16, 16, 256)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16_up)
    drop3 = Dropout(0.5)(resn_x16_up) # (None, 64, 64, 256)

    
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop2)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3) #(None, 64, 64, 256)
    
    merge1  = concatenate([x1,x2], axis = 1) #(None,1, 64, 64, 256)
    merge1 = ConvLSTM2D(filters = 128, kernel_size=(3,3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge1)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    drop= Dropout(0.5)(up7)
   

    x = conv_block(resn_x16, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    resn_x8 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') #(None, 8,8, 2048)
    
    
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x8)
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6)
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6)
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6) #(None, 128, 128, 128
    drop4 = Dropout(0.5)(up6) #(None, 128, 128, 128)

    
    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(drop)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(drop4) ##(None,1, 128, 128, 128)
    
    merge2  = concatenate([x1,x2], axis = 1) 
    merge2 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge2)
            
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    conv9 = Conv2D(nb_labels, 1, activation = 'softmax')(conv7)

    inputs=img_input
    model = Model(input = inputs, output = conv9)
    return model 



def Deeplab_ResNet50(input_shape,nb_labels):
   
    img_input = Input(shape=input_shape)
    
    bn_axis = 3

    x = conv_block(img_input, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = Conv2D(64, 7, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')


    x = conv_block_atrous(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='f') #/16

    #x1 = AveragePooling2D((2, 2), name='avg_pool')(x1)
#-----------End Resnet50
    
    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = conv_block_atrous(x, 3,[512, 512, 2048], stage=5, block='a',atrous_rate=(6, 6))
    b1 = identity_block_atrous(b1, 3, [512, 512, 2048], stage=5, block='b',atrous_rate=(6, 6))
    b1 = identity_block_atrous(b1, 3, [512, 512, 2048], stage=5, block='c',atrous_rate=(6, 6))
    b1 = Dropout(0.5)(b1)

    # hole = 12
    b2 = conv_block_atrous(x, 3, [512, 512, 2048], stage=6, block='a',atrous_rate=(12, 12))
    b2 = identity_block_atrous(b2, 3, [512, 512, 2048], stage=6, block='b',atrous_rate=(12, 12))
    b2 = identity_block_atrous(b2, 3, [512, 512, 2048], stage=6, block='c',atrous_rate=(12, 12))
    b2 = Dropout(0.5)(b2)

    # hole = 18
    b3 = conv_block_atrous(x, 3, [512, 512, 2048], stage=7, block='a',atrous_rate=(18, 18))
    b3 = identity_block_atrous(b3, 3, [512, 512, 2048], stage=7, block='b',atrous_rate=(18, 18))
    b3 = identity_block_atrous(b3, 3, [512, 512, 2048], stage=7, block='c',atrous_rate=(18, 18))
    b3 = Dropout(0.5)(b3)


    x1 = Conv2D(nb_labels, (1, 1))(x)
    merge  = concatenate([x1,b1,b2,b3], axis = 3) 
    m =  Conv2D(nb_labels, (1, 1))(merge)

    def resize_bilinear(images):
        return tf.compat.v1.image.resize_bilinear(images, [nb_rows, nb_cols])
    
    m = Lambda(resize_bilinear)(m)
    # Add softmax layer to get probabilities as output. We need to reshape
    # and then un-reshape because Keras expects input to softmax to
    # be 2D.
    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)
    
    inputs = img_input
    model = Model(input = inputs, output = x,name='Deeplab_ResNet50')
    
    return model



#ASPP
def Atrous_DeepForestM2(input_size,nb_labels):

    img_input = Input(shape=input_size)
    N = input_size[0]

    '''if K.image_data_format() == 'channels_first':
    channel_axis = 1
    input_shape = (nb_rows, nb_cols, channels)
    if K.image_data_format() == 'channels_last':
    channel_axis = 3'''
    bn_axis = 3

    x = conv_block(img_input, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = Conv2D(64, 7, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    resn_x32 = identity_block(x, 3, [128,128,512], stage=3, block='d')
    resn_x32_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x32)
    drop2 = Dropout(0.5)(resn_x32_up) # (None, 64, 64, 256)


    x = conv_block(resn_x32, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    resn_x16 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') #(None, 16, 16, 256)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16_up)
    drop3 = Dropout(0.5)(resn_x16_up) # (None, 64, 64, 256)

    
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop2)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3) #(None, 64, 64, 256)
    
    merge1  = concatenate([x1,x2], axis = 1) #(None,1, 64, 64, 256)
    merge1 = ConvLSTM2D(filters = 128, kernel_size=(3,3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge1)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    drop= Dropout(0.5)(up7)
   

    x = conv_block(resn_x16, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    resn_x8 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') #(None, 8,8, 2048)
    
    
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x8)
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6)
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6)
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up6) #(None, 128, 128, 128
    drop4 = Dropout(0.5)(up6) #(None, 128, 128, 128)

    
    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(drop)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(drop4) ##(None,1, 128, 128, 128)
    
    merge2  = concatenate([x1,x2], axis = 1) 
    merge2 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge2)
            
    conv7 = Conv2D(nb_labels, (1, 1))(merge2)
    kernel_size=3
    atour6=AtrousConvolution2D(128, kernel_size, kernel_size, atrous_rate=(6,6),
                            border_mode='same')(merge2)
    atour12=AtrousConvolution2D(128, kernel_size, kernel_size, atrous_rate=(12,12),
                            border_mode='same')(merge2)
    atour18=AtrousConvolution2D(128, kernel_size, kernel_size, atrous_rate=(18,18),
                            border_mode='same')(merge2)
    
    merge  = concatenate([conv7,atour6,atour12,atour18], axis = 3) 
    m =  Conv2D(nb_labels, (1, 1))(merge)

    def resize_bilinear(images):
        return tf.compat.v1.image.resize_bilinear(images, [nb_rows, nb_cols])
    
    m = Lambda(resize_bilinear)(m)
    # Add softmax layer to get probabilities as output. We need to reshape
    # and then un-reshape because Keras expects input to softmax to
    # be 2D.
    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    inputs=img_input
    model = Model(input = inputs, output = x)
    return model 



def Unet(input_shape,nb_labels):
    
    '''if K.image_data_format() == 'channels_first':
    channel_axis = 1
    input_shape = (nb_rows, nb_cols, channels)
    if K.image_data_format() == 'channels_last':
    channel_axis = 3'''
    inp = Input(input_shape)

    filter_size = (6, 6)
    blocks = [8,32,32,64,64,128]
    blocks2 = blocks
    activation = 'relu'
    filter_initializer = 'lecun_normal'
    channel_axis=3
    #encoder
    encoder = inp
    encoder_list = []
    for block_id , n_block in enumerate(blocks):
        with K.name_scope('Encoder_block{0}'.format(block_id)):
            encoder = Conv2D(filters = n_block, kernel_size = filter_size, activation = None, padding = 'same', kernel_initializer = filter_initializer) (encoder)
            encoder = BatchNormalization(axis=channel_axis, momentum=0.9) (encoder)
            encoder = Activation(activation) (encoder)
            encoder = AlphaDropout(0, 1*block_id) (encoder)
            encoder = Conv2D(filters = n_block, kernel_size = filter_size, dilation_rate = (2, 2), activation = None, padding='same', kernel_initializer = filter_initializer) (encoder)
            encoder = BatchNormalization(axis=channel_axis, momentum=0.9) (encoder)
            encoder = Activation(activation) (encoder)
            encoder_list.append(encoder)
            
            #maxpooling between every 2 blocks
            if block_id < len(blocks) - 1:
                encoder = MaxPooling2D(pool_size = (2, 2)) (encoder)
                
    #decoder
    decoder = encoder
    blocks = blocks[1:]
    for block_id, n_block in enumerate(blocks):
        with K.name_scope('Decoder_block_{0}'.format(block_id)):
            block_id_inv = len(blocks) - block_id
            decoder = concatenate([decoder, encoder_list[block_id_inv]], axis = channel_axis)
            decoder = AlphaDropout(0, 1*block_id) (decoder)
            decoder = Conv2D(filters = n_block, kernel_size = filter_size, activation = None, padding = 'same', kernel_initializer = filter_initializer) (decoder)
    #         decoder = BatchNormalization(axis=channel_axis, momentum=0.9) (decoder)
            decoder = Activation(activation) (decoder)
            decoder = Conv2DTranspose(filters = n_block, kernel_size = filter_size, kernel_initializer = filter_initializer, padding = 'same', strides=(2, 2)) (decoder)
    
    outp = Conv2DTranspose(filters=nb_labels, kernel_size = filter_size, activation = 'sigmoid', padding = 'same', kernel_initializer = keras.initializers.glorot_normal(seed=1337)) (decoder)
    model = Model(inputs = inp, outputs = outp)
    return model
