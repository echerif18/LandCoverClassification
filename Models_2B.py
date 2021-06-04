# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:38:31 2020

@author: eych
"""


import tensorflow as tf
import keras
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


from convModule import *



#Combination with Concatination: three outputs
def two_branches_unet(input_shape1, input_shape2,nb_labels):  
    
    inputA = Input(input_shape1, name='input1')
    inputB = Input(input_shape2, name='input2')
    
    N=input_shape2[0]

    #M1
    conv1 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(inputA)
    
    convf = Conv2D(nb_labels, 1, activation = 'softmax',name="M1_output")(conv1)
    
    model1 = Model(inputs=inputA, outputs=[convf])
    
    
    #M2
    channel_axis = 3
    filter_size = (6, 6)
    blocks = [8,32,32,64,64,128]
    blocks2 = blocks
    activation = 'relu'
    filter_initializer = 'lecun_normal'

    #encoder
    encoder = inputB
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

    outp = Conv2DTranspose(filters=nb_labels, kernel_size = filter_size, padding = 'same', kernel_initializer = keras.initializers.glorot_normal(seed=1337)) (decoder)
    #model = Model(inputs = inp, outputs = outp)
    conv9 = Conv2D(nb_labels, 1, activation = 'softmax',name="Unet_output")(outp) # I can eliminate softmax layer from both models

    model2 = Model(inputs = inputB, outputs = conv9)
    
    #Combination
    #combined = Add()([model1.output, model2.output])
    combined = concatenate([outp, conv1], axis = 3)
    #combined = Add()([conv6, conv2])
    conv10 = Conv2D(nb_labels, 1, activation = 'softmax',name="Comb_output")(combined)
    
    
    
    model = Model(inputs=[model1.input, model2.input], outputs=[model1.output, model2.output,conv10])
    #model = Model(input=[model1.input, model2.input], output=[conv10])
    return model,model1,model2


#Combination with Concatination: three outputs
def two_branches(input_shape1, input_shape2,nb_labels):  
    
    inputA = Input(input_shape1, name='input1')
    inputB = Input(input_shape2, name='input2')
    
    N=input_shape2[0]

    #M1
    conv1 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(inputA) 

    #pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    #conv2 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(pool1)
    #conv2=Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv2)
    
    convf = Conv2D(nb_labels, 1, activation = 'softmax',name="M1_output")(conv1)
    
    model1 = Model(inputs= inputA, outputs=[convf])
    
    #M2
    bn_axis = 3
    x = conv_block(inputB, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
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
    x32 = identity_block(x, 3, [128,128,512], stage=3, block='d') 


    x = conv_block(x32, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    resn_x16 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') #(None, 16, 16, 1024)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16)
    resn_x16_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x16_up)
    drop3 = Dropout(0.5)(resn_x16_up) #(None, 64, 64, 256)
    
    x = conv_block(drop3, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    resn_x8 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') #(None, 32, 32, 2048)
    resn_x8_up = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(resn_x8)
    drop4 = Dropout(0.5)(resn_x8_up)
    drop4 = BatchNormalization(axis=3)(drop4)
    drop4 = Activation('relu')(drop4) #(None, 64, 64, 256)
    
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop4)
    
    merge6  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 256)
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    conv9 = Conv2D(nb_labels, 1, activation = 'softmax',name="M2_output")(conv6) # I can eliminate softmax layer from both models

    model2 = Model(inputs = inputB, outputs = conv9)
    
    #Combination
    #combined = Add()([model1.output, model2.output])
    combined = concatenate([conv6, conv1], axis = 3)
    #combined = Add()([conv6, conv2])
    conv10 = Conv2D(nb_labels, 1, activation = 'softmax',name="Comb_output")(combined)
    
    
    
    model = Model(inputs=[model1.input, model2.input], outputs=[model1.output, model2.output,conv10])
    #model = Model(input=[model1.input, model2.input], output=[conv10])
    return model,model1,model2










##########################################
#Combination with Concatination: three outputs
################two_senti2 models
def two_branches_M2_unet(input_shape1, input_shape2,nb_labels):  
    
    inputA = Input(input_shape1, name='input1')
    inputB = Input(input_shape2, name='input2')
    
    N=input_shape2[0]

    #M1
    bn_axis = 3

    x = conv_block(inputA, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
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

    #inputs=img_input
    #model = Model(input = inputs, output = conv9)
    convf = Conv2D(nb_labels, 1, activation = 'softmax',name="M1_output")(conv7)
    
    model1 = Model(inputs=inputA, outputs=[convf])
    
    #M2
    channel_axis = 3
    filter_size = (6, 6)
    blocks = [8,32,32,64,64,128]
    blocks2 = blocks
    activation = 'relu'
    filter_initializer = 'lecun_normal'

    #encoder
    encoder = inputB
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

    outp = Conv2DTranspose(filters=nb_labels, kernel_size = filter_size, padding = 'same', kernel_initializer = keras.initializers.glorot_normal(seed=1337)) (decoder)
    #model = Model(inputs = inp, outputs = outp)
    conv9 = Conv2D(nb_labels, 1, activation = 'softmax',name="M2_output")(outp) # I can eliminate softmax layer from both models

    model2 = Model(inputs = inputB, outputs = conv9)
    
    #Combination
    #combined = Add()([model1.output, model2.output])
    combined = concatenate([outp, conv7], axis = 3)
    #combined = Add()([conv6, conv2])
    conv10 = Conv2D(nb_labels, 1, activation = 'softmax',name="Comb_output")(combined)
    
    
    
    model = Model(inputs=[model1.input, model2.input], outputs=[model1.output, model2.output,conv10])
    #model = Model(input=[model1.input, model2.input], output=[conv10])
    return model,model1,model2
