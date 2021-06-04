# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:23:20 2020

@author: eych
"""

import tensorflow as tf
import keras
from keras import backend as K

import datetime
import numpy as np
import os
import sys

print(sys.executable)



class Buid_1B:
    def __init__(self,nb_labels,model):
        self.nb_labels= nb_labels
        self.model= model


    def build(self,lr= 0.0001,loss='categorical_crossentropy'):
        def mean_iou(y_true, y_pred):
            prec = []
            for t in np.arange(0.5, 1.0, 1.0):
                y_pred_ = tf.to_int32(y_pred > t)
                score, up_opt = tf.metrics.mean_iou(labels=y_true,predictions = y_pred_, num_classes = self.nb_labels, weights = y_true) # Confusion matrix of [num_classes, num_classes]
                sess = tf.Session()
                sess.run(tf.local_variables_initializer())
                with tf.control_dependencies([up_opt]):
                    score = tf.identity(score)
                prec.append(score)
            return K.mean(K.stack(prec), axis=0)
        
        #lr = 0.0001
        optimizer = keras.optimizers.Adam(lr = lr)
        #loss='categorical_crossentropy'
        #loss='mae'
        metrics = [mean_iou,'accuracy']
            
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
    
    
    def train_model(self,training_images,training_labels,weights,name,loss,batch_size=10,epochs = 10):
        print('training start')
        logs_base_dir = "./logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(logs_base_dir, exist_ok=True)
            
        callbacks = [
        keras.callbacks.TensorBoard(logs_base_dir, histogram_freq=1),
         # Interrupt training if `val_loss` stops improving for over 1 epochs
         tf.keras.callbacks.EarlyStopping(patience=1, monitor='val_acc') ]
    
        
        with tf.Session() as sess:
        
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.model.fit(training_images, training_labels,class_weight=weights, verbose=1,callbacks =callbacks, validation_split = 0.2, batch_size = batch_size, epochs = epochs, initial_epoch = 0)
            self.model.save(name+"_Amazon_"+str(training_images.shape[3])+"bands_"+loss+"_"+str(str(training_images.shape[0]))+"test.h5")
            
    