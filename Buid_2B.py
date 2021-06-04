# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:12:46 2020

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

class Buid_2B:
    def __init__(self,nb_labels,model):
        self.nb_labels= nb_labels
        self.model= model		     
    
    
    def build(self,lr= 0.0001,loss='categorical_crossentropy',wl=[0.3,0.7,1.0]):

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
        
        optimizer = keras.optimizers.Adam(lr = lr)
        losses = { "M1_output": loss,
                  "M2_output": loss,
                  "Comb_output": loss}
        
        lossWeights = {"M1_output": wl[0], "M2_output":wl[1],"Comb_output":wl[2]}
        #loss='categorical_crossentropy'
        #loss='mae'
        metrics = [mean_iou,'accuracy']
            
        self.model.compile(loss = losses, optimizer = optimizer, metrics = metrics, loss_weights=lossWeights)
        
        
        
    def train_model(self,training_images_s1,training_images_s2,training_labels,weights,name,loss,valstart,batch_size=10,epochs = 10):
        print('Here training')
        logs_base_dir = "./logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(logs_base_dir, exist_ok=True)
            
        callbacks = [
        keras.callbacks.TensorBoard(logs_base_dir, histogram_freq=1),
         # Interrupt training if `val_loss` stops improving for over 1 epochs
         tf.keras.callbacks.EarlyStopping(patience=1, monitor='val_Comb_output_acc') ]
        
        
        with tf.Session() as sess:
        
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.model.fit([training_images_s1[:valstart],training_images_s2[:valstart]], [training_labels[:valstart],training_labels[:valstart],training_labels[:valstart]], validation_data=([training_images_s1[valstart:],training_images_s2[valstart:]], [training_labels[valstart:],training_labels[valstart:],training_labels[valstart:]]),class_weight=weights, verbose=1,callbacks =callbacks, batch_size = batch_size, epochs = epochs, initial_epoch = 0)
            self.model.save(name+"_Amazon_"+loss+"_"+str(training_images_s2.shape[0])+"test.h5")
