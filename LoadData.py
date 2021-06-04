# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:04:28 2020

@author: eych
"""


import os
import sys

from glob import glob
import numpy as np
import rasterio
import arcpy

import random
from random import randint
random.seed(1337)
np.random.seed(1337)

import keras
from keras.utils import to_categorical
from sys import platform



class LoadData:
    def __init__(self):
        self.data_tiles_list = []
        self.label_tiles_list = []
        self.weight=[]
        self.nb_labels=18 #known in prior: 18  in our case
    
    def load_data(self,data_path):
        #######look again for linux and win versions
        data = glob(data_path)
            
        #the order of the files is different in windows and linux
        if platform == "linux" or platform == "linux2":
            self.data_tiles_list.append(glob(data[1]+ "/*.tif"))
            self.label_tiles_list.append(glob(data[0]+ "/*.tif"))      
        elif platform == "win32":  
            self.data_tiles_list.append(glob(data[0]+ "/*.tif"))
            self.label_tiles_list.append(glob(data[1]+ "/*.tif"))

        
        
        self.data_tiles_list = [y for x in self.data_tiles_list for y in x]
        self.label_tiles_list = [y for x in self.label_tiles_list for y in x]
        #return data_tiles_list,label_tiles_list



    ##############################Labels########################
    def labels_forTr(self,size):
        tr_labels=[]
        for i in range(size):
            print(str(i) + "/" + str(size))
            weightspers=[]
            data_in = rasterio.open(self.label_tiles_list[i]) #opens the .tif/Raster in an array    
            tr_labels.append(data_in.read(1))
            for j in range(self.nb_labels):
                weightspers.append(np.sum(data_in.read(1)==j))
            self.weight.append(weightspers)
                
        training_labels = np.asarray(tr_labels) # converts list training_images into an array
        del tr_labels
        training_labels = np.expand_dims(training_labels, axis=3)
        
        print(training_labels.shape)
        training_labels=to_categorical(training_labels)
        return training_labels
    
    
    def labels_forEval(self,size):
        tr_labels=[]
        for i in range(size):
            print(str(i) + "/" + str(size))
            weightspers=[]
            data_in = rasterio.open(self.label_tiles_list[i]) #opens the .tif/Raster in an array    
            tr_labels.append(data_in.read(1))
            for j in range(self.nb_labels):
                weightspers.append(np.sum(data_in.read(1)==j))
            self.weight.append(weightspers)
                
        training_labels = np.asarray(tr_labels) # converts list training_images into an array
        del tr_labels
        training_labels = np.expand_dims(training_labels, axis=3)
        return training_labels
    
    ######to test with arcpy no weights calcul
    def labels_forTr_Arcpy(self,size):
        tr_labels=[]
        for i in range(size):
            print(str(i) + "/" + str(size))
            data_in  = arcpy.Raster(self.label_tiles_list[i]) 
            x = arcpy.RasterToNumPyArray(data_in)
            tr_labels.append(x)
                
        training_labels = np.asarray(tr_labels) # converts list training_images into an array
        del tr_labels
        training_labels = np.expand_dims(training_labels, axis=3)
    
        print(training_labels.shape)
        training_labels=to_categorical(training_labels)
        return training_labels
    
    
    

    
    
    
    
class LoadData1B(LoadData):
    
    def __init__(self):
        super().__init__() #inherit all prop and methods
        #LoadData.__init__(self)
    
    def weights_prep(self,training_labels,nb_rows,nb_cols):
        nb_tiles=len(training_labels)
        weights=np.sum(self.weight,axis=0)/(nb_tiles*nb_rows*nb_cols) #percentage of classes
        weights=100-np.array(weights)*100
        return weights
    
    
##############################Training Set

##############################With Rastrio
    def training_set34_Tr(self,size):
        tr_images = []
        for i in range(size):
            print(str(i) + "/" + str(size))
            data_in = rasterio.open(self.data_tiles_list[i]) #opens the .tif/Raster in an array
            
            t = np.expand_dims(data_in.read(1), axis=2) #expands the array by 1 dimension
            for j in range(2,35): #expands the array by j dimensions and adds them together (concatenate)
                t = np.concatenate((t, np.expand_dims(data_in.read(j), axis=2)), axis=2)
            #print(t.shape)
            tr_images.append(t)
                
        training_images = np.asarray(tr_images) # converts list training_images into an array
        #print(training_images.shape)
        del tr_images #Memory optimization
        return training_images
    
    
    
    def training_set10_Tr(self,size):
        tr_images = []
        for i in range(size):
            print(str(i) + "/" + str(size))
            data_in = rasterio.open(self.data_tiles_list[i]) #opens the .tif/Raster in an array
            
            t = np.expand_dims(data_in.read(1), axis=2) #expands the array by 1 dimension
            for j in range(2,11): #expands the array by j dimensions and adds them together (concatenate)
                t = np.concatenate((t, np.expand_dims(data_in.read(j), axis=2)), axis=2)
            #print(t.shape)
            tr_images.append(t)
            
        training_images = np.asarray(tr_images) # converts list training_images into an array
        #print(training_images.shape)
        del tr_images #Memory optimization
        return training_images
    
    
    def training_set24_Tr(self,size):
        tr_images = []
        for i in range(size):
            print(str(i) + "/" + str(size))
            data_in = rasterio.open(self.data_tiles_list[i]) #opens the .tif/Raster in an array
            
            t = np.expand_dims(data_in.read(25), axis=2) #expands the array by 1 dimension
            for j in range(26,35): #expands the array by j dimensions and adds them together (concatenate)
                t = np.concatenate((t, np.expand_dims(data_in.read(j), axis=2)), axis=2)
            #print(t.shape)
            tr_images.append(t)
              
        training_images = np.asarray(tr_images) # converts list training_images into an array
        #print(training_images.shape)
        del tr_images #Memory optimization
        return training_images
    
    
##############################With Arcpy
    
    
    def training_set_arcpy(self,size):
        tr_images = []
        for i in range(size):
            print(str(i) + "/" + str(size))
            data_in = arcpy.Raster(self.data_tiles_list[i]) #opens the .tif/Raster in an array
            x = arcpy.RasterToNumPyArray(data_in)
            x = np.moveaxis(x, 0, 2)
            #print(x.shape)
            tr_images.append(x)
        training_images = np.asarray(tr_images)
        del tr_images
            
        return training_images
    
    
    
#################Normalization
    def norm(self,training_images):
        channels=training_images.shape[3]
        mean=[]
        std=[]
        for i in range(channels):
            mean.append(np.mean(training_images[:,:,:,i]))
            std.append(np.std(training_images[:,:,:,i]))
        training_images_nor=training_images-mean
        training_images_nor/=std
        return training_images_nor
    



####################Augmentation################
    
    def rot(self,training_images,training_labels,rotStart,rotEnd):
        rotation = [1]
        for i in rotation:
            training_images_aug = np.concatenate((training_images, np.rot90(training_images[rotStart:rotEnd], i, (1,2))), axis=0)
            training_labels_aug = np.concatenate((training_labels, np.rot90(training_labels[rotStart:rotEnd], i, (1,2))), axis=0)
        if(len(training_labels)==len(training_images)):
            print("After rotation:")
            print("Number of training labels/images: " + str(training_labels_aug.shape))
            return training_images_aug,training_labels_aug
        else:
            print('Rotation not successful')
            return training_images,training_labels

    def flip(self,training_images,training_labels,flipStart,flipEnd):
        training_images_aug = np.concatenate((training_images, np.flip(training_images[flipStart:flipEnd], 2)), axis=0)
        training_labels_aug = np.concatenate((training_labels, np.flip(training_labels[flipStart:flipEnd], 2)), axis=0)
        if(len(training_labels)==len(training_images)):  
            print("After horizontal flip:")
            print("Number of training labels/images: " + str(training_labels_aug.shape))
            return training_images_aug,training_labels_aug
        else:
            print('Flipiing not successful')
            return training_images,training_labels   
        

    
    
    
    
    
    
class LoadData2B(LoadData):
    
    def __init__(self):
        #LoadData.__init__(self) #inherit only the constructor
        super().__init__() #inherit all prop and methods
        
        
    def weights_prep(self,training_labels,nb_rows,nb_cols):
        nb_tiles=len(training_labels)
        w=np.sum(self.weight,axis=0)/(nb_tiles*nb_rows*nb_cols) #percentage of classes
        w=100-np.array(w)*100
        weights={"M1_output": w,
                 "M2_output": w,
                 "Comb_output": w}
        return weights
    
    
    def training_s1(self,size):
        tr_images_s1 = []
        
        
        for i in range(size):
            print(str(i) + "/" + str(size))
               
            data_in = rasterio.open(self.data_tiles_list[i]) #opens the .tif/Raster in an array    
            t = np.concatenate((np.expand_dims(data_in.read(1), axis=2), np.expand_dims(data_in.read(2), axis=2)), axis=2) #expands the array by 1 dimension
            t1 =np.expand_dims(t, axis=0)
            #print(t1.shape)
        
            for j in range(3,11,2): #expands the array by j dimensions and adds them together (concatenate)
                t = np.concatenate((np.expand_dims(data_in.read(j), axis=2), np.expand_dims(data_in.read(j+1), axis=2)), axis=2)
                t1 = np.concatenate((t1, np.expand_dims(t, axis=0)), axis=0) 
            tr_images_s1.append(t1)
            
        
        training_images_s1 = np.asarray(tr_images_s1) # converts list training_images into an array
        del tr_images_s1
        print(training_images_s1.shape)
        
        return training_images_s1



    def training_s2(self,size):
        tr_images_s2 = []
        
        for i in range(size):
            print(str(i) + "/" + str(size))
                    
            data_in = rasterio.open(self.data_tiles_list[i]) #opens the .tif/Raster in an array        
            t = np.expand_dims(data_in.read(1), axis=2) #expands the array by 1 dimension
        
            for j in range(2,11): #expands the array by j dimensions and adds them together (concatenate)
                t = np.concatenate((t, np.expand_dims(data_in.read(j), axis=2)), axis=2)
                
            tr_images_s2.append(t)
            
        
        training_images_s2 = np.asarray(tr_images_s2) # converts list training_images into an array
        del tr_images_s2
        
        print(training_images_s2.shape)
        return training_images_s2

    
####################Augmentation################
    def rot(self,training_images_s1,training_images_s2,training_labels,rotStart,rotEnd):
        rotation = [1]
        for i in rotation:
            training_images_s1_aug = np.concatenate((training_images_s1, np.rot90(training_images_s1[rotStart:rotEnd], i, (2,3))), axis=0)
            training_images_s2_aug = np.concatenate((training_images_s2, np.rot90(training_images_s2[rotStart:rotEnd], i, (1,2))), axis=0)
            training_labels_aug = np.concatenate((training_labels, np.rot90(training_labels[rotStart:rotEnd], i, (1,2))), axis=0)
        if(len(training_labels)==len(training_images_s1)) & (len(training_labels)==len(training_images_s2)):
            print("After rotation:")
            print("Number of training labels/images: " + str(training_labels_aug.shape))
            return training_images_s1_aug,training_images_s2_aug,training_labels_aug
        else:
            print('Rotation not successful')
            return training_images_s1,training_images_s2,training_labels
    
    def flip(self,training_images_s1,training_images_s2,training_labels,flipStart,flipEnd):
        training_images_s1_aug = np.concatenate((training_images_s1, np.flip(training_images_s1[flipStart:flipEnd], 2)), axis=0)
        training_images_s2_aug = np.concatenate((training_images_s2, np.flip(training_images_s2[flipStart:flipEnd], 2)), axis=0)
        training_labels_aug = np.concatenate((training_labels, np.flip(training_labels[flipStart:flipEnd], 2)), axis=0)
        if(len(training_labels)==len(training_images_s1)) & (len(training_labels)==len(training_images_s2)):  
            print("After horizontal flip:")
            print("Number of training labels/images: " + str(training_labels_aug.shape))
            return training_images_s1_aug,training_images_s2_aug,training_labels_aug
        else:
            print('Flipiing not successful')
            return training_images_s1,training_images_s2,training_labels    
    
    
    
