"""
Created on Thu Nov 26 15:12:46 2020

@author: eych
"""

import os
import sys

from sys import platform

from Models_1B import *
from Build_fit import *

from Models_2B import *
from Buid_2B import *


from LoadData import LoadData1B
from LoadData import LoadData2B

parent_path= sys.path.append(os.path.dirname(__file__))
print('Executable env: '+ str(sys.executable))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = "D:/Tiles/Final_datasets/Sentinel_2_tiles_test/*/"
    print(data_path)
    print(platform)

    ###################Experiments with 1Branch-Models###############
    ###Data Loading

    ld=LoadData1B()
    ld.load_data(data_path)

    #rasterio
    training_labels=ld.labels_forTr(10)
    tr_set = ld.training_set10_Tr(10)

    ###Arcpy
    #training_labels=ld.labels_forTr_Arcpy(10)
    #tr_set=ld.training_set_arcpy(10)

    print(training_labels.shape)
    print(tr_set.shape)

    ####Parameters def
    nb_labels = training_labels.shape[3]
    print(nb_labels)
    # The dimensions of the input images
    nb_rows = training_labels.shape[1]
    print(nb_rows)
    nb_cols = training_labels.shape[2]
    print(nb_cols)

    ###class Weights definition for the weighted loss function
    ws = ld.weights_prep(training_labels, nb_rows, nb_cols)
    print(ws)


    training_images_nor = ld.norm(tr_set)
    print(len(training_images_nor))
    del tr_set

    training_images_nor_aug, training_labels_aug = ld.rot(training_images_nor, training_labels, 0, 5)
    del training_labels
    training_images_nor_aug, training_labels_aug = ld.flip(training_images_nor_aug, training_labels_aug, 5, 10)

    ####Parameters def
    channels = training_images_nor_aug.shape[3]
    print(channels)

    input_shape = (nb_rows, nb_cols, channels)


    #### Model building + Training
    model=DeepForestM2(input_shape, nb_labels)
    model.summary()
    name='DeepForestM2' 
    loss='categorical_crossentropy'
    m=Buid_1B(nb_labels,model)
    print(m.nb_labels)
    m.build()
    #m.train_model(training_images_nor_aug,training_labels_aug,ws,name,loss,epochs=1)



    ###################Experiments with 2Branch-Models###############
    '''ld=LoadData2B()


    ld.load_data(data_path)
    training_labels=ld.labels_forTr(10)
    #training_labels=ld.labels_forTr_Arcpy(10)
    print(training_labels.shape)

    nb_labels = training_labels.shape[3]
    print(nb_labels)
    # The dimensions of the input images
    nb_rows = training_labels.shape[1]
    print(nb_rows)
    nb_cols = training_labels.shape[2]
    print(nb_cols)

    training_images_s1=ld.training_s1(10)
    training_images_s2=ld.training_s2(10)

    timestamps = training_images_s1.shape[1]
    print(timestamps)

    channels1= training_images_s1.shape[4]
    print(channels1)
    channels2= training_images_s2.shape[3]
    print(channels2)

    input_shape1=(timestamps,nb_rows,nb_cols, channels1)
    print(input_shape1)
    input_shape2= (nb_rows,nb_cols,channels2)
    print(input_shape2)

    ws=ld.weights_prep(training_labels,nb_rows,nb_cols)
    print(ws)


    training_images_s1_nor_aug,training_images_s2_nor_aug,training_labels_aug=ld.rot(training_images_s1,training_images_s2,training_labels,0,5)
    del training_labels
    training_images_s1_nor_aug,training_images_s2_nor_aug,training_labels_aug=ld.flip(training_images_s1,training_images_s2,training_labels_aug,5,10)

    # Model building

    model_glob, model1, model2 = two_branches(input_shape1, input_shape2, nb_labels)
    model_glob.summary()
    name = 'two_branch M'
    loss = 'categorical_crossentropy'
    m = Buid_2B(nb_labels, model_glob)
    print(m.nb_labels)
    m.build()'''
    #m.train_model(training_images_s1, training_images_s2, training_labels, weights=weights, valstart=15, name=name,loss=loss, epochs=1, batch_size=2)

    print('End Main')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/