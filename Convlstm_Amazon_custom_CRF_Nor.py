# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:43:28 2020

@author: eych
"""


'''
Copyright 2018 Esri
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.​
'''

import json
import os
import sys
import arcpy
import numpy as np


sys.path.append(os.path.dirname(__file__))
from attribute_table import attribute_table     #For .emd configuration
import prf_utils    #For GPU

from keras.models import load_model
import keras.backend as K
import tensorflow as tf 
from keras.metrics import mean_squared_error

########CRF########
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

MAX_ITER = 10

POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


class Convlstm:
	def initialize(self, model, model_as_file):
	
		# clean the background session
		K.clear_session()

		# load the emd file into a dictionary
		if model_as_file:
			with open(model, 'r') as f:
				self.json_info = json.load(f)
		else:
			self.json_info = json.loads(model)

		# get the path to the trained model
		model_path = self.json_info['ModelFile']

		# load the trained model
		self.model = load_model(model_path, custom_objects={'mean_iou': mean_squared_error,'tf': tf,'nb_rows':self.json_info['ImageWidth'] , 'nb_cols': self.json_info['ImageHeight']})

		# build a default background tensorflow computational graph
		self.graph = tf.get_default_graph()
        

		
	def getParameterInfo(self, required_parameters):
		return required_parameters

	def getConfiguration(self, **scalars):
        
		self.padding = int(scalars['padding'])
		
		#get the threshold parameter
		#self.threshold = float(scalars['threshold'].replace(",", "."))

		return {
			'extractBands': tuple(self.json_info['ExtractBands']),
			# padding should be 0
			'padding': int(scalars['padding']),
            'invalidateProperties': 2 | 4 | 8, 
			'tx': self.json_info['ImageWidth'] - 2 * self.padding,
			'ty': self.json_info['ImageHeight'] - 2 * self.padding
		}
            
            

class Convlstm_Amazon_custom_CRF_Nor(Convlstm):
	def __init__(self):
		self.name = 'Image Classifier'
		self.description = 'Image classification python raster function to inference a tensorflow ' \
						   'deep learning model'
		

	def initialize(self, **kwargs):
		if 'model' not in kwargs:
			return

		model = kwargs['model']
		model_as_file = True
        
		try:
			with open(model, 'r') as f:
				self.json_info = json.load(f)
		except FileNotFoundError:
			try:
				self.json_info = json.loads(model)
				model_as_file = False
			except json.decoder.JSONDecodeError:
				raise Exception("Invalid model argument")

		sys.path.append(os.path.dirname(__file__))
		framework = self.json_info['Framework']

		if 'device' in kwargs:
			device = kwargs['device']
			if device < -1:
				os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
				device = prf_utils.get_available_device()
			os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
		else:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
		Convlstm.initialize(self, model, model_as_file)			

        

	def getParameterInfo(self):
		required_parameters = [
			{
				'name': 'raster',
				'dataType': 'raster',
				'required': True,
				'displayName': 'Raster',
				'description': 'Input Raster'
			},
			{
				'name': 'model',
				'dataType': 'string',
				'required': True,
				'displayName': 'Input Model Definition (EMD) File',
				'description': 'Input model definition (EMD) JSON file'
			},
			{
				'name': 'device',
				'dataType': 'numeric',
				'required': False,
				'displayName': 'Device ID',
				'description': 'Device ID'
			},
		]

		if 'ModelPadding' not in self.json_info:
			required_parameters.append(
				{
					'name': 'padding',
					'dataType': 'numeric',
					'value': 0,
					'required': False,
					'displayName': 'Padding',
					'description': 'Padding'
				},
			)

		if 'BatchSize' not in self.json_info:
			required_parameters.append(
				{
					'name': 'batch_size',
					'dataType': 'numeric',
					'required': False,
					'value': 1,
					'displayName': 'Batch Size',
					'description': 'Batch Size'
				},
			)

		return required_parameters

	def getConfiguration(self, **scalars):
		configuration=Convlstm.getConfiguration(self, **scalars) 
		#configuration = self.child_image_classifier.getConfiguration(**scalars)

		if 'DataRange' in self.json_info:
			configuration['dataRange'] = tuple(self.json_info['DataRange'])
		configuration['inheritProperties'] = 2|4|8
		configuration['inputMask'] = False
		return configuration

	def updateRasterInfo(self, **kwargs):
		self.stat=kwargs['raster_info']['statistics']
       
		kwargs['output_info']['bandCount'] = 1
		kwargs['output_info']['pixelType'] = 'u1'
		class_info = self.json_info['Classes']
		attribute_table['features'] = []
		for i, c in enumerate(class_info):
			attribute_table['features'].append(
				{
					'attributes':{
						'OID':i+1,
						'Value':c['Value'],
						'Class':c['Name'],
						'Red':c['Color'][0],
						'Green':c['Color'][1],
						'Blue':c['Color'][2]
					}
				}
			)
		kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

		return kwargs

	def updatePixels(self, tlc, shape, props, **pixelBlocks):
		# set pixel values in invalid areas to 0
		#raster_mask = pixelBlocks['raster_mask']
		#raster_pixels = pixelBlocks['raster_pixels']
		#raster_pixels[np.where(raster_mask == 0)] = 0
		#pixelBlocks['raster_pixels'] = raster_pixels
        
		image = np.array(pixelBlocks['raster_pixels'])
		channel, h, w = image.shape
		mean=[]
		std=[]
		#std=[2.45,2.55,2.53,2.61,2.58,2.74,2.98,3.33,3.54,3.05,3.76,2.61,112.57,138.53,245.70,212.98,573.88,767.37,783.08,824.89,599.30,455.92]
		#mean=[-13.65,-15.33,-14.70,-14.99,-14.76,-16.00,-16.34,-16.94,-17.12,-15.86,-16.06,-15.06,322.01,594.65,428.87,996.99,2497.83,3018.05,2971.54,3248.39,2115.94,1085.83]
		for i in range(len(self.stat)):
			for x,y in self.stat[i].items():
				if x=='mean':
					mean.append(y)
				if x=='standardDeviation':
					std.append(y)       
         
		image_in=np.transpose(image,[1,2,0])-mean
		image_in/=std
        
		#image_in=np.transpose(image,[1,2,0])                
		image_in = np.expand_dims(image_in, axis=0)
		with self.graph.as_default():
			pred = self.model.predict(image_in, verbose = 0)
			results = pred.argmax(axis=-1)               
        
		#result=results[:,:,:]
		test=np.transpose(pred[0],[2,0,1])
		c= test.shape[0]
		U = utils.unary_from_softmax(test)
		U = np.ascontiguousarray(U)

		img = np.ascontiguousarray(np.expand_dims(results[0], axis=3))

		d = dcrf.DenseCRF2D(w, h, c)
		d.setUnaryEnergy(U)
		d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W,normalization=dcrf.NORMALIZE_SYMMETRIC,kernel=dcrf.DIAG_KERNEL)        
		pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
		d.addPairwiseEnergy(pairwise_energy, compat=Bi_W)
		Q = d.inference(MAX_ITER)

		Q = np.array(Q).reshape((c, h, w))
		result=Q.argmax(axis=0)
		result = np.expand_dims(result, axis=0)
		#result = np.transpose(Q,[1,2,0])
		pixelBlocks['output_pixels']=result.astype(props['pixelType'], copy=False)

		return pixelBlocks