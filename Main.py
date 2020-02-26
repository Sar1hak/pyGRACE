
from __future__ import print_function
import skimage.io as io
import skimage.transform as trans
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, concatenate, UpSampling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import Adam

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as keras
import gdal
import tensorflow as tf

import numpy as np
import os
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import rasterio.merge
import tifffile

# import other files in use
from Norm_Convol_func import normalize_convolve
from Unet_func import unet
from merge_func import merge
from tif_stack import tif_stack
from tif_avg import average_value

#######MAIN#######
	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")


# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
 

# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	)

path = "H:/Project/"
folder=["NOAH TWSA","Precipitation","Temperature"]
#folder=['Rainfall']
os.chdir("H:/Project/")

	
if not os.path.exists(path):
	print("ERROR {} not found!".format(path))
	exit()
	
##### Read Tiff file and convert to GrayScale #####
	
for y in folder:
	for i in range(200001,201813):
# load the input image and convert it to grayscale
#image = cv2.imread(args["image"])
		file_path= "H:/Project/"+ y +"/"+str(i)+".tif"
		image = tifffile.imread(file_path)
		image = cv2.resize(image, (256,256))
		#image = cv2.imread(file_path)
		'''displaying tiff files depends you, 228 file open up :('''
		#cv2.imshow('Original', image)
		#cv2.waitKey()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#cv2.imshow('Grayscale', gray)
		#cv2.waitKey(0)

##### Stacking #####
		
		#if i>=20003:
		#zimage = tif_stack(tifffile.imread("H:/Project/" + y + "/"+str(i-2)+".tif"),tifffile.imread("H:/summer project/Rainfall/"+str(i-1)+".tif"),image)


##### Normalization and Convolution #####
		
# loop over the kernels
		for (kernelName, kernel) in kernelBank:
# apply the kernel to the grayscale image using both
# our custom `convole` function and OpenCV's `filter2D` function
			print("[INFO] applying {} kernel".format(kernelName))
			convoleOutput = normalize_convolve(gray, kernel,i)# if stackking function working, gray replace by zimage
			opencvOutput = cv2.filter2D(gray, -1, kernel) # if stackking function working, gray replace by zimage
 
# show the output images
			'''cv2.imshow("original", gray)
			cv2.imshow("{} - convole".format(kernelName), convoleOutput)
			cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
			cv2.waitKey(0)'''
			dim = (256,256)
			convoleOutput = cv2.resize(convoleOutput, dim, interpolation = cv2.INTER_AREA)
			if kernelName == 'sharpen':
				cv2.imwrite( "H:/Project/save data/"+ y+"/", convoleOutput );
		a = str(i)
		if a.endwith('12'):
			i = i + 88
					
# input image dimensions
img_rows, img_cols = 256, 256

img_channels = 3
nb_classes = 10

##### Merging Raster #####
bounds=None
res=None
nodata=None
precision=7
sets=['Training_set','Test_set']
count = 0

for x in range(200003,201812):#1,1812
	#file1=rasterio.open("H:/Project/Rainfall/"+str(x)+".tif")
	#file2=rasterio.open("H:/Project/Rainfall/"+str(x-1)+".tif")
	#file3=rasterio.open("H:/Project/Rainfall/"+str(x-2)+".tif")
	count = count + 1
	file1=rasterio.open("H:/Project/save data/NOAH TWSA/"+str(x)+".tif")
	file2=rasterio.open("H:/Project/save data/Precipitation/"+str(x)+".tif")
	file3=rasterio.open("H:/Project/save data/Temperature"+str(x)+".tif")
	dest, output_transform = merge([file1,file2,file3],bounds,res,nodata,precision)
	if count < 172:
		raster_name = 'H:/Project/'+sets[0]+'mergedraster'+str((x-200000))+'.tif'
	else:
		raster_name = 'H:/Project/'+sets[1]+'mergedraster'+str((x-200000))+'.tif'
	with rasterio.open("H:/save data/NOAH TWSA"+str(x)+".tif") as src:
		out_meta = src.meta.copy()    
	out_meta.update({"driver": "GTiff",
                 "height": dest.shape[1],
                 "width": dest.shape[2],
                 "transform": output_transform})
	with rasterio.open(raster_name, "w", **out_meta) as dest1:
		dest1.write(dest)
	a=str(x)
	if a.endwith('12'):
		x = x + 101 #99

##### Unet Training and Testing #####

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height))

test_datagen = ImageDataGenerator(rescale=1./255)  

train_generator = train_datagen.flow_from_directory(
        'H:/Project/trainig_set',
        target_size=(256, 256),
        batch_size=32,
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        'H:/Project/test_set',
        target_size=(256, 256),
        batch_size=32,
        shuffle=False)

model = unet()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit_generator( train_generator, steps_per_epoch=500, epochs=50, validation_data=test_generator, validation_steps=250)

##### Prediction #####
save_path="H:/Project/Output/"
results = model.predict_generator(test_generator,30,verbose=1)
q=1
for item in results:
	cv2.imwrite(os.path.join(save_path , 'output'+q+'.tif'), results)
	q=q+1

##### Plotting maps #####

#plotting loss graph
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

m_avg,yr_avg = average_value(folder)
outt=["Output"]
om_avg,oyr_avg = average_value(outt)

color1=("red","blue","green")
color2=("cyan","yellow","maroon")

# Plot Intial monthly graph
month=[1,2,3,4,5,6,7,8,9,10,11,12]
for i in range(0,3):
	# Create plot
	fig = plt.figure()
	ax = fig.add_subplot( 3, 1, (i+1))
	c=0
	for j in range(0,19):
			z1=m_avg[i][c:c+12]
			data2 = (z1)
	
			for data, color in zip(data2, color1):
				#p, y = data
				p=data
					#ax.scatter(p, y, alpha=0.8, c=color)
				plt.plot(month,p,c=color[i])	
				plt.title('Intial Monthly Avg. Rainfall vs Year')
				plt.xlabel('Month')
				plt.ylabel('Rainfall')
			c=c+11
	plt.show()
# Plot Output monthly graph
for i in range(0,3):
	# Create plot
	fig = plt.figure()
	ax = fig.add_subplot( 3, 1, (i+1))
	c=0
	for j in range(0,19):
			z1=om_avg[i][c:c+12]
			data2 = (z1)
	
			for data, color in zip(data2, color2):
				#p, y = data
				p=data
					#ax.scatter(p, y, alpha=0.8, c=color)
				plt.plot(month,p,c=color[i])	
				plt.title('Observed Monthly Avg. Rainfall vs Year')
				plt.xlabel('Month')
				plt.ylabel('Rainfall')
			c=c+11
	plt.show()

# Plot Intial Average Annual Graph

year=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

for i in range(0,3):
	# Create plot
	fig = plt.figure()
	ax = fig.add_subplot( 3, 1, (i+1))
	x1=yr_avg[i][:]#NOAH Temp Ppt
	plt.plot(year,x1,c=color1[i])	
	plt.title('Intial Yearly Avg. Rainfall vs Year')
	plt.xlabel('year')
	plt.ylabel('Rainfall')
plt.show()

# Plot Output Average Annual Graph

for i in range(0,3):
	# Create plot
	fig = plt.figure()
	ax = fig.add_subplot( 3, 1, (i+1))
	x1=oyr_avg[i][:]#NOAH Temp Ppt
	plt.plot(year,x1,c=color2[i])	
	plt.title('Observed Yearly Avg. Rainfall vs Year')
	plt.xlabel('year')
	plt.ylabel('Rainfall')
plt.show()
