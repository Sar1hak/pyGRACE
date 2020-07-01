from __future__ import print_function
from keras.models import Model
from keras.layers import ( Conv2D, 
                           MaxPooling2D, 
                           Dropout, 
                           Input, 
                           concatenate, 
                           UpSampling2D)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
import gdal
import tensorflow as tf
import numpy as np
import os
import argparse
 
tf.enable_eager_execution()
 
# import other files in use
from Norm_Convol_func import normalize_convolve
from Unet_func import unet
from tif_calc import average_value, predict_image
from plotting import loss_plot, monthly_avg_plot, yearly_avg_plot
from stage import stage1, stage2
#from merge_func import merge
#from tif_stack import tif_stack


#######MAIN#######
def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
	                help = "path to the input image")
    args = vars(ap.parse_args())

    path = os.getcwd()
    folder=["NOAH TWSA", "Precipitation", "Temperature"]	
    if not os.path.exists(path):
	    print("ERROR {} not found!".format(path))
	    exit()

	# input image dimensions
    img_rows, img_cols = 256, 256

    ##### Input Data Extraction and Normalization #####
    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_directory( '/data',
                                                         target_size = (img_rows, img_cols),
                                                         batch_size = 32,
                                                         shuffle = False)
    test_generator = test_datagen.flow_from_directory( '/data',
                                                       target_size = (img_rows, img_cols),
                                                       batch_size = 32,
                                                       shuffle = False)

    ##### Hidden Network formation #####
    #merged_features = stage2(NOAH_layer, Precip_layer, Temp_layer)    
    merged_features = stage2(stage1(), stage1, stage1)
    unet_model = unet()
    model = Add()([merged_features.output,unet_model.output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    # Define checkpoint callback
    checkpoint_path = "/checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
    callbacks = [ tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir='./logs'),
              cp_callback]

    history = model.fit_generator( train_generator, steps_per_epoch=100, epochs=20)
    test_generator.reset()
    ##### Image Prdiction and saving in Predicted directory #####
    predict = model.predict_generator(test_generator, verbose=1)
    predict = (predict > 0.5).astype(np.uint8)
    j=0
    for i in predict:
        predict_image(i, ("img" + str(j) + ".tif"))
        j=j + 1
    
    #### PLOT GRAPHS ####
    initial_month_avg, initial_year_avg = average_value(folder)
    final_month_avg, final_year_avg = average_value(folder)
    loss_plot(history)
    monthly_avg_plot(initial_month_avg, final_month_avg)
    yearly_avg_plot(initial_year_avg, final_year_avg)


if __name__ == "__main__":
    main()