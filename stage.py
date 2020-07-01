from keras.models import Model
from keras.layers import ( Conv2D, 
                           Input, 
                           concatenate)


def stage1(OUTPUT_MASK_CHANNELS=1):
	inputs = Input((256,256,3))
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal')(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                           kernel_initializer = 'he_normal')(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                           kernel_initializer = 'he_normal')(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                           kernel_initializer = 'he_normal')(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                           kernel_initializer = 'he_normal')(conv3)
	conv = Conv2D(OUTPUT_MASK_CHANNELS,1, 1, activation = 'sigmoid')(conv3)
    """
    Return:
        conv: Convolution layer path.
    """
	return conv


def stage2(NOAH_layer, Precip_layer, Temp_layer):
    """
    Arguments:
        NOAH_layer: Convolution layers set for NOAH inputs.
        Precip_layer: Convolution layers set for Precipitation inputs.
        Temp_layer: Convolution layers set for Temperature inputs.
    """
    merge_one = concatenate([NOAH_layer, Precip_layer])
    merged_features = concatenate([merge_one, Temp_layer])
    model = Model(inputs=[NOAH_layer, Precip_layer, Temp_layer], outputs=merged_features)
    """
    Return:
        model: Model formed by merging the features of the the three layers
    """
    return model
