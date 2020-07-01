import os
import gdal
import numpy
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

def average_value(folder):
    """
    Arguments:
        folder: list of folders on which the average values has to be calculated.
    """
	year_avg,month_avg=[],[]
    month_count=0
	for f in folder:
	    for filename in os.listdir(os.getcwd()):
            with gdal.Open(os.path.join(os.cwd(), filename), 'r') as files:
                data=files.ReadAsArray().astype(np.float32)
                [r1,c1] = data[1].shape
                data_sum=np.sum(data)
                m_avg.append(data_sum/(r1*c1))
                year_sum=year_sum + data_sum
                month_count=month_count+1
                if month_count==12:
                    month_avg.append(m_avg)
                    year_avg.append(year_sum/12.0)
    """
    Return:
        month_avg: average rainfall value for the months, stored as list
        year_avg: average rainfall value for the years, stored as list
    """
    return month_avg, year_avg

def predict_image(predict, image_name):
    """
    Arguments:
        predict: the predicted array map.
        image_name: predefined tiff file name.
    """
    path=path = os.path.join(os.getcwd(),'Predicted')
    fig, axes = plt.subplots(2, 1))
    imshow(np.squeeze(predict[0]))
    plt.set_title('Final Predicted Image')
    plt.savefig(os.path.join(savepath,image_name))
    plt.show()
    
    imshow(x_train[idx])
    plt.set_title('Initial Image')
    plt.show()
