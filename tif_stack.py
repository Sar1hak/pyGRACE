
#Stracking of tiff files
#import gdal


'''
from gdal import gdal_merge
def tif_stack(file1,file2,file3):

	out_filename = '200001.tif'
	out_format = 'tif'
	nodata_value = -9999
	return (gdal_merge.py -o out_filename -of out_format -ps 256 256 -separate -q -n nodata_value file1 file2 file3)
'''
'''
gdal_merge.py [-o out_filename] [-of out_format] [-co NAME=VALUE]*
              [-ps pixelsize_x pixelsize_y] [-tap] [-separate] [-q] [-v] [-pct]
              [-ul_lr ulx uly lrx lry] [-init "value [value...]"]
              [-n nodata_value] [-a_nodata output_nodata_value]
              [-ot datatype] [-createonly] input_files
'''

#import os
#import subprocess 
#def tif_stack(file1,file2,file3,index): 

#	out_filename = 'stacked'+ str(index) +'.tif'
#	out_format = 'gtiff'
#	nodata_value = -9999
	
#	command = ('C:/Users/LENOVO/Anaconda3/Scripts/gdal_merge.py -o '+out_filename+' -of '+ 
#	                out_format+' -separate -q -n '+str(nodata_value)+ ' ' +
#					file1 + ' ' + file2 + ' ' + file3 )
#	print (command)
       
#	## run gdal_merge.py tool with os call
#	os.system(command)

#	output = None
#	try:
#		output = subprocess.check_output(command,shell=True)
#	except subprocess.CalledProcessError as e:
#	    output = e.output
#	return output


'''       
      # warp to clean up data
      newoutputname = pathname +  "ET_ensemble_" + str(year) + "-" + str(month) + "-01.tif "
      warpcommand  = "gdalwarp -s_srs EPSG:4326 -t_srs epsg:4326 -dstnodata 0 -of GTiff " + outputname  + " " + newoutputname
       
      print " "
      print warpcommand
       
      # execute gdalwarp command with os call
      os.system(warpcommand)
''' 