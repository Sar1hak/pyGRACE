import os
import gdal
import numpy

def average_value():
	#os.chdir("H:/summer project/Rainfall")
	
	data,data1,data2=[],[],[]
	yr,m=[],[]
	y=0
	folder=['NOAH TWSA','Temperature','Precipitation']
    #open folder	
	for x in folder:
        #open file		
		for i in range(200001,201813):
			dir = gdal.Open('H:/summer project/Rainfall/'+ x +str(i)+'.tif')
			data = dir.ReadAsArray().astype(numpy.float32)
			
			[r1,c1] = data[1].shape
			sum1=0
			count = 0
			for i in range(0,r1):
				for j in range(0,c1):
					if(data[i][j]!=-9999):
						sum1 = data[i][j]+ sum1
						count=count+1
# Month average value			
			m.append(sum1/count)
			y=y+(sum1/count)
# Annual average value			
			yr.append((y/12.0))

			if str(i).endswith('12'):
				data.append(m)
				i = i + 88
				y=0
	data1.append(data)
	data2.append(yr)
	return data1,data