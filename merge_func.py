
#import rasterio.merge
'''
bounds=None
res=None
nodata=None
precision=7
'''
#def merge(input1,bounds, res, nodata, precision):
#    import warnings
#    warnings.warn("Deprecated; Use rasterio.merge instead", DeprecationWarning)
#    return rasterio.merge.merge(input1, bounds, res, nodata, precision)
		
'''
for file in folder:
	file1=rasterio.open("NOAH TWSA/"+str(x))
	file2=rasterio.open("Precipitation/"+str(x))
	file3=rasterio.open("Temperature"+str(x))
	dest, output_transform = merge([file1,file2,file3],bounds,res,nodata,precision)
	x=x+1
	with rasterio.open("raster1.tif") as src:
		out_meta = src.meta.copy()    
	out_meta.update({"driver": "GTiff",
                 "height": dest.shape[1],
                 "width": dest.shape[2],
                 "transform": output_transform})
	with rasterio.open("mergedRasters.tif", "w", **out_meta) as dest1:
		dest1.write(dest)
'''