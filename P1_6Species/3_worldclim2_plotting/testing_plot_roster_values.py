'''
Author: Jennifer Sailor
Date: 
Order in Sequence: Testing

This was to confirm that what the way I was calculating the raster values was correct
'''

#%%
import os
from osgeo import gdal
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj
#%%
#__________________Load Data__________________
#directories of folder
#world clim 2 folder
WC2_folder = 'Documents/server-share/GIS_Files/WorldClim2/Annual/2_5min'
WC2_dir = os.fsencode(WC2_folder)

WC2_list = sorted(os.listdir(WC2_dir))
filename = os.fsdecode(WC2_list[0])

tif_file = os.path.join(WC2_folder, filename)
WC2_tif = gdal.Open(tif_file)

#__________________Load in Butterflies_____________
Data_folder = 'Documents/jsailor/knp_butterflies/top6_clean_2_5_data'
Data_dir = os.fsencode(Data_folder)
dataname_list = sorted(os.listdir(Data_dir))
filename2 = os.fsdecode(dataname_list[2])
data_file = os.path.join(Data_folder, filename2)
data_b = pd.read_csv(data_file)

#turn into array
data_array = WC2_tif.GetRasterBand(1).ReadAsArray()

#change type to int
wc2data_ = data_array.astype('int32')
data_ = np.where(data_array < -2147483640, -500, data_array)

#number of cols and rows
cols = WC2_tif.RasterXSize
rows = WC2_tif.RasterYSize
trans = WC2_tif.GetGeoTransform()

#figure out origins
xOrigin = trans[0]; 
yOrigin = trans[3]
pixelwidth = trans[1]; 
pixelheight = trans[5] #is a negative
print(xOrigin, yOrigin, pixelheight, pixelwidth)

#get columns
data_b['raster_row'] = 0.0001
data_b['raster_col'] = 0.0001
data_b['pixel_val'] = 0.0001

for index, row in data_b.iterrows():
    raster_row = int((row['decimalLatitude'] - yOrigin) / pixelheight)
    raster_col = int((row['decimalLongitude'] - xOrigin) / pixelwidth)
    pixel_val = data_[raster_row][raster_col]

    data_b.at[index, 'raster_col'] = raster_col
    data_b.at[index, 'raster_row'] = raster_row
    data_b.at[index, 'pixel_val'] = pixel_val


print(data_b.head(), data_b.shape)
data_no_nan = data_b[data_b['pixel_val'] != -500]
print(data_no_nan.head(), data_no_nan.shape)
#%%
plt.figure(figsize=(10, 10))
#plt.axis([3500, 6000, 3500, 1000])
plt.scatter(data_b['raster_col'], data_b['raster_row'],color='red', s=5)
plt.imshow(data_)
plt.colorbar()

plot_name = f"Documents/jsailor/knp_butterflies/plot_wc2.png"
plt.savefig(plot_name)
plt.close()