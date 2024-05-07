'''
Author: Jennifer Sailor
Date: 
Order in Sequence: Skip and use wc2_plotting_updated

The problem with this code is it labels incorrectly
'''

#%%
import os
from osgeo import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
#__________________Load Data__________________
#directories of folder
#world clim 2 folder
WC2_folder = 'Documents/server-share/GIS_Files/WorldClim2/Annual/2_5min'
WC2_dir = os.fsencode(WC2_folder)
#clean data folder
Data_folder = 'Documents/jsailor/knp_butterflies/top6_clean_2_5_data'
Data_dir = os.fsencode(Data_folder)

index_WC2_var = 0 #used for naming and knowing what wc2 variable we ar eon

#________________Loop Through Files - WC2______________________
for file in os.listdir(WC2_dir):
	#load specific file
	filename = os.fsdecode(file)

	#only get files ending with .tif
	if filename.endswith(".tif"):
		tif_file = os.path.join(WC2_folder, filename)
		WC2_tif = gdal.Open(tif_file)
	else:
		continue

	index_DATA = 0 #keeps track of # of Butterfly data we are on
	index_WC2_var += 1 #loaded a file so add to index

	#__________________START PLOT___________________________
	#start plot for the 6 sub plots of each Butterfly data at each wc2 data
	fig, ax = plt.subplots(nrows=1, ncols=6, figsize = (20,4), sharex = True)

	#________________Loop Through Files - Butterflies______________________
	for file2 in os.listdir(Data_dir):
		#load specific file
		filename2 = os.fsdecode(file2)
		data_file = os.path.join(Data_folder, filename2)
		data_ = pd.read_csv(data_file)

		#___________Analysis using WC2 raster data_______________________
		cols = WC2_tif.RasterXSize; rows = WC2_tif.RasterYSize

		trans = WC2_tif.GetGeoTransform()

		xOrigin = trans[0]; yOrigin = trans[3]
		pixelwidth = trans[1]; pixelheight = -1*trans[5] #is a negative

		WC2_tif_data = WC2_tif.GetRasterBand(1).ReadAsArray(0,0,cols,rows)

		data_['raster_pixel'] = 0.0001

		#this is just used instead of -google
		num1 = -100 

		#_____________________Loop through Data______________________________
		for index, row in data_.iterrows():
		    raster_col = int((row['decimalLatitude'] - xOrigin) / pixelwidth)
		    raster_row = int((yOrigin - row['decimalLongitude']) / pixelheight)
		    raster_pix = WC2_tif_data[raster_row][raster_col]

		    if int(raster_pix) < 0:
		    	data_.at[index, 'raster_pixel'] = num1;
		    else:
		    	data_.at[index, 'raster_pixel'] = format(raster_pix, 'f')

		#take out any negatively large raster pixel values
		data_ = data_.drop(data_[data_['raster_pixel']== num1].index)
		data_ = data_.reset_index(drop=True)

		#TESTING
		#print(data_['raster_pixel'].max(), data_['raster_pixel'].min())

		#_________________ACTUALLY PLOT_________________________
		index_DATA += 1 
		#plot for each dataset in a subplot
		plt.subplot(1,6,index_DATA)
		plt.hist(data_['raster_pixel'], bins = 10) #set bins to whatever user wants
		plt.title(data_['species'][1])

		if index_DATA == 6: #all 6 species plotted
			#save plot # corresponds to wc2 variable
			plot_name = f"Documents/jsailor/knp_butterflies/wc2_plots/plot_{index_WC2_var}.png"
			print(index_WC2_var, "out of 19 done", filename)
			plt.savefig(plot_name)
			plt.close() #clear plot for next run
		
print("complete - look at what wc2 made each plot")