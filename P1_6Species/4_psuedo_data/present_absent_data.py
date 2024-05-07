'''
Author: Jennifer Sailor
Date:
Order in Sequence: 2

Building the Dataframe: consisting of whether present or apsent and the 19 variables associated with value
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
Pseudo_folder = 'Documents/jsailor/knp_butterflies/pseudo_data'
Pseudo_dir = os.fsencode(Pseudo_folder)

WC2_list = sorted(os.listdir(WC2_dir))
dataname_list = sorted(os.listdir(Data_dir))
pseudoname_list = sorted(os.listdir(Pseudo_dir))

results_list = ['BA', 'CM', 'DC', 'EC', 'MR', 'VC']


#________________Loop Through Files - Butterflies______________________
for i in range(len(dataname_list)):

	p_a_data = pd.DataFrame(columns = ['occurence_present', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])

	#load specific file
	filename2 = os.fsdecode(dataname_list[i])
	data_file = os.path.join(Data_folder, filename2)
	data_ = pd.read_csv(data_file)

	num_rows = data_.shape[0]

	filename3 = os.fsdecode(pseudoname_list[i])
	pseudo_file = os.path.join(Pseudo_folder, filename3)
	data_p = pd.read_csv(pseudo_file)

	print(filename2, filename3)

	#________________Loop Through Files - WC2______________________
	column_num = 0
	for j in range(len(WC2_list)):
		#load specific file
		filename = os.fsdecode(WC2_list[j])

		#only get files ending with .tif
		if filename.endswith(".tif"):
			tif_file = os.path.join(WC2_folder, filename)
			WC2_tif = gdal.Open(tif_file)
			column_num += 1
		else:
			continue

		print(filename, "index is", column_num)
		#___________Analysis using WC2 raster data_______________________
		cols = WC2_tif.RasterXSize; rows = WC2_tif.RasterYSize

		trans = WC2_tif.GetGeoTransform()

		xOrigin = trans[0]; yOrigin = trans[3]
		pixelwidth = trans[1]; pixelheight = trans[5] #is a negative

		WC2_tif_data = WC2_tif.GetRasterBand(1).ReadAsArray(0,0,cols,rows)

		#_____________________Loop through Data Present______________________________
		for index, row in data_.iterrows():
		    raster_row = int((row['decimalLatitude'] - yOrigin) / pixelheight)
		    raster_col = int((row['decimalLongitude'] - xOrigin) / pixelwidth)
		    raster_pix = WC2_tif_data[raster_row][raster_col]

		    p_a_data.at[index, 'occurence_present'] = 1

		    if int(raster_pix) < -100_000:
		    	p_a_data.at[index, str(column_num)] = float('NAN');
		    else:
		    	p_a_data.at[index, str(column_num)] = format(raster_pix, 'f')

		#_____________________Loop through Data Absent______________________________
		for index, row in data_p.iterrows():
		    raster_row = int((row['decimalLatitude'] - yOrigin) / pixelheight)
		    raster_col = int((row['decimalLongitude'] - xOrigin) / pixelwidth)
		    raster_pix = WC2_tif_data[raster_row][raster_col]

		    p_a_data.at[(index+num_rows), 'occurence_present'] = 0

		    if int(raster_pix) < -100_000:
		    	p_a_data.at[(index+num_rows), str(column_num)] = float('NAN');
		    else:
		    	p_a_data.at[(index+num_rows), str(column_num)] = format(raster_pix, 'f')

	#_____________________Drop NA's and make even______________________________
	print('Before drop: # of Present =',len(p_a_data[p_a_data['occurence_present']==1]),'# of Absent =' ,len(p_a_data[p_a_data['occurence_present']==0]) )
	
	data_drop = p_a_data.dropna() 
	data_drop.reset_index(drop=True)

	num_present = len(data_drop[data_drop['occurence_present']==1])
	num_absent = len(data_drop[data_drop['occurence_present']==0])
	difference = abs(num_present-num_absent)

	print('After drop:  # of Present =',num_present,' # of Absent =' , num_absent, "difference = ", difference )
	#making it even
	if num_present > num_absent:
		#remove amount difference from present
		data_drop.drop(data_drop.index[:difference], inplace = True)
	elif num_present < num_absent:
		#remove amount difference from absent
		data_drop.drop(data_drop.index[(-1*difference):], inplace = True) 
	#or they are even so do nothing

	#confirming eveness
	num_present = len(data_drop[data_drop['occurence_present']==1])
	num_absent = len(data_drop[data_drop['occurence_present']==0])
	print('Made even:  # of Present =',num_present,' # of Absent =' , num_absent )
	

	csv_name = f"Documents/jsailor/knp_butterflies/present_absent_data/present_absent_{results_list[i]}.csv"
	data_drop.to_csv(csv_name, index=False)
	#print(p_a_data.head())
	#print(p_a_data.tail())
	print(results_list[i], 'done')
 


		
print("complete")