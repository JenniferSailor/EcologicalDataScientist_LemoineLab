'''
Author: Jennifer Sailor
Date:
Order in Sequence: 1

Creating the Pseudo Data and save into CSV
1:1 ratio with the present and absent data
'''

#%%
import os
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj

from shapely.geometry import Point
import numpy as np
import random

#%%
#__________________Load Data__________________
#clean data folder
Data_folder = 'Documents/jsailor/knp_butterflies/top6_clean_2_5_data'
Data_dir = os.fsencode(Data_folder)

#_________________Read in Shape Files__________________________
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
shape_africa = world.query('continent == "Africa"')

#________________Create Function that Finds a Random Point_______________
geoAfricaShapes = gpd.GeoDataFrame(shape_africa)
geoAfrica = geoAfricaShapes['geometry'].unary_union

dataname_list = ['VC', 'MR', 'EC', 'CM', 'BA', 'DC']
data_index = 0

for file in os.listdir(Data_dir):
	filename = os.fsdecode(file)
	data_file = os.path.join(Data_folder, filename)
	data_ = pd.read_csv(data_file)
	num_rows = data_.shape[0]

	pseudo_data = pd.DataFrame(columns = ['decimalLatitude', 'decimalLongitude'])

	print('starting', dataname_list[data_index], filename)
	for i in range(num_rows):
		got_a_point = False
		
		while (got_a_point == False):
			lat = np.float64(random.uniform(-35, 40)) 
			lon = np.float64(random.uniform(-20, 55))
			point_ = Point(lon, lat)

			#has to be in africa
			while (geoAfrica.intersects(point_) == False): #want to intersect which is true
				lat = np.float64(random.uniform(-35, 40)) 
				lon = np.float64(random.uniform(-20, 55))
				point_ = Point(lon, lat)
			got_a_point = True
			
			#can't already be in pseudo_data
            for index, row in pseudo_data.iterrows():
				if row['decimalLongitude'].round(2) == lon.round(2) and row['decimalLatitude'].round(2) == lat.round(2):
            			got_a_point = False
            			break
			
            	if got_a_point == False:
            		continue
            	#can't be already in data_
			for index2, row2 in data_.iterrows():
				if round(row2['decimalLongitude'],2) == lon.round(2) and round(row2['decimalLatitude'], 2) == lat.round(2):
            			got_a_point = False
            			break
        if i == (num_rows//2):
          	print('halfway with', dataname_list[data_index])
        if i == (num_rows-1):
          	print('done with', dataname_list[data_index])

		pseudo_data.at[i, 'decimalLatitude'] = lat
		pseudo_data.at[i, 'decimalLongitude'] = lon

	#save data based on name of file
	#still need to check that it doesn't match any of the data points where we know they occur
	# or like with .2 of it
	csv_name = f"Documents/jsailor/knp_butterflies/pseudo_data/pseudo_{dataname_list[data_index]}.csv"
	pseudo_data.to_csv(csv_name, index=False)
	data_index += 1
print("ALLLLLL DONEEE!!!!")

