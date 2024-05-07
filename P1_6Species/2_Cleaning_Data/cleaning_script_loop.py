'''
Author: Jennifer Sailor
Date: 
Order in Sequence: 1

This loops through all the species folers 
and cleans the data in all 4 ways listed below
'''

#%%
import os
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj

from shapely.geometry import Point
import numpy as np
#%%
#________________Read In Data___________________________________
home_path = os.getcwd()
#os.chdir('Documents/server-share/GIS_Files/WorldClim2/Annual/5min')
os.chdir('Documents/server-share/GIS_Files/WorldClim2/Annual/2_5min')
# Open the GeoTIFF file
#dataset = gdal.Open('wc2.0_bio_5m_01.tif')
dataset = gdal.Open('wc2.0_bio_2.5m_01.tif')
os.chdir(home_path)

Data_folder = 'Documents/jsailor/knp_butterflies/top6_LL_data_'
Data_dir = os.fsencode(Data_folder)
dataname_list = sorted(os.listdir(Data_dir))

results_list = ['Belenois_aurota', 'Cacyreus_marshalli', 'Danaus_chrysippus', 'Eronia_cleodora', 'Mycalesis_rhacotis', 'Vanessa_cardui']

CITIES = pd.read_csv('Documents/jsailor/knp_butterflies/major_cities.csv')
#_________________Read in Shape Files__________________________
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
shape_africa = world.query('continent == "Africa"')

#%%
for i in range(len(dataname_list)):
     #________________SET DATA YOU WANT TO CLEAN___________________
     filename2 = os.fsdecode(dataname_list[i])
     data_file = os.path.join(Data_folder, filename2)
     data_ = pd.read_csv(data_file)

     print("Data file: ", data_.shape[0], results_list[i], dataname_list[i])

     #_______________PART 0 - REMOVE 'NA' VALUES____________________________
     #removes them if they are in either columns
     data_ = data_.dropna(subset=['decimalLatitude', 'decimalLongitude'])

     print("Removed NAN: ", data_.shape[0], results_list[i])

     #_______________PART 1 - REMOVE POINTS IN OCEAN__________________________
     geoAfricaShapes = gpd.GeoDataFrame(shape_africa)
     geoAfrica = geoAfricaShapes['geometry'].unary_union

     #for all index that don't intersect with shape file remove
     index_not_intersect = []
     for index, row in data_.iterrows():
          point_ = Point(row['decimalLongitude'], row['decimalLatitude'])
          intersection_ = geoAfrica.intersects(point_)
          if intersection_ == False:
               index_not_intersect.append(index)

     data_.drop(index_not_intersect, inplace = True)

     print("Removed Points in Ocean: ", data_.shape[0], results_list[i])

     #_______________PART 2 - REMOVE POINTS NEAR MAJOR CITIES__________________________
     num1 = 0.01 #b/c of papers

     #for each major cities remove all points within num1 diameter of a square
     for index, row in CITIES.iterrows():
          index = data_[ (data_['decimalLongitude'] > row['longitude']-num1) & (data_['decimalLongitude'] < row['longitude']+num1) & 
                         (data_['decimalLatitude'] > row['latitude']-num1) & (data_['decimalLatitude'] < row['latitude']+num1)].index
          data_.drop(index, inplace = True)
     print("Removed points near major cities: ", data_.shape[0], results_list[i])
     #_______________PART 3 - SPATIAL DISTRIBUTION__________________________
     #easy way - but not spatially statistically accurate
     '''
     #remove all but one duplicate of long, lat coordinate when rounded to __ decimal places
     decimal_places = 2 #b/c that's what lemoine did

     data_['grid_longitude'] = data_['decimalLongitude'].round(decimal_places)
     data_['grid_latitude'] = data_['decimalLatitude'].round(decimal_places)

     data_ = data_.drop_duplicates(subset=['grid_longitude', 'grid_latitude'])
     data_ = data_.drop(columns=['grid_latitude', 'grid_longitude'])
     '''
     #______________Retrieve Raster Values__________
     cols = dataset.RasterXSize
     rows = dataset.RasterYSize

     trans = dataset.GetGeoTransform()

     xOrigin = trans[0]; yOrigin = trans[3]
     pixelwidth = trans[1]; pixelheight = trans[5] #is a negative

     data = dataset.GetRasterBand(1).ReadAsArray(0,0,cols,rows)

     data_['raster_row'] = 0.0001
     data_['raster_col'] = 0.0001

     for index, row in data_.iterrows():
         raster_row = int((row['decimalLatitude'] - yOrigin) / pixelheight)
         raster_col = int((row['decimalLongitude'] - xOrigin) / pixelwidth)

         data_.at[index, 'raster_col'] = raster_col
         data_.at[index, 'raster_row'] = raster_row


     data_ = data_.drop_duplicates(subset=['raster_row', 'raster_col'])
     data_ = data_.drop(columns=['raster_row', 'raster_col'])


     print("Data has been Spatially Distributed: ", data_.shape[0], results_list[i])

     csv_name = f"Documents/jsailor/knp_butterflies/top6_clean_2_5_data/Africa_Lepidoptera_{results_list[i]}_CLEAN.csv"
     data_.to_csv(csv_name, index=False)

print("Complete")
