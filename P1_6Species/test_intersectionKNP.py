'''
Author: Jennifer Sailor
Date: 9/24
Order in Sequence: 

ccrs.UTM(zone_num, southern_hemisphere=True)
'''

#%%
import pygbif
import pandas as pd
import os
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import cartopy.crs as ccrs

#%%
#_______Load in shape file_____________
shape_KNP = gpd.read_file('Documents/jsailor/knp_butterflies/shape_files/2017_kruger_national_park.shp')

#%%
# -22.926603, 31.219978
#-24.913585, 31.713200
#-25.145312, 31.219978
lat = -24.4155
lon = 31.5898

var1 = [[-24.4155, 31.5898], [-22.926603, 31.219978], [-24.913585, 31.713200], [-25.145312, 31.219978]]

for i in range(len(var1)):
	lat = var1[i][0]
	lon = var1[i][1]

	geoAfricaShapes = gpd.GeoDataFrame(shape_KNP)
	geoAfrica = geoAfricaShapes['geometry'].unary_union

	#"EPSG:32662"
	#ccrs.PlateCarree().proj4_init
	trans = pyproj.Transformer.from_crs("EPSG:4326", ccrs.UTM(36, southern_hemisphere=True).proj4_init, always_xy = True) 
	#for all index that don't intersect with shape file remove

	point_ = Point(lon, lat)
	trans_point = trans.transform(point_.x, point_.y)
	point_ = Point(trans_point[0], trans_point[1])
	intersection_ = geoAfrica.intersects(point_)
	print(intersection_)
