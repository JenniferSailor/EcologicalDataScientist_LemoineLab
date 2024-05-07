'''
Author: Jennifer Sailor
Date:
Order in Sequence: 1

goal to cut down roster file to shape of knp
'''

#%%
import os
from osgeo import gdal, osr

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from shapely.geometry import mapping


#%%
#__________________Load Data__________________
#directories of folder
#world clim 2 folder
WC2_folder = 'Documents/server-share/GIS_Files/WorldClim2/Annual/2_5min'
WC2_dir = os.fsencode(WC2_folder)
WC2_list = sorted(os.listdir(WC2_dir))
#filename = os.fsdecode(WC2_list[0])

#tif_file = os.path.join(WC2_folder, filename)
#WC2_tif = gdal.Open(tif_file)
#data_array = WC2_tif.GetRasterBand(1).ReadAsArray()


#_________________Read in Shape Files__________________________
shape_KNP = gpd.read_file('Documents/jsailor/knp_butterflies/2017_kruger_national_park.shp')
shape_KNP.crs = "EPSG:32736"
reprojected_shape_KNP = shape_KNP.to_crs("EPSG:4326") #this is the standard
reprojected_shape_KNP.to_file("Documents/jsailor/knp_butterflies/2017_kruger_national_park_standard.shp")
print(gpd.read_file('Documents/jsailor/knp_butterflies/2017_kruger_national_park_standard.shp').crs)



#%%
#_______________Lemoine Code_____________________________
folder_path = WC2_folder
files = os.listdir(folder_path)
#only get files with .tif
tif_files = [file_name for file_name in files if file_name.endswith('.tif')]

tif_files.sort()

for i in range(len(tif_files)):
    input_raster = os.path.join(folder_path, tif_files[i])
    shapefile_path = 'Documents/jsailor/knp_butterflies/2017_kruger_national_park_standard.shp'
    output_raster = f"Documents/jsailor/knp_butterflies/cut_rasters/{os.path.splitext(tif_files[i])[0]}_KNP.tif"
    command = f'gdalwarp -q -cutline {shapefile_path} -crop_to_cutline {input_raster} {output_raster} --config GDALWARP_IGNORE_BAD_CUTLINE YES -dstNodata {np.nan}'
    os.system(command)
#DONE

#now just testing
WC2_tif = gdal.Open('Documents/jsailor/knp_butterflies/cut_rasters/wc2.0_bio_2.5m_01_KNP.tif')
data_array = WC2_tif.GetRasterBand(1).ReadAsArray()
data_array[np.isnan(data_array)] = -20
wc2data_ = data_array.astype('int32')
print(data_array)
#%%
#_________________Plot_____________________________________________
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.Robinson()})
zone_num = 36

im = ax.imshow(wc2data_)
#plt.imshow()
plt.colorbar(im)

ax.add_geometries(shape_KNP.geometry, alpha=1, facecolor='black', edgecolor='black', crs=ccrs.UTM(zone_num, southern_hemisphere=True))

#ax.set_title('KNP Occurences')
#ax.set_extent([30.8, 32.1, -25.7, -22.1], ccrs.PlateCarree())

plot_name = f"Documents/jsailor/knp_butterflies/plot_wc2.png"
plt.savefig(plot_name)
plt.close()