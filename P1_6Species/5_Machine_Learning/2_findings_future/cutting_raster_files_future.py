'''
Author: Jennifer Sailor
Date:
Order in Sequence: 1

cutting future raster files
'''

#%%
import numpy as np
import pandas as pd
import os
from osgeo import gdal

#%%
#_______________________SAVING FILE_______________________
def save_file(tosave, filename, filetype):
    filename2 = f"Documents/jsailor/knp_butterflies/ML_files/future_applied_pred/{filename}.{filetype}"
    if filetype == 'pkl':
        with open(filename2, 'wb') as file:
            pickle.dump(tosave, file)
    elif filetype == 'csv':
        tosave.to_csv(filename2, index = False)
    else:
        print('error in save file function')

#%%
#_________________________LOAD FILES______________________
SSP2_45_folder = 'Documents/server-share/GIS_Files/FutureClimate/CMIP6/2_5min/2041-2060/SSP2-45'
SSP2_45_dir = os.fsencode(SSP2_45_folder)
SSP2_45_list = sorted(os.listdir(SSP2_45_dir))

SSP5_85_folder = 'Documents/server-share/GIS_Files/FutureClimate/CMIP6/2_5min/2041-2060/SSP5-85'
SSP5_85_dir = os.fsencode(SSP5_85_folder)
SSP5_85_list = sorted(os.listdir(SSP5_85_dir))

#%%
#_____________CLIP RASTER FILES_____________________
for i in range(len(SSP2_45_list)):
	filename = os.fsdecode(SSP2_45_list[i])
    input_raster = os.path.join(SSP2_45_folder, filename)
    shapefile_path = 'Documents/jsailor/knp_butterflies/2017_kruger_national_park_standard.shp'
    output_raster = f"Documents/jsailor/knp_butterflies/FutureClim/SSP2-45/{os.path.splitext(filename)[0]}.tif"
    command = f'gdalwarp -q -cutline {shapefile_path} -crop_to_cutline {input_raster} {output_raster} --config GDALWARP_IGNORE_BAD_CUTLINE YES -dstNodata {np.nan}'
    os.system(command)

    filename = os.fsdecode(SSP5_85_list[i])
    input_raster = os.path.join(SSP5_85_folder, filename)
    shapefile_path = 'Documents/jsailor/knp_butterflies/2017_kruger_national_park_standard.shp'
    output_raster = f"Documents/jsailor/knp_butterflies/FutureClim/SSP5-85/{os.path.splitext(filename)[0]}.tif"
    command = f'gdalwarp -q -cutline {shapefile_path} -crop_to_cutline {input_raster} {output_raster} --config GDALWARP_IGNORE_BAD_CUTLINE YES -dstNodata {np.nan}'
    os.system(command)

