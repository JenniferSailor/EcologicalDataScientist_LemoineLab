'''
Author: Jennifer Sailor
Date: 9/19/23
Order in Sequence: 1

This in a manual approach to plot a species 
on KNP, South Africa, and Africa
all plots are saved
'''

#%%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj
import os
#%%
#________________Read In Data___________________________________
#________Load Data Folder on Species_________________
Data_folder = 'Documents/jsailor/knp_butterflies/all_kloppers/all_species_cleaned_data'
Data_dir = os.fsencode(Data_folder)
dataname_list = sorted(os.listdir(Data_dir))

Data_folder2 = 'Documents/jsailor/knp_butterflies/all_kloppers/all_species_API_data'
Data_dir2 = os.fsencode(Data_folder2)
dataname_list2 = sorted(os.listdir(Data_dir2))

#_________________Read in Shape Files__________________________
shape_KNP = gpd.read_file('Documents/jsailor/knp_butterflies/shape_files/2017_kruger_national_park.shp')

#%%
#__________________Plotting Fun Stuff_______________________________
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.Robinson()})
zone_num = 36

#Customize size of scatter plot
dot_size = 2

for i in range(1):
	i = 115
	#data
	#clean
	filename2 = os.fsdecode(dataname_list[i])
	data_file = os.path.join(Data_folder, filename2)
	data_ = pd.read_csv(data_file)
	#before clean
	filename3 = os.fsdecode(dataname_list2[i])
	data_file2 = os.path.join(Data_folder2, filename3)
	data_API = pd.read_csv(data_file2)

	print(filename2)

	# Plot scatter points
	ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= data_API, ms=dot_size, c='blue', transform=ccrs.PlateCarree(), label = 'API Pull')
	ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= data_, ms=dot_size, c='red', transform=ccrs.PlateCarree(), label = 'Clean')

# Plot Africa, Borders of Countries, and KNP
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black')
ax.add_geometries(shape_KNP.geometry, alpha=0.5, facecolor='grey', edgecolor='black', crs=ccrs.UTM(zone_num, southern_hemisphere=True))

#Plot fun stuff
#ax.set_facecolor(cfeature.COLORS['water'])
#ax.add_feature(cfeature.LAND)
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.LAKES, alpha=0.5)
#ax.add_feature(cfeature.RIVERS)

#gridlines
#gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5)
#gl.ylocator = mticker.FixedLocator(np.arange(-90,90,5))
#gl.xlocator = mticker.FixedLocator(np.arange(-180,180,5))

# Africa
ax.set_title('Africa Occurences')
plt.legend( bbox_to_anchor=(1, 0.5), loc='center left')
ax.set_extent([-20, 60, 40, -37], ccrs.PlateCarree()) #lat, long range
plt.savefig('Documents/jsailor/knp_butterflies/all_kloppers/Africamap.png')

# SA
ax.set_title('South Africa Occurences')
plt.legend( bbox_to_anchor=(0.7, 0.2), loc='center left')
ax.set_extent([16, 40, -35, -20], ccrs.PlateCarree())
plt.savefig('Documents/jsailor/knp_butterflies/all_kloppers/SAmap.png')

# KNP
ax.set_title('KNP Occurences')
plt.legend( bbox_to_anchor=(1, 0.5), loc='center left')
ax.set_extent([30.8, 32.1, -25.7, -22.1], ccrs.PlateCarree())
plt.savefig('Documents/jsailor/knp_butterflies/all_kloppers/KNPmap.png')

plt.close()