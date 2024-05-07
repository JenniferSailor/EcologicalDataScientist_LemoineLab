'''
Author: Jennifer Sailor
Date: 
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
#%%
#________________Read In Data___________________________________
DC = pd.read_csv('top6_LL_data_/Africa_Lepidoptera_Danaus_chrysippus_LL.csv')
VC = pd.read_csv('top6_LL_data_/Africa_Lepidoptera_Vanessa_cardui_LL.csv')
BA = pd.read_csv('top6_LL_data_/Africa_Lepidoptera_Belenois_aurota_LL.csv')
MR = pd.read_csv('top6_LL_data_/Africa_Lepidoptera_Mycalesis_rhacotis_LL.csv')
CM = pd.read_csv('top6_LL_data_/Africa_Lepidoptera_Cacyreus_marshalli_LL.csv')
EC = pd.read_csv('top6_LL_data_/Africa_Lepidoptera_Eronia_cleodora_LL.csv')

#_________________Read in Shape Files__________________________
shape_KNP = gpd.read_file('2017_kruger_national_park.shp')

#%%
#__________________Plotting Fun Stuff_______________________________
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.Robinson()})
zone_num = 36

#Customize size of scatter plot
dot_size = 2

# Plot scatter points
ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= DC, ms=dot_size, c='red', transform=ccrs.PlateCarree(), label = 'Danaus chrysippus')
#ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= VC, ms=dot_size, c='blue', transform=ccrs.PlateCarree(), label = 'Vanessa cardui')
#ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= BA, ms=dot_size, c='darkcyan', transform=ccrs.PlateCarree(), label = 'Belenois aurota')
#ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= MR, ms=dot_size, c='green', transform=ccrs.PlateCarree(), label = 'Mycalesis rhacotis')
#ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= CM, ms=dot_size, c='darkorange', transform=ccrs.PlateCarree(), label = 'Cacyreus marshalli')
#ax.plot('decimalLongitude', 'decimalLatitude', 'o', data= EC, ms=dot_size, c='limegreen', transform=ccrs.PlateCarree(), label = 'Eronia cleodora')

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
plt.savefig('Africamap.png')

# SA
ax.set_title('South Africa Occurences')
plt.legend( bbox_to_anchor=(0.7, 0.2), loc='center left')
ax.set_extent([16, 40, -35, -20], ccrs.PlateCarree())
plt.savefig('SAmap.png')

# KNP
ax.set_title('KNP Occurences')
plt.legend( bbox_to_anchor=(1, 0.5), loc='center left')
ax.set_extent([30.8, 32.1, -25.7, -22.1], ccrs.PlateCarree())
plt.savefig('KNPmap.png')

plt.close()