'''
Author: Jennifer Sailor
Date:
Order in Sequence: 3

plot the predictions created in ml_applied_findings
'''

#%%
import numpy as np
import pandas as pd
import os
from osgeo import gdal
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#%%

#_________PLOT FUNCTION__________________________
def plot_prediction(data_array, species_name, model_name, subfolder = False):
	
	fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.Robinson()})
	zone_num = 36
	# coolwarm 
	im = ax.imshow(data_array, cmap ='plasma', vmin = 0, vmax = 1)
	plt.colorbar(im)

	#ax.add_geometries(shape_KNP.geometry, alpha=1, facecolor='black', edgecolor='black', crs=ccrs.UTM(zone_num, southern_hemisphere=True))

	ax.set_title(f'{species_name} {model_name} Prediction')
	
	if subfolder == True:
		plot_name = f"Documents/jsailor/knp_butterflies/ML_files/future_applied_pred/pred_plots/individual/{species_name}_{model_name}_pred.png"
	else:
		plot_name = f"Documents/jsailor/knp_butterflies/ML_files/future_applied_pred/pred_plots/{species_name}_{model_name}_pred.png"

	plt.savefig(plot_name)
	plt.close()

#%%
#__________LOAD DATA___________________________
folder_path = "Documents/jsailor/knp_butterflies/ML_files/future_applied_pred"
files = os.listdir(folder_path)

ensemble_files = [file_name for file_name in files if file_name.endswith('ensemble_results.pkl')]
ensemble_files.sort()

#used for labeling
results_list = ['Belenois_aurota', 'Cacyreus_marshalli', 'Danaus_chrysippus', 'Eronia_cleodora', 'Mycalesis_rhacotis', 'Vanessa_cardui']
names = ["SSP2_45", "SSP5_85"]
names.sort()

#%%
#________PLOT EACH MODELS PRED FOR EACH BUTTERFLY__________________
model_name = 0
butterfly_name = -1
for i in range(len(individual_files)):
	if i== 0 or model_name == (len(names)-1):
		model_name = 0
		butterfly_name += 1
	else:
		model_name += 1
	individual_file_name = os.path.join(folder_path, individual_files[i])
	data_ = pickle.load(open(individual_file_name, 'rb'))
	#plot
	plot_prediction(data_, results_list[butterfly_name], names[model_name], subfolder=True)

#%%
#_______PLOT EACH ENSEMBLE MODEL FOR EACH BUTTERFLY_______________
for j in range(len(ensemble_files)):
	ensemble_file_name = os.path.join(folder_path, ensemble_files[j])
	data_ = pickle.load(open(ensemble_file_name, 'rb'))

	#plot
	if j < len(results_list):
		print("plotting: ", ensemble_files[j], names[0])
		plot_prediction(data_, results_list[j], f"Ensemble_{names[0]}")
	else:
		print("plotting: ", ensemble_files[j], names[1])
		plot_prediction(data_, results_list[(j-len(results_list))],  f"Ensemble_{names[1]}")

print("DONE!")
