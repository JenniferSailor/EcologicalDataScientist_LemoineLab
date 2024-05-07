'''
Author: Jennifer Sailor
Date:
Order in Sequence: 2


'''
'''
Goal: Create Future Predictions!

For each k species
        For each climate model:
                Load climate model i and get bioclim variables
                For each ML model:
                        Predict for ml model j
                average all j ML models to generate prediction for climate model i
        average all i climate models to generate prediction for species k

Hierachy on stacks:
    Highest: meta_ensemble
    Medium:  ensembler
    Lowest:  prediction


'''
#%%
import numpy as np
import pandas as pd
import os
from osgeo import gdal
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

#load the machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
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

#________________LOAD DATA FOLDERS__________________
Data_folder = 'Documents/jsailor/knp_butterflies/present_absent_data'
Data_dir = os.fsencode(Data_folder)
dataname_list = sorted(os.listdir(Data_dir))

#used for labeling
results_list = ['Belenois_aurota', 'Cacyreus_marshalli', 'Danaus_chrysippus', 'Eronia_cleodora', 'Mycalesis_rhacotis', 'Vanessa_cardui']

#ML Models
#doesn't need to be sorted because we are calling it specifically
names = ["Logistic_Regression", "Nearest_Neighbors", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes",  "QDA"]


#Raster files used after multicolinearity
raster_used = [4,6,14,18,19]

raster_not_used_st = ['1', '2', '3', '5', '7', '8', '9', '10', '11', '12', '13', '15', '16', '17']


#%%
#________________LOAD RASTER FILES__________________
SSP2_45_folder = 'Documents/jsailor/knp_butterflies/FutureClim/SSP2-45'
SSP2_45_dir = os.fsencode(SSP2_45_folder)
SSP2_45_list = sorted(os.listdir(SSP2_45_dir))

SSP5_85_folder = 'Documents/jsailor/knp_butterflies/FutureClim/SSP5-85'
SSP5_85_dir = os.fsencode(SSP5_85_folder)
SSP5_85_list = sorted(os.listdir(SSP5_85_dir))

#%%
#_________________LOOP THROUGH BUTTERFLIES______________
for i in range(len(dataname_list)):
    filename2 = os.fsdecode(dataname_list[i])
    data_file = os.path.join(Data_folder, filename2)
    data_ = pd.read_csv(data_file)

    species = results_list[i]

    #______________REDO: b/c didnt save the transform information_______________
    #______________just need to get the fit for scalar transform
    #DELETE COLS B/C MULTICOLLINEARITY
    #based off values in model_predictions
    data_ = data_.drop(raster_not_used_st, axis = 1)
    #SPLIT DATA INTO TRAIN AND SPLIT
    X = data_.iloc[:, 1:].values
    y = data_.iloc[:, 0].values
    #STANDARDIZATION
    scalar = StandardScaler()
    Xs_train = scalar.fit_transform(X)

    #________________LOOP THROUGH RASTER MODELS_________
    for j in range(len(SSP2_45_list)+ len(SSP5_85_list)):
        #if we are only looking at SSP2_45 files still
        if j < len(SSP2_45_list):
            filename1 = os.fsdecode(SSP2_45_list[j])
            SSP2_45_file = os.path.join(SSP2_45_folder, filename1)
            raster_tif = gdal.Open(SSP2_45_file)
        #now looking at SSP5_85 files
        else:
            filename1 = os.fsdecode(SSP5_85_list[(j-4)])
            SSP5_85_file = os.path.join(SSP5_85_folder, filename1)
            raster_tif = gdal.Open(SSP5_85_file)

        #get all the raster bands we need (those included in raster_used list)
        for r in range(len(raster_used)):
            data_array = raster_tif.GetRasterBand(raster_used[r]).ReadAsArray()
            mask = np.isnan(data_array)
            data_array[mask] = -1000
            #concatinate all into one array
            if r == 0:
                raster_array = np.array(data_array.ravel())
            else:
                raster_array = np.c_[raster_array, data_array.ravel()]
        
        #if we are starting SSP2_45 or SSP5_85
        if j == 0 or j == 4:
            #list of ensembles
            stacked_ensemble = []

        
        #JEN whats the shape of this 
        raster_array_s = scalar.transform(raster_array)
        
        #______________LOOP THROUGH ML MODELS______________________________
        
        model_index = 0
        #list of predictions
        stacked_prediction = []

        for name in names:
            mod_name = f"Documents/jsailor/knp_butterflies/ML_files/applied_findings/{species}_{name}.pkl"
            model = pickle.load(open(mod_name, 'rb'))

            prediction_results = model.predict_proba(raster_array_s)[:,1]
            prediction_results = prediction_results.reshape(data_array.shape).astype(float)
            prediction_results[mask] = np.nan

            stacked_prediction.append(prediction_results)

            model_index += 1

        print("Finished: ", species, name, filename1)  
        stacked_prediction = np.stack(stacked_prediction)
        print("             Ensemble ", stacked_prediction.shape) 
        #expect a warning here
        ensemble_prediction = np.nanmean(stacked_prediction, axis = 0)
        stacked_ensemble.append(ensemble_prediction)
        save_file(ensemble_prediction, f"ensemble/{species}_{os.path.splitext(filename1)[0]}_ensemble_results", "pkl")

        if j == 3:
            stacked_ensemble = np.stack(stacked_ensemble)
            print("Finished SSP2_45: ", species, name, 'meta_ensemble', stacked_ensemble.shape)
            meta_ensemble = np.nanmean(stacked_ensemble, axis=0)
            save_file(meta_ensemble, f"{species}_SSP2_45_meta_ensemble_results", "pkl")
        elif j == 7:
            stacked_ensemble = np.stack(stacked_ensemble)
            print("Finished SSP5_85: ", species, name, 'meta_ensemble', stacked_ensemble.shape)
            meta_ensemble = np.nanmean(stacked_ensemble, axis=0)
            save_file(meta_ensemble, f"{species}_SSP5_85_meta_ensemble_results", "pkl")

print("Done")
