'''
Author: Jennifer Sailor
Date:
Order in Sequence: 2

training full models
predicting on shape files
'''

#%%
import numpy as np
import pandas as pd
import os
from osgeo import gdal
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from sklearn import clone
#%%
#_______________________SAVING FILE_______________________
def save_file(tosave, filename, filetype):
    filename2 = f"Documents/jsailor/knp_butterflies/ML_files/applied_findings/{filename}.{filetype}"
    if filetype == 'pkl':
        with open(filename2, 'wb') as file:
            pickle.dump(tosave, file)
    elif filetype == 'csv':
        tosave.to_csv(filename2, index = False)
    else:
        print('error in save file function')
#%%
#________________LOAD DATA FOLDERS__________________
Data_folder = 'Documents/jsailor/knp_butterflies/present_absent_data'
Data_dir = os.fsencode(Data_folder)
dataname_list = sorted(os.listdir(Data_dir))

#used for labeling
results_list = ['Belenois_aurota', 'Cacyreus_marshalli', 'Danaus_chrysippus', 'Eronia_cleodora', 'Mycalesis_rhacotis', 'Vanessa_cardui']

#ML Models
names = ["Logistic_Regression", "Nearest_Neighbors", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes",  "QDA"]

classifiers = [
    LogisticRegression(random_state = 0), #default
    KNeighborsClassifier(leaf_size = 1, p = 1, n_neighbors = 7), #default: leaf_size = 30, p = 2, n_neighbors = 5
    GaussianProcessClassifier(), #default
    DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0), #default:criterion = 'gini', max_depth = 15
    RandomForestClassifier(max_depth = 15, random_state = 0), #default: max_depth = none
    MLPClassifier(max_iter = 1000, random_state = 0), #default: max_iter = 200
    AdaBoostClassifier(learning_rate = 0.1, n_estimators = 300, random_state = 0), #default: learning_rate = 1, n_estimators = 50
    GaussianNB(), #none
    QuadraticDiscriminantAnalysis() #none
    ]

print('JEN: double check')
#Raster files used after multicolinearity
raster_used = [4,6,14,18,19]
raster_not_used_st = ['1', '2', '3', '5', '7', '8', '9', '10', '11', '12', '13', '15', '16', '17']

#%%
#________________LOAD RASTER FILES__________________
raster_folder = 'Documents/jsailor/knp_butterflies/cut_rasters'
raster_dir = os.fsencode(raster_folder)
raster_list = sorted(os.listdir(raster_dir))

#________________CREATE KNP DATA FOR PRED OFF RASTER FILES_________
for i in range(len(raster_used)):
    filename1 = os.fsdecode(raster_list[(raster_used[i]-1)])
    raster_file = os.path.join(raster_folder, filename1)
    raster_tif = gdal.Open(raster_file)
    data_array = raster_tif.GetRasterBand(1).ReadAsArray()
    mask = np.isnan(data_array)
    data_array[mask] = -1000
    if i == 0:
        raster_array = np.array(data_array.ravel())
    else:
        raster_array = np.c_[raster_array, data_array.ravel()]

#%%
#_________________LOOP THROUGH BUTTERFLIES______________
for i in range(len(dataname_list)):

    filename2 = os.fsdecode(dataname_list[i])
    data_file = os.path.join(Data_folder, filename2)
    data_ = pd.read_csv(data_file)

    species = results_list[i]

    #_______________DELETE B/C MULTICOLLINEARITY________________
    #based off valies in model_predictions
    data_ = data_.drop(raster_not_used_st, axis = 1)

    #_______________SPLIT DATA INTO TRAIN AND SPLIT_______________
    X = data_.iloc[:, 1:].values
    y = data_.iloc[:, 0].values
    #print(X.shape, y.shape)

    #______________STANDARDIZATION______________________________
    scalar = StandardScaler()
    Xs_train = scalar.fit_transform(X)
    raster_array_s = scalar.transform(raster_array)
    
    #______________LOOP THROUGH ML MODELS______________________________
    
    model_index = 0
    stacked_prediction = []

    for clas, name in zip(classifiers, names):
        print("starting: ", species, name)
        model = clone(clas)

        model.fit(Xs_train, y)

        prediction_results = model.predict_proba(raster_array_s)[:,1]
        prediction_results = prediction_results.reshape(data_array.shape).astype(float)
        prediction_results[mask] = np.nan

        stacked_prediction.append(prediction_results)

        save_file(model, f"{species}_{name}", "pkl")
        save_file(prediction_results, f"{species}_{name}_pred_results", "pkl")
        model_index += 1

        #________________PLOT PREDICTIONS__________________
        

    stacked_prediction = np.stack(stacked_prediction)
    save_file(stacked_prediction, f"{species}_stacked_results", "pkl")
    #expect a warning here
    ensemble_prediction = np.nanmean(stacked_prediction, axis = 0)
    save_file(ensemble_prediction, f"{species}_ensemble_results", "pkl")

print("Done")
