'''
Author: Jennifer Sailor
Date: 
Order in Sequence: 1

loop through ML models and save accuracy, percision, recall, auc-roc scores
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
    filename2 = f"Documents/jsailor/knp_butterflies/ML_files/model_predictions/{filename}.{filetype}"
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
'''
params = [
       #logReg
    {
        'max_iter': [100, 1000, 2500, 5000],
        'C': [0.1, 1, 10],
        'random_state': [0]
    }, #KNN
    {
        'n_neighbors': list(range(3,10)),
        'leaf_size': [1,5,10,15,20,25,30,35,50],
        'p': [1,2]
    }, #GPC
    {
        'n_restarts_optimizer': [0,1,2],
        'max_iter_predict': [100, 150, 200],
        'random_state': [0]
    }, #DT
    {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'random_state': [0]
    }, #RF
    {
        'bootstrap': [True, False],
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 5, 10, 15],
        'random_state': [0]
    }, #MLP
    {
        'momentum': [0.9, 0.8, 0.5],
        'alpha': [.0001, .01, 1],
        'max_iter': [200, 600, 1000, 1500],
        'random_state': [0]
    }, #Ada
    {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [1, 0.5, .1, 2],
        'random_state': [0]
    }, #Gaussian
    {
        'priors': [None]
    }, #QDA - no hyperparameters
    {
        'priors': [None]
    }

]
'''
#%%
#CHANGE WHEN WANT TO RUN ALL # len(dataname_list)
for i in range(len(dataname_list)):

    filename2 = os.fsdecode(dataname_list[i])
    data_file = os.path.join(Data_folder, filename2)
    data_ = pd.read_csv(data_file)

    species = results_list[i]

    #_______________TEST FOR MULTICOLLINEARITY________________
    #noticed that in loop all are the same so will just do once and then set the same for all
    if i == 0:
        VIF = pd.DataFrame()
        VIF["variable"] = data_.columns
        VIF['vif'] = [variance_inflation_factor(data_.values, i) for i in range(data_.shape[1])]

        threshold = 5
        dropped_col_names = []

        high_vifs_vars = VIF[VIF['vif'] > threshold]['variable']
        #print('length:' , len(high_vifs_vars))
        while (len(high_vifs_vars) > 0):
            highest_vif = VIF[VIF['vif'] == max(VIF['vif'])]['variable']
            data_ = data_.drop(highest_vif, axis = 1)
            dropped_col_names.append(str(int(highest_vif)))

            VIF = pd.DataFrame()
            VIF["variable"] = data_.columns
            VIF['vif'] = [variance_inflation_factor(data_.values, i) for i in range(data_.shape[1])]

            high_vifs_vars = VIF[VIF['vif'] > threshold]['variable']
            #print('length:' , len(high_vifs_vars))
    else:
        data_ = data_.drop(dropped_col_names, axis = 1)

    print("Columns after cleaning up multicollinearity: ", data_.columns)
    

    #_______________SPLIT DATA INTO TRAIN AND SPLIT_______________
    X = data_.iloc[:, 1:].values
    y = data_.iloc[:, 0].values

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #70% goes to train

    #print(X_train.shape, '%70', (X_train.shape[0]/X.shape[0])) 
    #print(X_test.shape, '%30') 

    #TO ADD: will want to save before standardizing
    save_file(X_train, f"{species}_x_train", "pkl")
    save_file(y_train, f"{species}_y_train", "pkl")
    save_file(X_train, f"{species}_x_test", "pkl")
    save_file(y_train, f"{species}_y_test", "pkl")
    #______________STANDARDIZATION______________________________
    scalar = StandardScaler()
    Xs_train = scalar.fit_transform(X_train)
    Xs_test = scalar.transform(X_test)
    '''
    print("Goal: 3rd line close to 0 and 4th line close to 1")
    
    print(np.vstack([
        Xs_train.mean(axis=0),
        Xs_train.std(axis=0)
    ]), 'TRAIN \n')

    print(np.vstack([
        Xs_test.mean(axis=0),
        Xs_test.std(axis=0)
    ]), 'TEST')
    '''
    #______________LOOP THROUGH ML MODELS______________________________
    #list to keep track all the areas under the curves to then average
    auc_list = []; precision_list = []; recall_list = []; accuracy_list = [];
    model_index = 0
    for clas, name in zip(classifiers, names):
        print("starting: ", species, name)
        model = clone(clas)

        model.fit(Xs_train, y_train)

        #deleted all hyperparamter tuning but it was based off AUC_ROC

        prob_test = model.predict_proba(Xs_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, prob_test)
        auc_roc = auc(fpr, tpr)

        auc_list.append(auc_roc)

        prediction_results = model.predict(Xs_test)
        precision_list.append(precision_score(y_test, prediction_results))
        recall_list.append(recall_score(y_test, prediction_results))
        accuracy_list.append(accuracy_score(y_test, prediction_results))
        save_file(model, f"{species}_{name}", "pkl")
        model_index += 1

        #________________PLOT AUC FIGURE__________________
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = {0:.2f})'.format(auc_roc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(f'{species} {name}')
        plt.savefig(f'Documents/jsailor/knp_butterflies/ROC_plots/{species}_{name}_ROC.png') #ADD variable that says what species we on
        plt.close()


    #ADD save the auc list
    results_df = pd.DataFrame({'Names': names, 'AUC': auc_list, 'Accuracy': accuracy_list, 'Precision': precision_list, 'Recall': recall_list})
    save_file(results_df, f"{species}_results", "csv")
    print(results_df)

print("Done")
