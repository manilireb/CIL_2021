# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:02:53 2021

@author: florinl

Script that creates a submission file for Kaggle which does a linear combination of a regression using a Multi-Layer perceptron and a Ridge regression of all 
out base models.
"""




import numpy as np
import pandas as pd

import json
import csv

from KNN_Methods.KNN_Basics import KNN_Basic

import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split

from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from SlopeOne.slope_one import Slope_One
from Co_Clustering.Clustering_Coclustering import Clustering_Coclustering

from utilities.data_preprocess import Data
from utilities.data_preprocess_submission import Data_submission

from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

if __name__ == "__main__":

    df = Data().get_df()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    
    models = [
        KNN_WithMeans(sim_name="pearson_baseline", user_based=False),
        KNN_WithZScore(sim_name="pearson_baseline", user_based=False),
        MFNMF(biased=False),
        KNN_WithZScore(sim_name="pearson", user_based=False),
        KNN_WithMeans(sim_name="pearson", user_based=False),
        MFSVD(biased=True),
        KNN_WithMeans(sim_name="msd", user_based=False),
        KNN_WithZScore(sim_name="msd", user_based=False),
        KNN_WithZScore(sim_name="pearson_baseline", user_based=True),
        KNN_WithMeans(sim_name="pearson_baseline", user_based=True),
        KNN_WithMeans(sim_name="cosine", user_based=False),
        Slope_One(),
        KNN_WithZScore(sim_name="pearson", user_based=True),
        KNN_WithZScore(sim_name="cosine", user_based=False),
        KNN_WithMeans(sim_name="pearson", user_based=True),
        KNN_WithZScore(sim_name="msd", user_based=True),
        Clustering_Coclustering(),
        KNN_WithZScore(sim_name="cosine", user_based=True),
        KNN_WithMeans(sim_name="msd", user_based=True),
        KNN_WithMeans(sim_name="cosine", user_based=True),
        MFSVD(biased=False),
        KNN_Basic(sim_name="msd", user_based=True),
        MFNMF(biased=False),
        KNN_Basic(sim_name="pearson_baseline", user_based=True),
        KNN_Basic(sim_name="cosine", user_based=True),
        KNN_Basic(sim_name="pearson", user_based=True),
        KNN_Basic(sim_name="msd", user_based=False),
        KNN_Basic(sim_name="pearson_baseline", user_based=False),
        KNN_Basic(sim_name="pearson", user_based=False),
        KNN_Basic(sim_name="cosine", user_based=False),
    ]
    
    # getting prediction of models
    train, test = train_test_split(data, test_size =.3, random_state=42)
    predictions = []
    for model in models:
        print("fitting: ", model)
        m = model.get_opt_model()
        p = m.fit(train).test(test)
        test_pred = [pred[3] for pred in p]
        predictions.append(test_pred)
    pred_array = np.array(predictions).T
    ground_truth = np.array([pred[2] for pred in p])
    
    # Define NN Model
    mlp = MLPRegressor(max_iter = 500, random_state = 14)
    
    # hyperparameter space
    parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu' ,'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }
    
    
    #%% MLP regressor
    #grid search CV
    clf = GridSearchCV(mlp, parameter_space,scoring = "neg_mean_squared_error", n_jobs=-1, cv=3, verbose=2)
    clf.fit(pred_array, ground_truth)
    clf.best_params_
    
    #%% Ridge Regrssion
    # using CV to learn optimal alpha for Ridge regression default is leave one out CV
    alphas = np.logspace(-3, np.log10(10), 1000)
    regressor = RidgeCV(alphas=alphas, store_cv_values=True, cv=None)  # Default scoring MSE
    regressor.fit(pred_array, ground_truth)
    
    #%% Combining MLP and Ridge using linear regression
    y_hat_test_mlp = clf.predict(pred_array)
    y_hat_test_ridge = regressor.predict(pred_array)
    
    linear_regression = LinearRegression().fit(np.column_stack((y_hat_test_mlp,y_hat_test_ridge)), ground_truth)
    
    # save parameters
    with open('mlp_params_all.csv', 'w', newline = "") as f:
        for key in clf.best_params_.keys():
            f.write("%s,%s\n"%(key,clf.best_params_[key]))
    with open('ridge_alpha_all.csv', 'w', newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(['best alpha ' ,str(regressor.alpha_)])
    np.savetxt('ridge_coef_all.csv', regressor.coef_, delimiter=",") 
    np.savetxt('linear_regression_coef_all.csv', linear_regression.coef_, delimiter=",")
    
    
    
    
    # loading and preparing prediction set
    df_submission_test = Data("sampleSubmission.csv").get_df()
    data_submission_test = Dataset.load_from_df(df_submission_test[["userID", "itemID", "rating"]], reader)
    submission_full_train = data_submission_test.build_full_trainset()
    submission_full_testset = submission_full_train.build_testset()
    
    #full trainset
    full_train_set = data.build_full_trainset()
    
    #creating predictions
    predictions = []
    UID = []
    IID = []
    for model in models:
        print("fitting: ", model)
        m = model.get_opt_model()
        p = m.fit(full_train_set).test(submission_full_testset)
        test_pred = [pred[3] for pred in p]
        test_uid = [pred[0] for pred in p]
        test_iid = [pred[1] for pred in p]
        predictions.append(test_pred)
        UID.append(test_uid)
        IID.append(test_iid)
    pred_array_submission = np.array(predictions).T
    UID = np.array(UID).T
    IID = np.array(IID).T
    UID = UID[:,0]+1
    IID = IID[:,0]+1
    
    y_hat_mlp = clf.predict(pred_array_submission) #applying ridge regression
    y_hat_ridge = regressor.predict(pred_array_submission)  # applying ridge regression
    y_hat = linear_regression.predict(np.column_stack((y_hat_mlp,y_hat_ridge)))
    
    #preparing submission
    pos = []
    pred = []
    for i in range(len(y_hat)):
        pos.append("r"+str(UID[i])+"_c"+str(IID[i]))
        pred.append(y_hat[i])
    pos.insert(0,'Id')
    pred.insert(0,'Prediction')
    
    with open('submission_mlp_ridge_more_models_all.csv', 'w', newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(zip(pos, pred))
    
    # only mlp
    pos = []
    pred = []
    for i in range(len(y_hat_mlp)):
        pos.append("r"+str(UID[i])+"_c"+str(IID[i]))
        pred.append(y_hat_mlp[i])
    pos.insert(0,'Id')
    pred.insert(0,'Prediction')
    
    with open('submission_mlp_more_models_all.csv', 'w', newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(zip(pos, pred))
        
        
        
    # saves
    np.savetxt('ground_truth_all.csv', ground_truth, delimiter=",") 
    np.savetxt('pred_array_all.csv', pred_array, delimiter=",") 
    np.savetxt('pred_array_submission_all.csv', pred_array_submission, delimiter=",") 
