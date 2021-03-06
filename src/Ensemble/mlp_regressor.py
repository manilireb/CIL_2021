# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:02:53 2021

@author: florinl

Script for creating a submission file for kagge that performs regression using a Multi-Layer Perceptron on the best 6 base models.
"""



#imports

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV

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
from utilities.data_preprocess import Data
from utilities.data_preprocess_submission import Data_submission

from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

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
    mlp = MLPRegressor(max_iter = 500)
    
    # hyperparameter space
    parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }
    
    #grid search CV
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(pred_array, ground_truth)
    
    
    
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
    
    y_hat = clf.predict(pred_array_submission) #applying ridge regression

    #preparing submission
    pos = []
    pred = []
    for i in range(len(y_hat)):
        pos.append("r"+str(UID[i])+"_c"+str(IID[i]))
        pred.append(y_hat[i])
    pos.insert(0,'Id')
    pred.insert(0,'Prediction')
    
    with open('submission_mlp_reg.csv', 'w', newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(zip(pos, pred))
