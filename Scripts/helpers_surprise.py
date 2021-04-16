#imports
import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader

import json
import csv




#functions

def read_txt_surprise(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def preprocess_data_surprise(data):
    """preprocessing the text data, conversion to uid,iid,rating format."""
    def deal_line_surprise(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)
    # parse each line
    data = [deal_line_surprise(line) for line in data]
    return data

def load_data_surprise(path_dataset):
    """Load data in text format, one rating per line."""
    data = read_txt_surprise(path_dataset)[1:]
    return preprocess_data_surprise(data)

def create_submission(algo,data,algo_name):
    """This functions creates a csv file in the format required for an aicrowd submission, it needs the trainset, algorithm with hyperparameters and the algorithm name as string as input. To run properly the testset on which to predict needs to be saved under 'outputs/testset.csv', it saves the csv file to testset_prediction/submission_algo_name.csv"""
    full_trainset=data.build_full_trainset()
    algo.fit(full_trainset)
    submission_test = Dataset.load_from_file('outputs/testset.csv', reader=reader)
    submission_full_train=submission_test.build_full_trainset()
    best_prediction=algo.test(submission_full_train.build_testset())
    pos=[]
    prediction=[]
    for uid,iid,r_ui,est,_ in best_prediction:
        pos.append("r"+str((int(float(uid))))+"_c"+str(int(float(iid))))
        prediction.append(round(est))
    pos.insert(0,'Id')
    prediction.insert(0,'Prediction')
    with open('testset_predictions/submission_'+algo_name+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(pos, prediction))

def create_pd_df(predictions):
    '''Creates a pandas DataFrame with columns uid, iid and predictions. Input is a surprise predictions list'''
    UID=[]
    IID=[]
    prediction=[]
    for uid,iid,r_ui,est,_ in predictions:
        UID.append(uid)
        IID.append(iid)
        prediction.append(est)
    return  pd.DataFrame(np.transpose([UID,IID,prediction]),columns=['uid','iid','prediction'])

def get_real_rating(predictions):
    '''gets the real ratings from surprise predictions, this function is used to 
    calculate the rmse when the ensemble method is used '''
    real=[]
    for uid,iid,r_ui,est,_ in predictions:
        real.append(r_ui)
    return real

def get_lowest_rmse(path_to_json):
    '''This function reads the json file, that is created by saving the logs of the bayesian optimization provided by the bayesian-optimization package, and returns the lowest rmse'''
    with open(path_to_json, 'r') as handle:
        json_data = [json.loads(line) for line in handle]
    rmse=[]
    for d in json_data:
        rmse.append(d.get('target'))
    return np.abs(np.max(rmse))

def get_opt_parameters(json_file_name):
    '''This function reads the json file, that is created by saving the logs of the bayesian optimization provided by the bayesian-optimization package, and returns the optimal hyperparametrs'''
    with open(json_file_name, 'r') as handle:
        json_data = [json.loads(line) for line in handle]
    rmse=[]
    for d in json_data:
        rmse.append(d.get('target'))
    index=np.where(rmse==np.max(rmse))[0][0]
    return json_data[index].get('params')

def get_X_y(algos, names,trainset,testset):
    '''Returns a pandas dataframe with the predictions of each model on the test set as columns and an numpy array containing the real ratings of the testset. Inputs are the a list of surprise algorithms with the tuned hyperparameters, their names as a list and a surprise train and testset. '''
    print('Fitting and predicting for algorithm 1...')
    prediction_0=algos[0].fit(trainset).test(testset)
    predictions_df=create_pd_df(prediction_0)
    predictions_df.columns=['uid','iid',names[0]]
    for i in range(1,len(algos)):
        print('Fitting and predicting for algorithm '+str(i+1)+'...')
        prediction_temp=create_pd_df(algos[i].fit(trainset).test(testset))
        predictions_df[names[i]]=prediction_temp['prediction']
    print('gathered all predictions, getting real ratings')
    y=get_real_rating(prediction_0)
    return predictions_df ,y
