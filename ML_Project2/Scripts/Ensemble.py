#imports

import numpy as np
import pandas as pd

#surprise package imports
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader

from surprise import Dataset
from surprise.model_selection import train_test_split

from surprise import SVD
from surprise import NMF
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import CoClustering

from sklearn.linear_model import RidgeCV
from helpers_surprise import*

import json
import csv





#run 

#loading the trainset
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file('outputs/trainset.csv', reader=reader)

#loading tuned hyperparameters
op_KNNWithZScoreItemPearsonBaseline=get_opt_parameters('./outputs/logsKNNWithZScoreItemPearsonBaseline.json')
op_KNNWithMeansItemPearsonBaseline=get_opt_parameters('./outputs/logsKNNWithMeansItemPearsonBaseline.json')
op_KNNWithZScoreUserPearsonBaseline=get_opt_parameters('./outputs/logsKNNWithZScoreUserPearsonBaseline.json')
op_KNNWithMeansUserPearsonBaseline=get_opt_parameters('./outputs/logsKNNWithMeansUserPearsonBaseline.json')
op_NMFUnBiased=get_opt_parameters('./outputs/logsNMFUnBiased.json')
op_SVDUnBiased=get_opt_parameters('./outputs/logsSVDUnBiased.json')
op_CoCl=get_opt_parameters("./outputs/logsCoCl.json")

#creating models with tuned hyperparameters
KNNWithZScoreItemPearsonBaseline=KNNWithZScore(k=int(op_KNNWithZScoreItemPearsonBaseline.get('k')), min_k=int(op_KNNWithZScoreItemPearsonBaseline.get('min_k')), sim_options={'name': 'pearson_baseline',
                   'min_support':int(op_KNNWithZScoreItemPearsonBaseline.get('min_support')),
                   'shrinkage':op_KNNWithZScoreItemPearsonBaseline.get('shrinkage'),
                   'user_based': False
               },bsl_options={'method': 'als',
               'n_epochs': int(op_KNNWithZScoreItemPearsonBaseline.get('n_epochs')),
               'reg_u': op_KNNWithZScoreItemPearsonBaseline.get('reg_u'),
               'reg_i': op_KNNWithZScoreItemPearsonBaseline.get('reg_i')
               }, verbose=False)

KNNWithMeansItemPearsonBaseline=KNNWithMeans(k=int(op_KNNWithMeansItemPearsonBaseline.get('k')), min_k=int(op_KNNWithMeansItemPearsonBaseline.get('min_k')), sim_options={'name': 'pearson_baseline',
                   'min_support':int(op_KNNWithMeansItemPearsonBaseline.get('min_support')),
                   'shrinkage':op_KNNWithMeansItemPearsonBaseline.get('shrinkage'),
                   'user_based': False
               },bsl_options={'method': 'als',
               'n_epochs': int(op_KNNWithMeansItemPearsonBaseline.get('n_epochs')),
               'reg_u': op_KNNWithMeansItemPearsonBaseline.get('reg_u'),
               'reg_i': op_KNNWithMeansItemPearsonBaseline.get('reg_i')
               }, verbose=False)

KNNWithZScoreUserPearsonBaseline=KNNWithZScore(k=int(op_KNNWithZScoreUserPearsonBaseline.get('k')), min_k=int(op_KNNWithZScoreUserPearsonBaseline.get('min_k')), sim_options={'name': 'pearson_baseline',
                   'min_support':int(op_KNNWithZScoreUserPearsonBaseline.get('min_support')),
                   'shrinkage':op_KNNWithZScoreUserPearsonBaseline.get('shrinkage'),
                   'user_based': True
               },bsl_options={'method': 'als',
               'n_epochs': int(op_KNNWithZScoreUserPearsonBaseline.get('n_epochs')),
               'reg_u': op_KNNWithZScoreUserPearsonBaseline.get('reg_u'),
               'reg_i': op_KNNWithZScoreUserPearsonBaseline.get('reg_i')
               }, verbose=False)

KNNWithMeansUserPearsonBaseline=KNNWithMeans(k=int(op_KNNWithMeansUserPearsonBaseline.get('k')), min_k=int(op_KNNWithMeansUserPearsonBaseline.get('min_k')), sim_options={'name': 'pearson_baseline',
                   'min_support':int(op_KNNWithMeansUserPearsonBaseline.get('min_support')),
                   'shrinkage':op_KNNWithMeansUserPearsonBaseline.get('shrinkage'),
                   'user_based': True
               },bsl_options={'method': 'als',
               'n_epochs': int(op_KNNWithMeansUserPearsonBaseline.get('n_epochs')),
               'reg_u': op_KNNWithMeansUserPearsonBaseline.get('reg_u'),
               'reg_i': op_KNNWithMeansUserPearsonBaseline.get('reg_i')
               }, verbose=False)

NMFUnBiased=NMF(n_factors=int(op_NMFUnBiased.get('n_factors')),lr_bu=op_NMFUnBiased.get('lr_bu'),reg_qi=op_NMFUnBiased.get('reg_qi'),reg_bu=op_NMFUnBiased.get('reg_bu'),n_epochs=int(op_NMFUnBiased.get('n_epochs')),biased=False)

SVDUnBiased=SVD(n_factors=int(op_SVDUnBiased.get('n_factors')),lr_pu=op_SVDUnBiased.get('lr_pu'),lr_bu=op_SVDUnBiased.get('lr_bu'),reg_qi=op_SVDUnBiased.get('reg_qi'),reg_bu=op_SVDUnBiased.get('reg_bu'),reg_pu=op_SVDUnBiased.get('reg_pu'),n_epochs=int(op_SVDUnBiased.get('n_epochs')),biased=False)

CoCl=CoClustering(n_epochs=int(op_CoCl.get('n_epochs')),n_cltr_u=int(op_CoCl.get('n_cltr_u')),n_cltr_i=int(op_CoCl.get('n_cltr_i')))

algos=[KNNWithZScoreItemPearsonBaseline,KNNWithMeansItemPearsonBaseline,KNNWithZScoreUserPearsonBaseline,KNNWithMeansUserPearsonBaseline,NMFUnBiased,SVDUnBiased,CoCl]
names=['KNNWithZScoreItemPearsonBaseline','KNNWithMeansItemPearsonBaseline','KNNWithZScoreUserPearsonBaseline','KNNWithMeansUserPearsonBaseline','NMFUnBiased','SVDUnBiased','CoCl']


#tuning and fitting Ridge regression on the trainset
R_prime, R_star = train_test_split(data, test_size=.3)
X,y=get_X_y(algos,names,R_prime,R_star)
alphas=np.logspace(-3,np.log10(10),1000) #we use logspace so we can start close to 0 
regressor = RidgeCV(alphas=alphas, store_cv_values=True, cv=None) #if we have cv=none default scoring is MSE
regressor.fit(X[names], y)

#preparing the testset and trainset 
full_trainset=data.build_full_trainset()
submission_test = Dataset.load_from_file('outputs/testset.csv', reader=reader)
submission_full_train=submission_test.build_full_trainset()
submission_full_testset=submission_full_train.build_testset()

#saving inputs to the Ridgeregression

X.to_csv('outputs/Ensemble_X.csv.gzip',compression='gzip')
np.save('outputs/Ensemble_y.npy',y,allow_pickle='TRUE')

#Using trained model to predict on testset
X_sub,y_sub=get_X_y(algos,names,full_trainset,submission_full_testset)
y_hat=regressor.predict(X_sub[names])
X_sub.to_csv('outputs/Ensemble_X_sub.csv.gzip',compression='gzip')
#create csv that can be uploaded to aicrowd
df_submission=pd.DataFrame(data=np.transpose([X_sub['uid'],X_sub['iid'],y_hat]), columns=['uid','iid','prediction'])
pos=[]
prediction=[]
for i in range(0,len(y_hat)):
    pos.append("r"+str((int(float(df_submission.loc[i,'uid']))))+"_c"+str(int(float(df_submission.loc[i,'iid']))))
    prediction.append(round(df_submission.loc[i,'prediction']))
pos.insert(0,'Id')
prediction.insert(0,'Prediction')
with open('testset_predictions/submission_ridge.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(pos, prediction))








