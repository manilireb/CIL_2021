#imports

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import json
import csv


print('Loading model predictions...')
X=pd.read_csv('Scripts/outputs/Ensemble_X.csv.gzip',compression='gzip')
X_test=pd.read_csv('Scripts/outputs/Ensemble_X_sub.csv.gzip',compression='gzip')
y=np.load('Scripts/outputs/Ensemble_y.npy',allow_pickle='TRUE')


names=['KNNWithZScoreItemPearsonBaseline','KNNWithMeansItemPearsonBaseline','KNNWithZScoreUserPearsonBaseline','KNNWithMeansUserPearsonBaseline','NMFUnBiased','SVDUnBiased','CoCl']

print('Tuning and training Ridge regression...')
#tuning and fitting Ridge regression on the trainset
alphas=np.logspace(-3,np.log10(10),1000) #we use logspace so we can start close to 0 
regressor = RidgeCV(alphas=alphas, store_cv_values=True, cv=None) #if we have cv=none default scoring is MSE
regressor.fit(X[names], y)
print('The optimal lambda is:', regressor.alpha_)
print('Predicting on trainset...')
#Using trained model to predict on testset
y_hat=regressor.predict(X_test[names])

print('Creating submission csv...')
#create csv that can be uploaded to aicrowd
df_submission=pd.DataFrame(data=np.transpose([X_test['uid'],X_test['iid'],y_hat]), columns=['uid','iid','prediction'])
pos=[]
prediction=[]
for i in range(0,len(y_hat)):
    pos.append("r"+str((int(float(df_submission.loc[i,'uid']))))+"_c"+str(int(float(df_submission.loc[i,'iid']))))
    prediction.append(round(df_submission.loc[i,'prediction']))
pos.insert(0,'Id')
prediction.insert(0,'Prediction')
with open('submission/submission_ridge.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(pos, prediction))
print('Submission csv created!')
