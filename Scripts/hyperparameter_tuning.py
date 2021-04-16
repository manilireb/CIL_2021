#imports

import numpy as np
import pandas as pd

#surprise package imports
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import NMF
from surprise import KNNWithZScore
from surprise import CoClustering

#bayes_opt imports
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

import json
import csv

from helpers_surprise import*

#run

#read the data
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file('ratings.csv', reader=reader)

# In the following we tune the hyperparameters through bayesian optimization for all the methods that we use from the surprise package.
# ### KNN user based

# #### KNN basic

# ##### Cosine

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,40],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicUserCosin(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'cosine',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs':int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicUserCosin = BayesianOptimization(
  f = BO_KNNBasicUserCosin,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicUserCosin.json")
optimizerKNNBasicUserCosin.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicUserCosin.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### msd

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,40],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicUserMSD(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'msd',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicUserMSD= BayesianOptimization(
  f = BO_KNNBasicUserMSD,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicUserMSD.json")
optimizerKNNBasicUserMSD.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicUserMSD.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,40],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicUserPearson(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs':int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicUserPearson= BayesianOptimization(
  f = BO_KNNBasicUserPearson,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicUserPearson.json")
optimizerKNNBasicUserPearson.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicUserPearson.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson baseline

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'shrinkage': [0,200],
    'n_epochs': [5,40],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicUserPearsonBaseline(k,min_k,min_support,shrinkage,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson_baseline',
                   'min_support':int(min_support),
                   'shrinkage': shrinkage,
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicUserPearsonBaseline= BayesianOptimization(
  f = BO_KNNBasicUserPearsonBaseline,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicUserPearsonBaseline.json")
optimizerKNNBasicUserPearsonBaseline.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicUserPearsonBaseline.maximize(
  init_points = 3,
  n_iter = 10
 )



# #### KNNWithMeans

# ##### Cosin

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,40],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansUserCosin(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'cosine',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansUserCosin = BayesianOptimization(
  f = BO_KNNWithMeansUserCosin,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansUserCosin.json")
optimizerKNNWithMeansUserCosin.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansUserCosin.maximize(
  init_points = 3,
  n_iter = 10
 )


# ##### msd

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansUserMSD(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'msd',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansUserMSD= BayesianOptimization(
  f = BO_KNNWithMeansUserMSD,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansUserMSD.json")
optimizerKNNWithMeansUserMSD.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansUserMSD.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansUserPearson(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansUserPearson= BayesianOptimization(
  f = BO_KNNWithMeansUserPearson,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansUserPearson.json")
optimizerKNNWithMeansUserPearson.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansUserPearson.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson baseline

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'shrinkage': [0,200],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansUserPearsonBaseline(k,min_k,min_support,shrinkage,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson_baseline',
                   'min_support':int(min_support),
                   'shrinkage': shrinkage,
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansUserPearsonBaseline= BayesianOptimization(
  f = BO_KNNWithMeansUserPearsonBaseline,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansUserPearsonBaseline.json")
optimizerKNNWithMeansUserPearsonBaseline.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansUserPearsonBaseline.maximize(
  init_points = 3,
  n_iter = 5
 )


# #### KNNWithZScore

# ##### Cosin

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreUserCosin(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'cosine',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreUserCosin = BayesianOptimization(
  f = BO_KNNWithZScoreUserCosin,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreUserCosin.json")
optimizerKNNWithZScoreUserCosin.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreUserCosin.maximize(
  init_points = 3,
  n_iter = 10
 )


# ##### msd

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreUserMSD(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'msd',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreUserMSD= BayesianOptimization(
  f = BO_KNNWithZScoreUserMSD,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreUserMSD.json")
optimizerKNNWithZScoreUserMSD.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreUserMSD.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreUserPearson(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson',
                   'min_support':int(min_support),
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo =KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreUserPearson= BayesianOptimization(
  f = BO_KNNWithZScoreUserPearson,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreUserPearson.json")
optimizerKNNWithZScoreUserPearson.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreUserPearson.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson baseline

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'shrinkage': [0,200],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreUserPearsonBaseline(k,min_k,min_support,shrinkage,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson_baseline',
                   'min_support':int(min_support),
                   'shrinkage': shrinkage,
                   'user_based': True
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreUserPearsonBaseline= BayesianOptimization(
  f = BO_KNNWithZScoreUserPearsonBaseline,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreUserPearsonBaseline.json")
optimizerKNNWithZScoreUserPearsonBaseline.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreUserPearsonBaseline.maximize(
  init_points = 3,
  n_iter = 5
 )


# ### KNN item based

# #### KNN basic

# ##### Cosin

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicItemCosin(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'cosine',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicItemCosin = BayesianOptimization(
  f = BO_KNNBasicItemCosin,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicItemCosin.json")
optimizerKNNBasicItemCosin.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicItemCosin.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### msd

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicItemMSD(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'msd',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicItemMSD= BayesianOptimization(
  f = BO_KNNBasicItemMSD,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicItemMSD.json")
optimizerKNNBasicItemMSD.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicItemMSD.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicItemPearson(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicItemPearson= BayesianOptimization(
  f = BO_KNNBasicItemPearson,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicItemPearson.json")
optimizerKNNBasicItemPearson.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicItemPearson.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson baseline

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'shrinkage': [0,200],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNBasicItemPearsonBaseline(k,min_k,min_support,shrinkage,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson_baseline',
                   'min_support':int(min_support),
                   'shrinkage': shrinkage,
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNBasic(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNBasicItemPearsonBaseline= BayesianOptimization(
  f = BO_KNNBasicItemPearsonBaseline,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNBasicItemPearsonBaseline.json")
optimizerKNNBasicItemPearsonBaseline.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNBasicItemPearsonBaseline.maximize(
  init_points = 3,
  n_iter = 5
 )


# #### KNNWithMeans

# ##### Cosin

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansItemCosin(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'cosine',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansItemCosin = BayesianOptimization(
  f = BO_KNNWithMeansItemCosin,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansItemCosin.json")
optimizerKNNWithMeansItemCosin.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansItemCosin.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### msd


tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansItemMSD(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'msd',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansItemMSD= BayesianOptimization(
  f = BO_KNNWithMeansItemMSD,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansItemMSD.json")
optimizerKNNWithMeansItemMSD.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansItemMSD.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansItemPearson(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansItemPearson= BayesianOptimization(
  f = BO_KNNWithMeansItemPearson,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansItemPearson.json")
optimizerKNNWithMeansItemPearson.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansItemPearson.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson baseline

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'shrinkage': [0,200],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithMeansItemPearsonBaseline(k,min_k,min_support,shrinkage,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson_baseline',
                   'min_support':int(min_support),
                   'shrinkage': shrinkage,
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithMeans(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithMeansItemPearsonBaseline= BayesianOptimization(
  f = BO_KNNWithMeansItemPearsonBaseline,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithMeansItemPearsonBaseline.json")
optimizerKNNWithMeansItemPearsonBaseline.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithMeansItemPearsonBaseline.maximize(
  init_points = 3,
  n_iter = 5
 )


# #### KNNWithZScore

# ##### Cosin


tuning_params = dict()
tuning_params = { 
   'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreItemCosin(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'cosine',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreItemCosin = BayesianOptimization(
  f = BO_KNNWithZScoreItemCosin,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreItemCosin.json")
optimizerKNNWithZScoreItemCosin.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreItemCosin.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### msd

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreItemMSD(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'msd',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreItemMSD= BayesianOptimization(
  f = BO_KNNWithZScoreItemMSD,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreItemMSD.json")
optimizerKNNWithZScoreItemMSD.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreItemMSD.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreItemPearson(k,min_k,min_support,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson',
                   'min_support':int(min_support),
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo =KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options, bsl_options=bsl_options,verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreItemPearson= BayesianOptimization(
  f = BO_KNNWithZScoreItemPearson,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreItemPearson.json")
optimizerKNNWithZScoreItemPearson.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreItemPearson.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### pearson baseline

tuning_params = dict()
tuning_params = { 
    'k': [5,800],
    'min_k':[1,20],
    'min_support': [1,50],
    'shrinkage': [0,200],
    'n_epochs': [5,15],
    'reg_u': [1,30],
    'reg_i': [1,30]
 }

def BO_KNNWithZScoreItemPearsonBaseline(k,min_k,min_support,shrinkage,n_epochs,reg_u,reg_i):
    sim_options = {'name': 'pearson_baseline',
                   'min_support':int(min_support),
                   'shrinkage': shrinkage,
                   'user_based': False
               }
    bsl_options = {'method': 'als',
               'n_epochs': int(n_epochs),
               'reg_u': reg_u,
               'reg_i': reg_i
               }
    algo = KNNWithZScore(k=int(k), min_k=int(min_k), sim_options=sim_options,bsl_options=bsl_options, verbose=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerKNNWithZScoreItemPearsonBaseline= BayesianOptimization(
  f = BO_KNNWithZScoreItemPearsonBaseline,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsKNNWithZScoreItemPearsonBaseline.json")
optimizerKNNWithZScoreItemPearsonBaseline.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerKNNWithZScoreItemPearsonBaseline.maximize(
  init_points = 3,
  n_iter = 5
 )


# ### SVD

# ##### Biased

tuning_params = dict()

tuning_params = { 
    'n_factors': [5,150],
    "lr_bu": [0.001, 0.009],
    "lr_bi": [0.001, 0.009],
    'lr_pu':  [0.001, 0.009],
    'n_epochs': [2,200],
    'reg_qi' : [0.01,0.9],
    'reg_bu' : [0.01,0.9],
    'reg_bi' : [0.01,0.9],
    'reg_pu' : [0.01,0.9]
 }

def BO_SVDBiased(n_factors,lr_pu,lr_bu,lr_bi,reg_qi,reg_bu,reg_bi,reg_pu,n_epochs):
    algo = SVD(n_factors=int(n_factors),lr_pu=lr_pu,lr_bu=lr_bu,lr_bi=lr_bi,reg_qi=reg_qi,
               reg_bu=reg_bu,reg_bi=reg_bi,reg_pu=reg_pu,n_epochs=int(n_epochs),biased=True)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerSVDBiased = BayesianOptimization(
  f = BO_SVDBiased,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsSVDBiased.json")
optimizerSVDBiased.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerSVDBiased.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### Unbiased

tuning_params = dict()

tuning_params = { 
    'n_factors': [5,150],
    "lr_bu": [0.001, 0.009],
    'lr_pu':  [0.001, 0.009],
    'n_epochs': [2,200],
    'reg_qi' : [0.01,0.9],
    'reg_bu' : [0.01,0.9],
    'reg_pu' : [0.01,0.9]
 }

def BO_SVDUnBiased(n_factors,lr_pu,lr_bu,reg_qi,reg_bu,reg_pu,n_epochs):
    algo = SVD(n_factors=int(n_factors),lr_pu=lr_pu,lr_bu=lr_bu,reg_qi=reg_qi,
               reg_bu=reg_bu,reg_pu=reg_pu,n_epochs=int(n_epochs),biased=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerSVDUnBiased = BayesianOptimization(
  f = BO_SVDUnBiased,
  pbounds = tuning_params,
  verbose = 2,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsSVDUnBiased.json")
optimizerSVDUnBiased.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerSVDUnBiased.maximize(
  init_points = 3,
  n_iter = 5
 )


# ### Non negative Matrix Factorization

# ##### Biased

tuning_params = dict()

tuning_params = { 
    'n_factors': [5,150],
    "lr_bu": [0.001, 0.009],
    "lr_bi": [0.001, 0.009],
    'n_epochs': [25,400],
    'reg_qi' : [0.01,0.9],
    'reg_bu' : [0.01,0.9],
    'reg_bi' : [0.01,0.9]
 }

def BO_NMFBiased(n_factors, lr_bu,lr_bi,n_epochs,reg_qi,reg_bu,reg_bi):
    algo = NMF(n_factors=int(n_factors),lr_bu=lr_bu,lr_bi=lr_bi,reg_qi=reg_qi,
               reg_bu=reg_bu,reg_bi=reg_bi,n_epochs=int(n_epochs),biased=True)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerNMFBiased = BayesianOptimization(
  f = BO_NMFBiased,
  pbounds = tuning_params,
  verbose = 10,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsNMFBiased.json")
optimizerNMFBiased.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerNMFBiased.maximize(
  init_points = 3,
  n_iter = 5
 )


# ##### Unbiased



tuning_params = dict()

tuning_params = { 
    'n_factors': [5,150],
    "lr_bu": [0.001, 0.009],
    'n_epochs': [25,400],
    'reg_qi' : [0.01,0.9],
    'reg_bu' : [0.01,0.9]
 }

def BO_NMFUnBiased(n_factors, lr_bu,n_epochs,reg_qi,reg_bu):
    algo = NMF(n_factors=int(n_factors),lr_bu=lr_bu,reg_qi=reg_qi,
               reg_bu=reg_bu,n_epochs=int(n_epochs),biased=False)
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))


optimizerNMFUnBiased = BayesianOptimization(
  f = BO_NMFUnBiased,
  pbounds = tuning_params,
  verbose = 10,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsNMFUnBiased.json")
optimizerNMFUnBiased.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerNMFUnBiased.maximize(
  init_points = 3,
  n_iter = 5
 )


# ### CoClustering


tuning_params = dict()

tuning_params = { 
    'n_epochs': [2,200],
    'n_cltr_u' : [1,100],
    'n_cltr_i' : [1,100]
 }

def BO_CoCl(n_epochs, n_cltr_u,n_cltr_i):
    algo = CoClustering(n_epochs=int(n_epochs),n_cltr_u=int(n_cltr_u),n_cltr_i=int(n_cltr_i))
    cv_res=cross_validate(algo, data, measures=[u'rmse'], cv=3, return_train_measures=False, n_jobs=-1, pre_dispatch=u'2*n_jobs', verbose=1)


    return -np.mean(cv_res.get('test_rmse'))

optimizerCoCl = BayesianOptimization(
  f = BO_CoCl,
  pbounds = tuning_params,
  verbose = 10,
  random_state = 5, 
 )

logger = JSONLogger(path="./outputs/logsCoCl.json")
optimizerCoCl.subscribe(Events.OPTMIZATION_STEP, logger)

optimizerCoCl.maximize(
  init_points = 3,
  n_iter = 5
 )

