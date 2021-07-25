#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:59:17 2021

@author: manuel
"""

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from utilities.data_preprocess import get_git_root


class GBE:
    def __init__(self):
        self.X = np.array(pd.read_csv("Ensemble_features.csv"))
        self.y = np.array(pd.read_csv("Ensemble_targets.csv")).squeeze()
        self.random_state = 42
        self.tuning_params = {"learning_rate": [0.005, 0.85], "n_estimators": [80, 850], "max_depth": [2, 20]}

    def get_rmse(self, y_pred, y_true):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def optimizer_function(self, learning_rate, n_estimators, max_depth):
        cv = 5
        kf = KFold(n_splits=cv, random_state=self.random_state, shuffle=True)
        test_RMSE_list = []
        for X_train, X_test, y_train, y_test in kf.split(self.X, self.y):
            reg = GradientBoostingRegressor(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
            )
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            test_rmse = self.get_rmse(y_test, y_pred)
            test_RMSE_list.append(test_rmse)
        mean = np.mean(test_RMSE_list)
        return -mean

    def log_hyperparams_to_json(self):
        optimizer = BayesianOptimization(
            f=self.optimzier_function, pbounds=self.tuning_params, random_state=self.random_state
        )
        path = get_git_root() + "/logs/GBE.json"
        logger = JSONLogger(path=path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=7, n_iter=20)


if __name__ == "__main__":
    GBE = GBE()
    GBE.log_hyperparams_to_json()
