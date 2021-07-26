#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:59:17 2021

@author: manuel
"""

import json
import time

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
        self.tuning_params = {
            "learning_rate": [0.005, 0.85],
            "n_estimators": [80, 400],
            "max_depth": [2, 7],
            "max_features": [8, 17],
        }

    def get_rmse(self, y_pred, y_true):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def optimizer_function(self, learning_rate, n_estimators, max_depth, max_features):
        cv = 3
        kf = KFold(n_splits=cv, random_state=self.random_state, shuffle=True)
        test_RMSE_list = []
        print("parameters:", learning_rate, str(int(n_estimators)), str(int(max_depth)), str(int(max_features)))
        for train, test in kf.split(self.X):
            reg = GradientBoostingRegressor(
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                random_state=self.random_state,
                max_features=int(max_features),
            )
            reg.fit(self.X[train], self.y[train])
            y_pred = reg.predict(self.X[test])
            print("Computation done")
            test_rmse = self.get_rmse(self.y[test], y_pred)
            test_RMSE_list.append(test_rmse)
        mean = np.mean(test_RMSE_list)
        return -mean

    def log_hyperparams_to_json(self):
        optimizer = BayesianOptimization(
            f=self.optimizer_function, pbounds=self.tuning_params, random_state=self.random_state
        )
        path = get_git_root() + "/logs/GBE.json"
        logger = JSONLogger(path=path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=4, n_iter=10)

    def get_opt_hyperparams(self):
        file_name = get_git_root() + "/logs/GBE.json"
        with open(file_name) as handle:
            json_data = [json.loads(line) for line in handle]
        rmse = []
        for dic in json_data:
            rmse.append(dic.get("target"))
        index = np.argmax(rmse)
        return json_data[index].get("params")

    def get_opt_model(self):
        opt_hyperparams = self.get_opt_hyperparams()
        opt_hyperparams["n_estimators"] = int(opt_hyperparams["n_estimators"])
        opt_hyperparams["max_depth"] = int(opt_hyperparams["max_depth"])
        opt_hyperparams["max_features"] = int(opt_hyperparams["max_features"])
        algo = GradientBoostingRegressor(**opt_hyperparams, random_state=self.random_state)
        return algo


if __name__ == "__main__":
    GBE = GBE()
    GBE.log_hyperparams_to_json()
