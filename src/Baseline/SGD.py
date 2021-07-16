#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Reber
"""

import numpy as np
import scipy.sparse as sp
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from tqdm import tqdm

from utilities.data_preprocess import Data, get_git_root


class SGD(BaseEstimator, TransformerMixin):
    """
    Class for the SGD baseline algorithm

    """

    def __init__(self):
        self.data_matrix = Data().get_sparse_ratings_matrix()
        self.nnz_users, self.nnz_items = self.data_matrix.nonzero()
        self.indices = np.arange(self.nnz_users.shape[0])
        self.num_users = 10000
        self.num_items = 1000
        self.num_epochs = 30
        self.random_state = 42
        self.user_matrix = None
        self.item_matrix = None
        self.tuning_params = {
            "num_features": [2, 50],
            "lambda_user": [0.0005, 0.9],
            "lambda_item": [0.0005, 0.9],
            "gamma": [0.0005, 0.03],
        }
        np.random.seed(self.random_state)

    def init_matrix(self, train, num_features):
        """
        This method is used for initializing user_matrix and item_matrix

        Parameters
        ----------
        train : sparse.lil.lil_matrix
            Train data.
        num_features : int
            Number of features.

        Returns
        -------
        user_matrix : np.ndarray
            Initialized user matrix.
        item_matrix : np.ndarray
            Initialized item matrix.

        """
        user_matrix = np.random.rand(self.num_users, num_features)
        item_matrix = np.random.rand(num_features, self.num_items)
        item_nnz = train.getnnz(axis=0)
        item_sum = train.sum(axis=0)
        item_matrix[0, :] = item_sum / item_nnz
        return user_matrix, item_matrix

    def get_train_test_matrix(self, train_indices, test_indices):
        """
        Method for creating a sparse train and test matrix

        Parameters
        ----------
        train_indices : np.ndarray
            Indices of the train data.
        test_indices : np.ndarray
            Indices of the test data.

        Returns
        -------
        train : sparse.lil.lil_matrix
            Sparse train data matrix.
        test : sparse.lil.lil_matrix
            Sparse test data matrix.

        """
        train_nnz_items = self.nnz_items[train_indices]
        train_nnz_users = self.nnz_users[train_indices]
        train = sp.lil_matrix((self.num_users, self.num_items))
        train[train_nnz_users, train_nnz_items] = self.data_matrix[train_nnz_users, train_nnz_items]
        test_nnz_items = self.nnz_items[test_indices]
        test_nnz_users = self.nnz_users[test_indices]
        test = sp.lil_matrix((self.num_users, self.num_items))
        test[test_nnz_users, test_nnz_items] = self.data_matrix[test_nnz_users, test_nnz_items]
        return train, test

    def fit(self, data, num_features, lambda_user, lambda_item, gamma):
        """
        Function that fits the user and the item matrix.

        Parameters
        ----------
        data : sparse.lil.lil_matrix
            Data matrix.
        num_features : int
            Number of features.
        lambda_user : float
            Regularization parameter.
        lambda_item : float
            Regularization parameter.
        gamma : float
            Regularization parameter.

        Returns
        -------
        None.

        """
        user_matrix, item_matrix = self.init_matrix(data, num_features)
        nnz_users, nnz_items = data.nonzero()
        nnz_data = list(zip(nnz_users, nnz_items))
        for it in tqdm(range(self.num_epochs)):
            np.random.shuffle(nnz_data)
            for u, i in nnz_data:
                user = user_matrix[u, :]
                item = item_matrix[:, i]
                err = data[u, i] - user @ item
                user_matrix[u, :] += gamma * (err * item - lambda_user * user)
                item_matrix[:, i] += gamma * (err * user - lambda_item * item)

        self.user_matrix = user_matrix
        self.item_matrix = item_matrix

    def get_test_rmse(self, test):
        """
        Return rmse of the test data

        Parameters
        ----------
        test : sparse.lil.lil_matrix
            Test data.

        Returns
        -------
        rmse : float
            RMSE for the test data.

        """
        nnz_user, nnz_item = test.nonzero()
        nnz_test = list(zip(nnz_user, nnz_item))
        rmse = 0.0
        for u, i in nnz_test:
            user = self.user_matrix[u, :]
            item = self.item_matrix[:, i]
            pred = user @ item
            if pred > 5:
                pred = 5
            if pred < 1:
                pred = 1
            rmse += (self.data_matrix[u, i] - pred) ** 2
        rmse = np.sqrt(rmse / len(nnz_test))
        return rmse

    def optimizer_function(self, num_features, lambda_user, lambda_item, gamma):
        """
        This function is used for optimize the hyperparameters with the Gaussian process.

        Parameters
        ----------
        num_features : int
            Number of features.
        lambda_user : float
            Regularization parameter.
        lambda_item : float
            Regularization parameter.
        gamma : float
            Regularization parameter.

        Returns
        -------
        float
            mean rmse of a 5-fold cv.

        """
        cv = 5
        kf = KFold(n_splits=cv, random_state=self.random_state, shuffle=True)
        test_RMSE_list = []
        for train_indices, test_indices in kf.split(self.indices):
            train, test = self.get_train_test_matrix(train_indices, test_indices)
            self.fit(train, int(num_features), lambda_user, lambda_item, gamma)
            test_rmse = self.get_test_rmse(test)
            test_RMSE_list.append(test_rmse)
        mean_test_rmse = np.mean(test_RMSE_list)
        return -mean_test_rmse

    def log_hyperparams_to_json(self):
        """
        This funciton is used for logging the optimal hyperparameters.

        Returns
        -------
        None.

        """
        optimizer = BayesianOptimization(
            f=self.optimizer_function, pbounds=self.tuning_params, random_state=self.random_state
        )
        path = get_git_root() + "/logs/SGD.json"
        logger = JSONLogger(path=path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=7, n_iter=20)


if __name__ == "__main__":

    SGD = SGD()
    SGD.log_hyperparams_to_json()
