from itertools import groupby

import numpy as np
import scipy.sparse as sp
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from sklearn.model_selection import KFold
from tqdm import tqdm

from utilities.data_preprocess import Data, get_git_root


class ALS:
    def __init__(self):
        D = Data()
        self.data_matrix = D.get_sparse_ratings_matrix()
        self.random_state = 42
        np.random.seed(self.random_state)
        self.user_features = None
        self.item_features = None
        self.tuning_params = {
            "num_epochs": [5, 100],
            "num_features": [5, 50],
            "lambda_user": [0.0005, 1.0],
            "lambda_item": [0.0005, 1.0],
        }

    def init_MF(self, train, num_features):
        num_users, num_items = train.get_shape()

        user_features = np.random.rand(num_users, num_features)
        item_features = np.random.rand(num_items, num_features)

        item_nnz = train.getnnz(axis=0)
        item_sum = train.sum(axis=0)

        for ind in range(num_items):
            item_features[ind, 0] = item_sum[0, ind] / item_nnz[ind]

        return user_features, item_features

    def group_by(self, data, index):
        sorted_data = sorted(data, key=lambda x: x[index])
        groupby_data = groupby(sorted_data, lambda x: x[index])
        return groupby_data

    def build_index_groups(self, train):
        nnz_row, nnz_col = train.nonzero()
        nnz_train = list(zip(nnz_row, nnz_col))
        grouped_nnz_train_by_row = self.group_by(nnz_train, index=0)
        nnz_row_col_indices = [(g, np.array([v[1] for v in value])) for g, value in grouped_nnz_train_by_row]

        grouped_nnz_train_by_col = self.group_by(nnz_train, index=1)
        nnz_col_row_indices = [(g, np.array([v[0] for v in value])) for g, value in grouped_nnz_train_by_col]

        return nnz_train, nnz_row_col_indices, nnz_col_row_indices

    def update_user_feature(self, train, item_features, lambda_user, nnz_items_per_user, nnz_user_item_indices):
        num_user = nnz_items_per_user.shape[0]
        num_features = item_features.shape[1]
        lambda_I = lambda_user * sp.eye(num_features)
        updated_user_features = np.zeros((num_user, num_features))

        for user, item in nnz_user_item_indices:
            M = item_features[item, :]
            V = M.T @ train[user, item].T
            A = M.T @ M + nnz_items_per_user[user] * lambda_I
            X = np.linalg.solve(A, V)
            updated_user_features[user, :] = np.copy(X.T)

        return updated_user_features

    def update_item_features(self, train, user_features, lambda_item, nnz_users_per_item, nnz_item_user_indices):
        num_items = nnz_users_per_item.shape[0]
        num_features = user_features.shape[1]
        lambda_I = lambda_item * sp.eye(num_features)
        updated_item_features = np.zeros((num_items, num_features))

        for item, user in nnz_item_user_indices:
            M = user_features[user, :]
            V = M.T @ train[user, item]
            A = M.T @ M + nnz_users_per_item[item] * lambda_I
            X = np.linalg.solve(A, V)
            updated_item_features[item, :] = np.copy(X.T)

        return updated_item_features

    def compute_rmse(self, data, nnz):
        mse = 0.0
        for row, col in nnz:
            user_info = self.user_features[row, :]
            item_info = self.item_features[col, :]
            mse += (data[row, col] - user_info @ item_info) ** 2

        return np.sqrt(mse / len(nnz))

    def fit(self, train, num_epochs, num_features, lambda_user, lambda_item):

        user_features, item_features = self.init_MF(train, num_features)

        nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=1), train.getnnz(axis=0)

        nnz_train, nnz_user_item_indices, nnz_item_user_indices = self.build_index_groups(train)

        for it in tqdm(range(num_epochs)):

            user_features = self.update_user_feature(
                train, item_features, lambda_user, nnz_items_per_user, nnz_user_item_indices
            )

            item_features = self.update_item_features(
                train, user_features, lambda_item, nnz_users_per_item, nnz_item_user_indices
            )

        self.user_features = user_features
        self.item_features = item_features

    def get_test_rmse(self, test):
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = self.compute_rmse(test, nnz_test)
        return rmse

    def optimizer_function(self, num_epochs, num_features, lambda_user, lambda_item):
        cv = 5
        kf = KFold(n_splits=cv, random_state=self.random_state, shuffle=True)
        test_RMSE_list = []

        for train_index, test_index in kf.split(self.data_matrix):
            train_ratings, test_ratings = self.data_matrix[train_index], self.data_matrix[test_index]
            self.fit(train_ratings, num_epochs, num_features, lambda_user, lambda_item)
            test_rmse = self.get_test_rmse(test_ratings)
            test_RMSE_list.append(test_rmse)

        mean_test_rmse = np.mean(test_RMSE_list)
        return -mean_test_rmse

    def log_hyperparams_to_json(self):
        optimizer = BayesianOptimization(
            f=self.optimizer_function, pbounds=self.tuning_params, random_state=self.random_state
        )
        path = get_git_root() + "/logs/ALS.json"
        logger = JSONLogger(path=path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=7, n_iter=20)
