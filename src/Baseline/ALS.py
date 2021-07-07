import numpy as np
import scipy.sparse as sp
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from tqdm import tqdm

from utilities.data_preprocess import Data, get_git_root


class ALS(BaseEstimator, TransformerMixin):
    """
    Class for the ALS baseline algorithm.

    Parameters
    --------
    data_matrix : sparse.lil.lil_matrix
        The data matrix.
    nnz_users :
        list of nonzero user indices
    nnz_items :
        list of nonzero item indices
    indices : np.ndarray
        list from 0 to the number of nonzero elements in the data
    num_users: int
        number of users
    num_items: int
        number of items
    num_epochs: int
        number of epochs
    user_matrix: np.ndarray
        The user matrix
    item_matrix: np.ndarray
        The item matrix
    tuning_params: dict
        dictionary with the parameter intervals

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
            "lambda_user": [0.0005, 30.0],
            "lambda_item": [0.0005, 30.0],
        }
        np.random.seed(self.random_state)

    def init_matrix(self, train, num_features):
        """
        This method is used for initializing user_matrix and item_matrix.

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
            Initialized item matirx.

        """
        user_matrix = 5 * np.random.rand(self.num_users, num_features)
        item_matrix = 5 * np.random.rand(num_features, self.num_items)
        item_nnz = train.getnnz(axis=0)
        item_sum = train.sum(axis=0)
        item_matrix[0, :] = item_sum / item_nnz
        return user_matrix, item_matrix

    def update_user_matrix(self, data, user_matrix, item_matrix, num_features, lambda_user):
        """
        Method for update the user matrix.


        Parameters
        ----------
        data : sparse.lil.lil_matrix
            Data matrix.
        user_matrix : np.ndarray
            User matrix.
        item_matrix : np.ndarray
            Item matrix.
        num_features : int
            Number of features.
        lambda_user : float
            regularization parameter for the user matrix.

        Returns
        -------
        user_matrix : np.ndarray
            Updated user matrix.

        """
        lambda_I = lambda_user * sp.eye(num_features)
        for u in range(self.num_users):
            indices = self.get_observed_items_per_user(data, u)
            V = item_matrix[:, indices]
            A = V @ V.T + lambda_I
            B = V @ data[u, indices].T
            user_matrix[u, :] = np.linalg.solve(A, B).squeeze()
        return user_matrix

    def update_item_matrix(self, data, user_matrix, item_matrix, num_features, lambda_item):
        """
        Method for update the item matrix

        Parameters
        ----------
        data : sparse.lil.lil_matrix
            data matrix.
        user_matrix : np.ndarray
            User matrix.
        item_matrix : np.ndarray
            Item matrix.
        num_features : int
            Number of features.
        lambda_item : float
            Regularization parameter for the item matirx.

        Returns
        -------
        item_matrix : np.ndarray
            Updated item matrix.

        """
        lambda_I = lambda_item * sp.eye(num_features)
        for i in range(self.num_items):
            indices = self.get_observed_users_per_item(data, i)
            U = user_matrix[indices, :]
            A = U.T @ U + lambda_I
            B = U.T @ data[indices, i]
            item_matrix[:, i] = np.linalg.solve(A, B).squeeze()
        return item_matrix

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

    def get_observed_items_per_user(self, data, user_idx):
        """
        This method returns a list of all nonzero indices for a given user in the
        data matirx

        Parameters
        ----------
        data : sparse.lil.lil_matrix
            Data matrix.
        user_idx : int
            ID of the user.

        Returns
        -------
        np.ndarray
            List of all nonzero indices for the given user.

        """
        return data.getrow(user_idx).nonzero()[1]

    def get_observed_users_per_item(self, data, item_idx):
        """
        This method returns a list of all nonzero indices for a given item in the
        data matrix

        Parameters
        ----------
        data : sparse.lil.lil_matrix
            Data matrix.
        item_idx : int
            ID of the item.

        Returns
        -------
        np.ndarray
            List of all nonzero indices for a given item.

        """
        return data.getcol(item_idx).nonzero()[0]

    def fit(self, data, num_features, lambda_user, lambda_item):
        """
        Function that fits a user and item matrix

        Parameters
        ----------
        data : sparse.lil.lil_matrix
            Data matrix.
        num_features : int
            Number of latent features of the data.
        lambda_user : float
            Regularization parameter.
        lambda_item : float
            Regularization parameter.

        Returns
        -------
        None.

        """

        user_matrix, item_matrix = self.init_matrix(data, num_features)
        for it in tqdm(range(self.num_epochs)):
            user_matrix = self.update_user_matrix(data, user_matrix, item_matrix, num_features, lambda_user)
            item_matrix = self.update_item_matrix(data, user_matrix, item_matrix, num_features, lambda_item)
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix

    def get_test_rmse(self, test):
        """
        Return rmse for the test data

        Parameters
        ----------
        test : sparse.lil.lil_matrix
            Test data.

        Returns
        -------
        rmse : float
            RMSE for the test data.

        """
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
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

    def optimizer_function(self, num_features, lambda_user, lambda_item):
        """
        This function is used for optimize the hyperparameters with the Gaussian process.

        Parameters
        ----------
        num_features : int
            Number of latent features.
        lambda_user : float
            Regularization parameter for the user.
        lambda_item : float
            Regularization parameter for the item.

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
            self.fit(train, int(num_features), lambda_user, lambda_item)
            test_rmse = self.get_test_rmse(test)
            test_RMSE_list.append(test_rmse)
        mean_test_rmse = np.mean(test_RMSE_list)
        return -mean_test_rmse

    def log_hyperparams_to_json(self):
        """
        This function is used for storing the optimal hyperparameters

        Returns
        -------
        None.

        """
        optimizer = BayesianOptimization(
            f=self.optimizer_function, pbounds=self.tuning_params, random_state=self.random_state
        )
        path = get_git_root() + "/logs/ALS.json"
        logger = JSONLogger(path=path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=4, n_iter=10)
