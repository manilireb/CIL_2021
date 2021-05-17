import sys

sys.path.append("../")

import numpy as np
from surprise import SVD
from surprise.model_selection import cross_validate

from algo_base import BaseAlgo
from utilities.names import get_log_file_name_mf as get_log_file_name


class MFSVD(BaseAlgo):
    """
    This class provides the instantiation of the SVD methods.
    Take a look at examples/hyperparameters_svd.py to see how to use it

    Parameters
    ----------
    biased : bool
        Wheter we use the biased method or not.
    n_epochs : int, optional
        Number of epochs. The default is 100.


    """

    def __init__(self, biased, n_epochs=100):
        super().__init__()
        self.n_epochs = n_epochs
        self.biased = biased
        self.tuning_params = {
            "n_factors": [5, 150],
            "lr_bu": [0.001, 0.009],
            "lr_bi": [0.001, 0.009],
            "lr_pu": [0.001, 0.009],
            "lr_qi": [0.001, 0.009],
            "reg_qi": [0.01, 0.9],
            "reg_bu": [0.01, 0.9],
            "reg_bi": [0.01, 0.9],
            "reg_pu": [0.01, 0.9],
        }
        self.algo = SVD
        self.log_file_name = get_log_file_name("SVD", biased)

    def optimizer_function(self, n_factors, lr_bu, lr_bi, lr_pu, lr_qi, reg_qi, reg_bu, reg_bi, reg_pu):
        """
        Function that gets optimized by the gaussian process.
        The function returns (-1) times the mean of a 5-fold crossvalidation on the specified hyperparameters.
        The gaussian process tries to find the parameters that yield the maximum value for the given function. Since we are looking for the parameters that yield minimum value we multply by (-1).

        Parameters
        ----------
        n_factors : int
            The number of factors.
        lr_bu : float
            Learning rate.
        lr_bi : float
            Learning rate.
        lr_pu : float
            Learning rate.
        lr_qi : float
            Learning rate.
        reg_qi : float
            Regularization term.
        reg_bu : float
            Regularization term.
        reg_bi : float
            Regularization term.
        reg_pu : float
            Regularization term.

        Returns
        -------
        float
            (-1) times mean of the 5-fold CV.

        """
        algo = self.algo(
            n_factors=int(n_factors),
            n_epochs=self.n_epochs,
            biased=self.biased,
            lr_bu=lr_bu,
            lr_bi=lr_bi,
            lr_pu=lr_pu,
            lr_qi=lr_qi,
            reg_qi=reg_qi,
            reg_bu=reg_bu,
            reg_bi=reg_bi,
            reg_pu=reg_pu,
            random_state=self.random_state,
        )
        cv_res = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=True)
        return -np.mean(cv_res.get("test_rmse"))

    def get_test_rmse(self):
        """
        Returns the average test rmse of a 5-fold cross-validation on the algorithm with the optimal hyperparameters found by the Gaussian process.

        Returns
        -------
        float
            average of the test rmse of a 5-fold cv.

        """
        opt_hyperparams = self.get_opt_hyperparams()
        opt_hyperparams["n_factors"] = int(opt_hyperparams["n_factors"])
        algo = self.algo(n_epochs=self.n_epochs, biased=self.biased, **opt_hyperparams, random_state=self.random_state)
        cv = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=False)
        return np.mean(cv.get("test_rmse"))
