import numpy as np
from surprise import NMF
from surprise.model_selection import cross_validate

from algo_base import BaseAlgo
from names import get_log_file_name_mf as get_log_file_name


class MFNMF(BaseAlgo):
    def __init__(self, biased, n_epochs=100):
        super().__init__()
        self.n_epochs = n_epochs
        self.biased = biased
        self.tuning_params = {
            "n_factors": [1, 150],
            "reg_pu": [0.01, 0.9],
            "reg_qi": [0.01, 0.9],
            "reg_bu": [0.01, 0.9],
            "reg_bi": [0.01, 0.9],
            "lr_bu": [0.001, 0.09],
            "lr_bi": [0.001, 0.09],
        }
        self.algo = NMF
        self.log_file_name = get_log_file_name("NMF", biased)

    def optimizer_function(self, n_factors, reg_pu, reg_qi, reg_bu, reg_bi, lr_bu, lr_bi):
        """
        Function that gets optimized by the gaussian process.
        The function returns (-1) times the mean of a 5-fold crossvalidation on the specified hyperparameters.
        The gaussian process tries to find the parameters that yield(-1) times mean of the 5-fold CV the maximum value for the given function. Since we are looking for the parameters that yield minimum value we multply by (-1).

        Parameters
        ----------
        n_factors : int
            The number of factors.
        reg_pu : float
            Regularizarion term.
        reg_qi : float
            Regularization term.
        reg_bu : float
            Regularization term.
        reg_bi : float
            Regularization term.
        lr_bu : float
            learning rate.
        lr_bi : float
            Learning rate.

        Returns
        -------
        float
            (-1) times mean of the 5-fold CV.

        """

        algo = self.algo(
            n_factors=int(n_factors),
            n_epochs=self.n_epochs,
            biased=self.biased,
            reg_pu=reg_pu,
            reg_qi=reg_qi,
            reg_bu=reg_bu,
            reg_bi=reg_bi,
            lr_bu=lr_bu,
            lr_bi=lr_bi,
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
