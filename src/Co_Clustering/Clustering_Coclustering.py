import numpy as np
from surprise import CoClustering
from surprise.model_selection import cross_validate

from algo_base import BaseAlgo
from utilities.names import get_log_file_name


class Clustering_Coclustering(BaseAlgo):
    """
    This class provides the instantiation of the Coclustering methods.
    Take a look at examples/hyperparameters_coclustering.py to see how to use it

    Parameters
    ----------
    n_epochs : int, optional
        Number of epochs. The default is 100.

    """

    def __init__(self):
        super().__init__()
        self.tuning_params = {"n_epochs": [5, 400], "n_cltr_u": [1, 100], "n_cltr_i": [1, 100]}
        self.algo = CoClustering
        self.log_file_name = get_log_file_name("CoClustering")

    def optimizer_function(self, n_epochs, n_cltr_u, n_cltr_i):
        """
        Function that gets optimized by the gaussian process.
        The function returns (-1) times the mean of a 5-fold crossvalidation on the specified hyperparameters.
        The gaussian process tries to find the parameters that yield the maximum value for the given function. Since we are looking for the parameters that yield minimum value we multply by (-1).


        Parameters
        ----------
        n_epochs : int
            Number of epochs.
        n_cltr_u : float
            Number of user clusters.
        n_cltr_i : float
            Number of item clusters.

        Returns
        -------
        float
            (-1) times mean of the 5-fold CV.

        """
        algo = self.algo(
            n_cltr_u=int(n_cltr_u), n_cltr_i=int(n_cltr_i), n_epochs=int(n_epochs), random_state=self.random_state
        )
        cv_res = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=True)
        return -np.mean(cv_res.get("test_rmse"))

    def get_test_rmse(self):
        """
        Returns the average test rmse of a 5-fold cross-validation on the algorithm with the optimal hyperparameters found by the gaussian process.

        Returns
        -------
        float
            average of the test rmse of a 5-fold cv.

        """
        opt_hyperparams = self.get_opt_hyperparams()
        opt_hyperparams["n_epochs"] = int(opt_hyperparams["n_epochs"])
        opt_hyperparams["n_cltr_u"] = int(opt_hyperparams["n_cltr_u"])
        opt_hyperparams["n_cltr_i"] = int(opt_hyperparams["n_cltr_i"])
        algo = self.algo(**opt_hyperparams, random_state=self.random_state)
        cv = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=False)
        return np.mean(cv.get("test_rmse"))

    def get_opt_model(self):
        """
        Returns the optimal model given by the trained hyperparams

        Returns
        -------
        algo : prediction_algorithms.co_clustering.CoClustering
            The Surprise CoClustering model with the optimal hyperparams

        """
        opt_hyperparams = self.get_opt_hyperparams()
        opt_hyperparams["n_epochs"] = int(opt_hyperparams["n_epochs"])
        opt_hyperparams["n_cltr_u"] = int(opt_hyperparams["n_cltr_u"])
        opt_hyperparams["n_cltr_i"] = int(opt_hyperparams["n_cltr_i"])
        algo = self.algo(**opt_hyperparams, random_state=self.random_state)
        return algo
