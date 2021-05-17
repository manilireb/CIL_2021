import sys

sys.path.append("../")

import numpy as np
from surprise.model_selection import cross_validate

from algo_base import BaseAlgo


class KNN_Basis(BaseAlgo):
    """
    This class provides the instantiation and the hyperparameter tuning of the KNN inspired algorithms.
    Parameters
    ----------
    sim_name : str
        Name of similarity measure method
    user_based : bool
        Whether similarities will be computed between users or between items.
    """

    def __init__(self, sim_name, user_based):
        super().__init__()
        self.sim_options = {"name": sim_name, "user_based": user_based}
        self.tuning_params = {"k": [5, 800], "min_k": [1, 20], "min_support": [1, 50]}
        self.sim_name = sim_name
        if self.sim_name == "pearson_baseline":
            self.tuning_params["shrinkage"] = [0, 200]
        self.algo = None

    def optimizer_function(self, k, min_k, min_support, shrinkage=None):
        """
        Function that gets optimized by the gaussian process.
        The function returns (-1) times the mean of a 5-fold crossvalidation on the specified hyperparameters.
        The gaussian process tries to find the parameters that yield the maximum value for the given function. Since we are looking for the parameters that yield minimum value we multply by (-1).
        Parameters
        ----------
        k : int
            The (max) number of neighbors to take into account for aggregation.
        min_k : int
            The minimum number of neighbors to take into account for aggregation.
        min_support : int
            The minimum number of common items (when 'user_based' is 'True') or minimum number of common users (when 'user_based' is 'False') for the similarity not to be zero
        shrinkage : int
            Shrinkage parameter to apply (only relevant for pearson_baseline similarity). The default is None
        Returns
        -------
        float
            (-1) times mean of the 5-fold CV.
        """
        self.sim_options["min_support"] = int(min_support)

        if self.sim_name == "pearson_baseline":
            self.sim_options["shrinkage"] = int(shrinkage)

        algo = self.algo(k=int(k), min_k=int(min_k), sim_options=self.sim_options, verbose=False)
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
        self.sim_options["min_support"] = int(opt_hyperparams["min_support"])
        del opt_hyperparams["min_support"]
        self.sim_options["shrinkage"] = int(opt_hyperparams["shrinkage"])
        del opt_hyperparams["shrinkage"]
        opt_hyperparams["k"] = int(opt_hyperparams["k"])
        opt_hyperparams["min_k"] = int(opt_hyperparams["min_k"])
        algo = self.algo(sim_options=self.sim_options, **opt_hyperparams)
        cv = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=False)
        return np.mean(cv.get("test_rmse"))
