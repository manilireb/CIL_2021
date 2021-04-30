import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
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
        if sim_name == "pearson_baseline":
            self.tuning_params["shrinkage"] = [0, 200]
        self.log_file_name = None
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
        algo = self.algo(k=int(k), min_k=int(min_k), sim_options=self.sim_options, verbose=False)
        cv_res = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=True)
        return -np.mean(cv_res.get("test_rmse"))

    def log_hyperparameters_to_json(self):
        """
        This functions logs the optimal hyperparameters of the given range calculated by the Gaussian process into a json file.
        """
        optimizer = BayesianOptimization(f=self.optimizer_function, pbounds=self.tuning_params, random_state=42)
        logger = JSONLogger(path="./logs/" + self.log_file_name)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=3, n_iter=10)
