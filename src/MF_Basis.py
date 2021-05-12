from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

from algo_base import BaseAlgo


class MF_Basis(BaseAlgo):
    """
    This class provides the instantiation of the  MF based algorithms

    """

    def __init__(self):
        super().__init__()
        self.tuning_params = None
        self.random_state = 42
        self.log_file_name = None

    def log_hyperparameters_to_json(self):
        """
        This functions logs the optimal hyperparameters of the given range calculated by the Gaussian process into a json file.
        """

        optimizer = BayesianOptimization(
            f=self.optimizer_function, pbounds=self.tuning_params, random_state=self.random_state
        )
        logger = JSONLogger(path="./logs/" + self.log_file_name)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=3, n_iter=10)
