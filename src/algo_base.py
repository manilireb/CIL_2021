from abc import ABC, abstractmethod

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from surprise import Dataset, Reader

from data_preprocess import Data


class BaseAlgo(ABC):
    """
    Abstract class that defines the main functionalties.
    """

    def __init__(self):
        df = Data.get_df()
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
        self.tuning_params = None
        self.log_file_name = None
        self.random_state = 42

    @abstractmethod
    def optimizer_function(self):
        pass

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
