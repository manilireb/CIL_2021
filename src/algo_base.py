import json
from abc import ABC, abstractmethod

import numpy as np
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

    def get_opt_hyperparams(self):
        """
        This function returns a dict of the optimal hyperparams stored in the log_file. The file only exists if the log_hyperparameters_to_json function has already been executed before.

        Returns
        -------
        dict
            Dictionary of the optimal hyperparameters found by the Gaussian process. If the hyperparameters were not found yet, then it return an empty dictionary.

        """
        file_name = "../examples/logs/" + self.log_file_name
        try:
            with open(file_name) as handle:
                json_data = [json.loads(line) for line in handle]

            rmse = []
            for dic in json_data:
                rmse.append(dic.get("target"))

            index = np.argmax(rmse)
            return json_data[index].get("params")
        except:
            print("The file doesn't exist since you not optimized your hyperparameters yet!")
            return dict()

    @abstractmethod
    def get_test_rmse(self):
        pass
