from abc import ABC, abstractmethod

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

    @abstractmethod
    def optimizer_function(self):
        pass

    @abstractmethod
    def log_hyperparameters_to_json(self):
        pass
