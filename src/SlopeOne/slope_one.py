import numpy as np
from surprise import SlopeOne
from surprise.model_selection import cross_validate

from algo_base import BaseAlgo
from utilities.names import get_log_file_name


class Slope_One(BaseAlgo):
    """
    This class provides an instantiation of the SlopeOne method.
    Take a look at examples/rmse_slope_one.py to see how to use it.
    Note that the slope one method has no hyperparameters, therefore we dont need to optimize them

    Parameters
    ----------

    """

    def __init__(self):
        super().__init__()
        self.log_file_name = get_log_file_name("SlopeOne")
        self.algo = SlopeOne

    def optimizer_function(self):
        """
        Method has no hyperparameters, so we don't need an optimizer function.

        """
        pass

    def get_test_rmse(self):
        """
        Returns the average of a 5-fold cross validation.

        Returns
        -------
        float
            average of the test rmse of a 5-fold cv.

        """
        algo = self.algo()
        cv = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=False)
        return np.mean(cv.get("test_rmse"))
