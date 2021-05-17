import sys

sys.path.append("../")


from surprise import SlopeOne

from algo_base import BaseAlgo
from utilities.names import get_log_file_name


class Slope_One(BaseAlgo):
    """
    This class provides an instantiation of the SlopeOne method.
    Take a look at examples/hyperparams_slope_one.py to see how to use it.

    Parameters
    ----------

    """

    def __init__(self):
        super().__init__()
        self.log_file_name = get_log_file_name("SlopeOne")
        self.algo = SlopeOne
