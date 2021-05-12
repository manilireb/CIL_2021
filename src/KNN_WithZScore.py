from surprise import KNNWithZScore

from KNN import KNN_Basis
from names import get_log_file_name


class KNN_WithZScore(KNN_Basis):
    """
    This class provides an instantiation of the KNNWithZScore methods.
    Take a look at examples/hyperparams_knn_withzscore.py to see how to use it

    Parameters
    ----------
    sim_name : str
        Name of similarity measure method.
    user_based : bool
        Wheter similarities will be computed between user or between items.

    """

    def __init__(self, sim_name, user_based):
        super().__init__(sim_name, user_based)
        self.log_file_name = get_log_file_name("KNNWithZScore", user_based, sim_name)
        self.algo = KNNWithZScore
