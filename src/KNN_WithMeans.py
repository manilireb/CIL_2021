from surprise import KNNWithMeans

from KNN import KNN_Basis
from names import get_log_file_name_knn as get_log_file_name


class KNN_WithMeans(KNN_Basis):
    """
    This class provides the instantiation of the KNNWithMeans methods.
    Take a look at examples/hyperparams_knn_withmeans.py to see how to use it.
    Parameters
    ----------
    sim_name : str
        Name of similarity measure method
    user_based : bool
        Whether similarities will be computed between users or between items.
    """

    def __init__(self, sim_name, user_based):
        super().__init__(sim_name, user_based)
        self.log_file_name = get_log_file_name("KNNWithMeans", user_based, sim_name)
        self.algo = KNNWithMeans
