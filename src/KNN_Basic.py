from surprise import KNNBasic

from KNN import KNN_Basis


class KNN_Basic(KNN_Basis):
    """
    This class provides the instantiation and the hyperparameter tuning of the KNNBasic methods.
    Take a look at examples/hyperparams_knn_basic.py to see how to use it.
    Parameters
    ----------
    sim_name : str
        Name of similarity measure method
    user_based : bool
        Whether similarities will be computed between users or between items.
    """

    def __init__(self, sim_name, user_based):
        super().__init__(sim_name, user_based)
        log_file_name = "KNNBasic_"
        sim = "user_" if user_based else "item_"
        log_file_name += sim
        log_file_name += sim_name
        log_file_name += ".json"
        self.log_file_name = log_file_name

        self.algo = KNNBasic
