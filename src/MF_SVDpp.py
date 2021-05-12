import numpy as np
from surprise import SVDpp
from surprise.model_selection import cross_validate

from MF_Basis import MF_Basis


class MFSVDpp(MF_Basis):
    def __init__(self, n_epochs=100):
        super().__init__()
        self.n_epochs = n_epochs
        self.algo = SVDpp
        self.log_file_name = "SVDpp.json"
        self.tuning_params = self.tuning_params = {
            "n_factors": [5, 150],
            "lr_bu": [0.001, 0.009],
            "lr_bi": [0.001, 0.009],
            "lr_pu": [0.001, 0.009],
            "lr_qi": [0.001, 0.009],
            "lr_yj": [0.001, 0.009],
            "reg_qi": [0.01, 0.9],
            "reg_bu": [0.01, 0.9],
            "reg_bi": [0.01, 0.9],
            "reg_pu": [0.01, 0.9],
            "reg_yj": [0.01, 0.9],
        }

    def optimizer_function(self, n_factors, lr_bu, lr_bi, lr_pu, lr_qi, lr_yj, reg_qi, reg_bu, reg_bi, reg_pu, reg_yj):

        algo = self.algo(
            n_factors=int(n_factors),
            n_epochs=self.n_epochs,
            lr_bu=lr_bu,
            lr_bi=lr_bi,
            lr_pu=lr_pu,
            lr_qi=lr_qi,
            lr_yj=lr_yj,
            reg_qi=reg_qi,
            reg_bu=reg_bu,
            reg_bi=reg_bi,
            reg_pu=reg_pu,
            reg_yj=reg_yj,
            random_state=self.random_state,
        )
        cv_res = cross_validate(algo, self.data, measures=["rmse"], cv=5, n_jobs=-1, verbose=True)
        return -np.mean(cv_res.get("test_rmse"))
