import matplotlib.pyplot as plt
import numpy as np

import data_loader
from data_loader import (NUMBER_OF_MOVIES, get_data_matrix, get_mask_matrix,
                         impute_with_col_mean, impute_with_row_mean)
from model import BaseModel


class SVD(BaseModel):
    def __init__(self):
        super().__init__()
        self.data_matrix = get_data_matrix(
            self.train_users, self.train_items, self.train_predictions
        )
        self.mask = get_mask_matrix(self.train_users, self.train_items)

    def get_approx_matrix(self, n_eigenvalues, impute=None):
        """
        Get best rank-k approximation of the data matrix,
        where k = n_eigenvalues

        Parameters
        ----------
        n_eigenvalues : int
            Rank approximation number.
        impute : func, optional
            Funtcion used to impute unobserved variable.
            E.g. impute_with_row_mean from the data_loader module

        Returns
        -------
        reconstructed_matrix : ndarray
            Best rank-n_eigenvalues approximation of data_matrix

        """
        if impute != None:
            impute(self.data_matrix, self.mask)

        U, s, Vt = np.linalg.svd(self.data_matrix, full_matrices=False)
        S = np.zeros((NUMBER_OF_MOVIES, NUMBER_OF_MOVIES))
        eigenvalues_indices = np.arange(n_eigenvalues)
        S[eigenvalues_indices, eigenvalues_indices] = s[eigenvalues_indices]

        reconstructed_matrix = U @ S @ Vt
        return reconstructed_matrix


if __name__ == "__main__":
    svd_model = SVD()
    
    print("-------------- Impute with overall mean ---------------------")
    approx_matrix = svd_model.get_approx_matrix(n_eigenvalues=2)
    print("SVD with 2 eigenvalues:", svd_model.get_score(approx_matrix))

    print()
    print("-------------- Impute with row mean ---------------------")
    approx_matrix = svd_model.get_approx_matrix(
        n_eigenvalues=2, impute=impute_with_row_mean
    )
    print("SVD with 2 eigenvalues:", svd_model.get_score(approx_matrix))

    print()
    print("-------------- Impute with col mean ---------------------")
    approx_matrix = svd_model.get_approx_matrix(
        n_eigenvalues=2, impute=impute_with_col_mean
    )
    print("SVD with 2 eigenvalues:", svd_model.get_score(approx_matrix))
