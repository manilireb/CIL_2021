import matplotlib.pyplot as plt
import numpy as np

import data_loader
from data_loader import (NUMBER_OF_MOVIES, get_data_matrix, get_mask_matrix,
                         impute_with_col_mean, impute_with_row_mean)
from model import BaseModel


def pretty_print(func):
    def wrapper(*args, **kwargs):
        obj = args[0]
        imputer = obj.imputer
        if imputer == None:
            print("Impute with overall mean")
        else:
            print("Impute with function:", imputer.__name__)
        res = func(*args, **kwargs)
        n_eigvals = kwargs["n_eigenvalues"]
        print(
            "SVD with", n_eigvals, "eigenvalues has RMSE score of:", obj.get_score(res),
        )
        return res

    return wrapper


class SVD(BaseModel):
    def __init__(self, imputer=None):
        super().__init__()
        self.imputer = imputer
        self.data_matrix = get_data_matrix(
            self.train_users, self.train_items, self.train_predictions
        )
        self.mask = get_mask_matrix(self.train_users, self.train_items)
        if imputer != None:
            imputer(self.data_matrix, self.mask)
        self.U, self.s, self.Vt = np.linalg.svd(self.data_matrix, full_matrices=False)

    @pretty_print
    def get_approx_matrix(self, n_eigenvalues):
        """
        Get best rank-k approximation of the data matrix,
        where k = n_eigenvalues

        Parameters
        ----------
        n_eigenvalues : int
            Rank approximation number.

        Returns
        -------
        reconstructed_matrix : ndarray
            Best rank-n_eigenvalues approximation of data_matrix

        """
        S = np.zeros((NUMBER_OF_MOVIES, NUMBER_OF_MOVIES))
        eigenvalues_indices = np.arange(n_eigenvalues)
        S[eigenvalues_indices, eigenvalues_indices] = self.s[eigenvalues_indices]

        reconstructed_matrix = self.U @ S @ self.Vt
        return reconstructed_matrix

    def plot_singular_values(self):
        """
        plots the first 20 singular values to check how many of them are
        important to include in our approximation. According to the
        plots I would suggest that 3 is a reasonable number

        Returns
        -------
        None.

        """
        plt.title("Singular Values of Data Matrix")
        plt.plot(self.s[:20], "bo")
        plt.ylabel("singular values")
        plt.show()


if __name__ == "__main__":

    svd_model = SVD()
    n_eigvals = 3
    approx_matrix = svd_model.get_approx_matrix(n_eigenvalues=n_eigvals)
    print()

    svd_model_row_mean = SVD(imputer=impute_with_row_mean)
    approx_matrix = svd_model_row_mean.get_approx_matrix(n_eigenvalues=n_eigvals)
    print()

    svd_model_col_mean = SVD(imputer=impute_with_col_mean)
    approx_matrix = svd_model_col_mean.get_approx_matrix(n_eigenvalues=n_eigvals)
