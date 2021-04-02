import math

from sklearn.metrics import mean_squared_error

import data_loader
from data_loader import (extract_users_items_predictions,
                         get_train_and_test_data)


class BaseModel:
    def __init__(self):
        self.train_data, self.test_data = get_train_and_test_data("data_train.csv")
        (
            self.train_users,
            self.train_items,
            self.train_predictions,
        ) = extract_users_items_predictions(self.train_data)
        (
            self.test_users,
            self.test_items,
            self.test_predictions,
        ) = extract_users_items_predictions(self.test_data)
        self.rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

    def get_score(self, data_matrix):
        """
        Computes the RMSE score of the predicted values of the test data
        

        Parameters
        ----------
        data_matrix : ndarray
            the data matrix of the train data. The other values are imputed

        Returns
        -------
        TYPE : float
            rmse of the test data.

        """
        predictions = data_matrix[self.test_users, self.test_items]
        return self.rmse(predictions, self.test_predictions)


if __name__ == "__main__":

    model = BaseModel()
    data_matrix = data_loader.get_data_matrix(
        model.train_users, model.train_items, model.train_predictions
    )
    mask = data_loader.get_mask_matrix(model.train_users, model.train_items)
    print("RMSE with overall mean:", model.get_score(data_matrix))
    data_loader.impute_with_row_mean(data_matrix, mask)
    print("RMSE with row mean:", model.get_score(data_matrix))
    data_loader.impute_with_col_mean(data_matrix, mask)
    print("RMSE with col mean:", model.get_score(data_matrix))
