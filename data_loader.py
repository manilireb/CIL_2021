import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# global constants
NUMBER_OF_USERS, NUMBER_OF_MOVIES = (10000, 1000)


def get_train_and_test_data(filename, train_size=0.9):
    """
    returns ndarrays of the train and the test data    

    Parameters
    ----------
    filename : String
        name of the csv file that contains the data
    train_size : Float
        size of the train data in percentage. The default is 0.9.

    Returns
    -------
    (ndarray, ndarray)
        ndarrays of the train and the test data.

    """
    data_pd = pd.read_csv(filename)
    train_pd, test_pd = train_test_split(
        data_pd, train_size=train_size, random_state=42
    )
    return np.array(train_pd), np.array(test_pd)


def get_row_and_col_number(entry):
    """
    Extracts the row and the col number out of the string given
    in the data_train.csv file

    Parameters
    ----------
    entry : String
        String given by the data in the first column of data_train.
        E.g. 'r631_c741'

    Returns
    -------
    row : Int
        The row number of the entry.
    col : Int
        The column number of the entry.

    """
    row_col_string = entry.split("_")
    row = int(row_col_string[0][1:])
    col = int(row_col_string[1][1:])
    return row, col


def extract_users_items_predictions(data):
    """
    Returns arrays of the row (user), the col (items) and the entries (predictions)
    of the data
    
    Parameters
    ----------
    data : ndarray
        train_data or test_data given by the function get_train_and_test_data

    Returns
    -------
    users : ndarray
        rows of the matrix.
    items : ndarray
        columns of the matrix.
    predictions : ndarray
        predictions of the users.

    """
    predictions = data[:, 1]
    row_col_strings = data[:, 0]
    row_cols = [get_row_and_col_number(entry) for entry in row_col_strings]
    users = np.array([row - 1 for (row, col) in row_cols])
    items = np.array([col - 1 for (row, col) in row_cols])
    return users, items, predictions


def get_data_matrix(users, items, predictions):
    """
    Returns the data matrix

    Parameters
    ----------
    users : ndarray
        array of row indices given by get_row_and_col_array
    items : ndarray
        array of column indices given by get_row_and_col_array.
    predicitons : ndarray
        array of predictions that are observed

    Returns
    -------
    data_matrix : ndarray
        10'000 x 1000 matrix filled with with the ratings at the corresponding entries
        the missing values are filled with the mean rating

    """
    data_matrix = np.full((NUMBER_OF_USERS, NUMBER_OF_MOVIES), np.mean(predictions))
    data_matrix[users, items] = predictions
    return data_matrix


def get_mask_matrix(users, items):
    """
    Returns the mask matrix. I.e. matrix of dimension same as data matrix
    and the entries are 1 if the corresponding entry was observed and 0 otherwise

    Parameters
    ----------
    users : ndarray
         array of row indices given by get_row_and_col_array
    items : ndarray
         array of column indices given by get_row_and_col_array.

    Returns
    -------
    mask : ndarray
        mask matrix

    """
    mask = np.zeros((NUMBER_OF_USERS,NUMBER_OF_MOVIES))
    mask[users, items] = 1
    return mask

def impute_with_row_mean(data_matrix, mask):
    """
    Fills the unobserved data with the mean of the corresponding row
    (Attention! this function has side effects on data_matrix)

    Parameters
    ----------
    data_matrix : ndarray
        the data matrix
    mask : ndarray
        the mask matrix (same shape as data matrix).

    Returns
    -------
    None.

    """
    for i in range(NUMBER_OF_USERS):
        row = data_matrix[i, :] * mask[i, :]
        row_nonzero = row[row != 0]
        row_mean = np.mean(row_nonzero)
        data_matrix[i, mask[i, :] == 0] = row_mean


def impute_with_col_mean(data_matrix, mask):
    """
    Fills the unobserved data with the mean of the correpsonding column
    (Attention! this function has side effects on data_matrix)

    Parameters
    ----------
    data_matrix : ndarray
        the data matrix.
    mask : ndarray
        the mask matrix (Same shape as data matrix).

    Returns
    -------
    None.

    """
    for i in range(NUMBER_OF_MOVIES):
        col = data_matrix[:, i] * mask[:, i]
        col_nonzero = col[col != 0]
        col_mean = np.mean(col_nonzero)
        data_matrix[mask[:, i] == 0, i] = col_mean


if __name__ == "__main__":
    train_data, test_data = get_train_and_test_data("data_train.csv")
    users, items, predictions = extract_users_items_predictions(train_data)
    data_matrix = get_data_matrix(users, items, predictions)
    mask = get_mask_matrix(users, items)
