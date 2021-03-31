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


def get_row_and_col_array(data):
    """
    Returns arrays of the row and col indices
    
    Parameters
    ----------
    data : ndarray
        train_data or test_data given by the function get_train_and_test_data

    Returns
    -------
    rows : ndarray
        rows of the matrix.
    cols : ndarray
        columns of the matrix.

    """

    row_col_strings = data[:, 0]
    row_cols = [get_row_and_col_number(entry) for entry in row_col_strings]
    rows = np.array([row - 1 for (row, col) in row_cols])
    cols = np.array([col - 1 for (row, col) in row_cols])
    return rows, cols


def get_data_matrix(data, rows, cols):
    """
    Return the data matrix

    Parameters
    ----------
    data : ndarray
        train_data or test_data given by the function get_trian_and_test_data.
    rows : ndarray
        array of row indices given by get_row_and_col_array
    cols : ndarray
        array of column indices given by get_row_and_col_array.

    Returns
    -------
    data_matrix : ndarray
        10'000 x 1000 matrix filled with with the ratings at the corresponding entries
        the missing values are filled with the mean rating

    """
    known_ratings = data[:, 1]
    data_matrix = np.full(
        (NUMBER_OF_USERS, NUMBER_OF_MOVIES), np.mean(known_ratings), dtype=np.uint8
    )
    data_matrix[rows, cols] = known_ratings
    return data_matrix


def get_mask_matrix(rows, cols):
    """
    Returns the mask matrix. I.e. matrix of dimension same as data matrix
    and the entries are 1 if the corresponding entry was observed and 0 otherwise

    Parameters
    ----------
    rows : ndarray
         array of row indices given by get_row_and_col_array
    cols : ndarray
         array of column indices given by get_row_and_col_array.

    Returns
    -------
    mask : ndarray
        mask matrix

    """
    mask = np.zeros((NUMBER_OF_USERS, NUMBER_OF_MOVIES), dtype=np.uint8)
    mask[rows, cols] = 1
    return mask


if __name__ == "__main__":
    train_data, test_data = get_train_and_test_data("data_train.csv")
    rows, cols = get_row_and_col_array(train_data)
    data_matrix = get_data_matrix(train_data, rows, cols)
    mask = get_mask_matrix(rows, cols)
