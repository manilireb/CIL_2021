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


if __name__ == "__main__":
    train_data, test_data = get_train_and_test_data("data_train.csv")

    # short test to see if the function works
    for i in range(30):
        row, col = get_row_and_col_number(train_data[i, 0])
        print(row, col)
