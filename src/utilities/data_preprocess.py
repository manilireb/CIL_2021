import os

import git
import numpy as np
import pandas as pd


def get_git_root():
    """
    The purpose of this function is to avoid FileNotFoundErrors. It just returns the root directory of the git repository.
    Using this function we can load the data in every subdirectory of the repository.


    Returns
    -------
    git_root : str
        path to the git root directory.

    """
    path = os.getcwd()
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


class Data:
    """
    This class bundles all methods for preprocessing the data given in the 'data_train.csv' file.
    Take a look at the examples/fetch_data.py file to see how to use it.

    Parameters
    ----------
    filename : str, optional
        The name of the file where the data is stored. The default is 'data_train.csv'.

    """

    def __init__(self, filename="data_train.csv"):
        self.dir = get_git_root()
        self.dir += "/data/"
        self.filename = self.dir + filename
        self.n_users = 10000
        self.n_items = 1000

    def load_data(self):
        """
        loads the data from the csv file into a np.ndarray.

        Returns
        -------
        Type: np.ndarray
            data from the csv file .

        """

        data_pd = pd.read_csv(self.filename)
        return np.array(data_pd)

    def extract_row_and_col_number(self, entry):
        """
        extracts the row and the col number of an entry in the data and returns it as ints.

        Parameters
        ----------
        entry : str
            String given by the data in the first column of data_train.
            E.g. 'r631_c741'

        Returns
        -------
        row : int
            the row number.
        col : int
            the col number.

        """

        row_col_string = entry.split("_")
        row = int(row_col_string[0][1:])
        col = int(row_col_string[1][1:])
        return row, col

    def get_user_and_item_ids(self, data):
        """
        Returns arrays with the userIDs and the itemIDs.

        Parameters
        ----------
        data : np.ndarray
            data returned by the load_data method.

        Returns
        -------
        userID : np.ndarray
            array of the userIDs.
        itemID : np.ndarray
            array of the itemIDs

        """

        vec_extract = np.vectorize(self.extract_row_and_col_number)
        userID, itemID = vec_extract(data[:, 0])
        userID -= 1
        itemID -= 1
        return userID, itemID

    @staticmethod
    def get_df():
        """
        Static method that returns the pandas dataframe that
        can be used by the surprise library for custom datasets

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the userID itemID and rating as
            columns

        """

        D = Data()
        data = D.load_data()
        userID, itemID = D.get_user_and_item_ids(data)
        rating = data[:, 1]
        data_np = np.stack((userID, itemID, rating), axis=-1)
        df = pd.DataFrame(data_np)
        df.columns = ["userID", "itemID", "rating"]
        return df
