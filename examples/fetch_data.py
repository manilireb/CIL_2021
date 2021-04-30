import sys

sys.path.append("../src/")


import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

from data_preprocess import Data

df = Data.get_df()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

algo = SVD()

cross_validate(algo, data, measures=["RMSE"], cv=5, verbose=True)
