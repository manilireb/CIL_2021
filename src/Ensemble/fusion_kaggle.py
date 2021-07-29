import csv

import numpy as np
from surprise import Dataset, Reader

from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from utilities.data_preprocess import Data

'''
Script for create a submission file for kaggle for the averaging of the best 6 models.
'''

if __name__ == "__main__":

    df = Data().get_df()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    df = Data("sampleSubmission.csv").get_df()
    test_data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    train = data.build_full_trainset()
    test = test_data.build_full_trainset().build_testset()

    models = [
        KNN_WithMeans(sim_name="pearson_baseline", user_based=False),
        KNN_WithZScore(sim_name="pearson_baseline", user_based=False),
        MFNMF(biased=False),
        KNN_WithZScore(sim_name="pearson", user_based=False),
        KNN_WithMeans(sim_name="pearson", user_based=False),
        MFSVD(biased=True),
    ]

    predictions = []
    for model in models:
        m = model.get_opt_model()
        p = m.fit(train).test(test)
        test_pred = [pred[3] for pred in p]
        predictions.append(test_pred)
    pred_array = np.array(predictions).T
    fusion = np.mean(pred_array, axis=1)

    UID = np.array([pred[0] for pred in p]) + 1
    IID = np.array([pred[1] for pred in p]) + 1

    pos = ["Id"]
    ratings = ["Prediction"]
    for i in range(len(fusion)):
        pos.append(f"r{UID[i]}_c{IID[i]}")
        ratings.append(fusion[i])
    with open("submission_fuison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(pos, ratings))
