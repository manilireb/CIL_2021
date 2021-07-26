import csv

import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from Co_Clustering.Clustering_Coclustering import Clustering_Coclustering
from Ensemble.GBE import GBE
from KNN_Methods.KNN_Basics import KNN_Basic
from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from SlopeOne.slope_one import Slope_One
from utilities.data_preprocess import Data


def main():

    df = Data().get_df()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    models = [
        KNN_WithMeans(sim_name="pearson_baseline", user_based=False),
        KNN_WithZScore(sim_name="pearson_baseline", user_based=False),
        MFNMF(biased=False),
        KNN_WithZScore(sim_name="pearson", user_based=False),
        KNN_WithMeans(sim_name="pearson", user_based=False),
        MFSVD(biased=True),
        KNN_WithMeans(sim_name="msd", user_based=False),
        KNN_WithZScore(sim_name="msd", user_based=False),
        KNN_WithZScore(sim_name="pearson_baseline", user_based=True),
        KNN_WithMeans(sim_name="pearson_baseline", user_based=True),
        KNN_WithMeans(sim_name="cosine", user_based=False),
        Slope_One(),
        KNN_WithZScore(sim_name="pearson", user_based=True),
        KNN_WithZScore(sim_name="cosine", user_based=False),
        KNN_WithMeans(sim_name="pearson", user_based=True),
        KNN_WithZScore(sim_name="msd", user_based=True),
        Clustering_Coclustering(),
        KNN_WithZScore(sim_name="cosine", user_based=True),
        KNN_WithMeans(sim_name="msd", user_based=True),
        KNN_WithMeans(sim_name="cosine", user_based=True),
        MFSVD(biased=False),
        KNN_Basic(sim_name="msd", user_based=True),
        MFNMF(biased=False),
        KNN_Basic(sim_name="pearson_baseline", user_based=True),
        KNN_Basic(sim_name="cosine", user_based=True),
        KNN_Basic(sim_name="pearson", user_based=True),
        KNN_Basic(sim_name="msd", user_based=False),
        KNN_Basic(sim_name="pearson_baseline", user_based=False),
        KNN_Basic(sim_name="pearson", user_based=False),
        KNN_Basic(sim_name="cosine", user_based=False),
    ]

    # getting prediction of models
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    predictions = []
    for model in models:
        print("fitting: ", model)
        m = model.get_opt_model()
        p = m.fit(train).test(test)
        test_pred = [pred[3] for pred in p]
        predictions.append(test_pred)
    pred_array = np.array(predictions).T
    ground_truth = np.array([pred[2] for pred in p])

    reg = GBE.get_opt_model()
    reg.fit(pred_array, ground_truth)

    # loading and preparing prediction set
    df = Data("sampleSubmission.csv").get_df()
    test_data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    train = data.build_full_trainset()
    test = test_data.build_full_trainset().build_testset()

    # creating predictions
    predictions = []
    for model in models:
        print("fitting: ", model)
        m = model.get_opt_model()
        p = m.fit(train).test(test)
        test_pred = [pred[3] for pred in p]
        predictions.append(test_pred)

    pred_array_submission = np.array(predictions).T
    UID = np.array([pred[0] for pred in p]) + 1
    IID = np.array([pred[1] for pred in p]) + 1

    y_hat = reg.predict(pred_array_submission)

    # preparing submission
    pos = ["Id"]
    pred = ["Prediction"]
    for i in range(len(y_hat)):
        pos.append(f"r{UID[i]}_c{IID[i]}")
        pred.append(y_hat[i])

    with open("submission_ridge_more_models_all.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(pos, pred))


if __name__ == "__main__":
    main()
