import pandas as pd 
import numpy as np

from surprise import Dataset, Reader
from utilities.data_preprocess import Data
from surprise.model_selection import train_test_split
from surprise import accuracy

from Co_Clustering.Clustering_Coclustering import Clustering_Coclustering
from KNN_Methods.KNN_Basics import KNN_Basic
from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from SlopeOne.slope_one import Slope_One

if __name__ == "__main__":

    f = open("averaging_output.txt", "w")

    df = Data.get_df()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=.15)

    models = [
            Clustering_Coclustering(),
            MFNMF(biased=True),
            MFNMF(biased=False),
            MFSVD(biased=True),
            MFSVD(biased=False),
            KNN_Basic(sim_name="cosine", user_based=True),
            KNN_Basic(sim_name="msd", user_based=True),
            KNN_Basic(sim_name="pearson", user_based=True),
            KNN_Basic(sim_name="pearson_baseline", user_based=True),
            KNN_Basic(sim_name="cosine", user_based=False),
            KNN_Basic(sim_name="msd", user_based=False),
            KNN_Basic(sim_name="pearson", user_based=False),
            KNN_Basic(sim_name="pearson_baseline", user_based=False),
            KNN_WithMeans(sim_name="cosine", user_based=True),
            KNN_WithMeans(sim_name="msd", user_based=True),
            KNN_WithMeans(sim_name="pearson", user_based=True),
            KNN_WithMeans(sim_name="pearson_baseline", user_based=True),
            KNN_WithMeans(sim_name="cosine", user_based=False),
            KNN_WithMeans(sim_name="msd", user_based=False),
            KNN_WithMeans(sim_name="pearson", user_based=False),
            KNN_WithMeans(sim_name="pearson_baseline", user_based=False),
            KNN_WithZScore(sim_name="cosine", user_based=True),
            KNN_WithZScore(sim_name="msd", user_based=True),
            KNN_WithZScore(sim_name="pearson", user_based=True),
            KNN_WithZScore(sim_name="pearson_baseline", user_based=True),
            KNN_WithZScore(sim_name="cosine", user_based=False),
            KNN_WithZScore(sim_name="msd", user_based=False),
            KNN_WithZScore(sim_name="pearson", user_based=False),
            KNN_WithZScore(sim_name="pearson_baseline", user_based=False),
            Slope_One(),
        ]
    
    predictions = []

    for model in models:
        f.write('fiting model ' + model.log_file_name[:-5] + '\n')
        m = model.get_opt_model()
        m.fit(trainset)
        test_pred = m.test(testset)
        predictions.append(test_pred)
        f.write('FINISHED fitting model ' + model.log_file_name[:-5] + ' predictions now: \n')
        f.write(*predictions + '\n')

    pred_array = np.array(predictions)
    final_prediction = pred_array.sum()/pred_array.size()

    f.write(str(accuracy.rmse(final_prediction, verbose= True)) + '\n')
    f.close()
