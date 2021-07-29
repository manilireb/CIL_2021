import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import KFold

from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from utilities.data_preprocess import Data

'''
script for creating the 5-fold cv on the average of the best 6 models
'''

if __name__ == "__main__":

    df = Data().get_df()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    f = open("fusion_output.txt", "w")
    f.write("Averaging of the following models: \n")

    models = [
        KNN_WithMeans(sim_name="pearson_baseline", user_based=False),
        KNN_WithZScore(sim_name="pearson_baseline", user_based=False),
        MFNMF(biased=False),
        KNN_WithZScore(sim_name="pearson", user_based=False),
        KNN_WithMeans(sim_name="pearson", user_based=False),
        MFSVD(biased=True),
    ]

    cv_rmse = []

    kf = KFold(n_splits=5, random_state=42)
    verbose = True
    for train, test in kf.split(data):
        predictions = []
        for model in models:
            if verbose:
                f.write(model.log_file_name[:-5] + "\n")
            m = model.get_opt_model()
            p = m.fit(train).test(test)
            test_pred = [pred[3] for pred in p]
            predictions.append(test_pred)
        verbose = False
        pred_array = np.array(predictions).T
        fusion = np.mean(pred_array, axis=1)
        ground_truth = np.array([pred[2] for pred in p])
        rmse = np.sqrt(np.sum((fusion - ground_truth) ** 2) / len(test))
        cv_rmse.append(rmse)

    f.write("5-fold RMSE of the fusion: " + str(sum(cv_rmse) / 5) + "\n")
    f.close()
