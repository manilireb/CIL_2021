from Co_Clustering.Clustering_Coclustering import Clustering_Coclustering
from KNN_Methods.KNN_Basics import KNN_Basic
from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from SlopeOne.slope_one import Slope_One

if __name__ == "__main__":

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

    test_rmses_with_names = []

    for mod in models:
        test_rmse = mod.get_test_rmse()
        name = mod.log_file_name[:-5]
        test_rmses_with_names.append((test_rmse, name))

    test_rmses_with_names.sort(key=lambda tup: tup[0])

    with open("model_performances.txt", "w") as f:
        for tup in test_rmses_with_names:
            line = tup[1] + " : " + str(tup[0])
            f.write(line)
            f.write("\n")
