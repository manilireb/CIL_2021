import sys

sys.path.append("../")

from KNN_Methods.KNN_WithMeans import KNN_WithMeans

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

KNNWithMeansUserCosine = KNN_WithMeans(sim_name="cosine", user_based=True)
KNNWithMeansUserMsd = KNN_WithMeans(sim_name="msd", user_based=True)
KNNWithMeansUserPearson = KNN_WithMeans(sim_name="pearson", user_based=True)
KNNWithMeansUserPearsonBaseline = KNN_WithMeans(sim_name="pearson_baseline", user_based=True)

KNNWithMeansItemCosine = KNN_WithMeans(sim_name="cosine", user_based=False)
KNNWithMeansItemMsd = KNN_WithMeans(sim_name="msd", user_based=False)
KNNWithMeansItemPearson = KNN_WithMeans(sim_name="pearson", user_based=False)
KNNWithMeansItemPearsonBaseline = KNN_WithMeans(sim_name="pearson_baseline", user_based=False)


KNNWithMeansUserCosine.log_hyperparameters_to_json()
KNNWithMeansUserMsd.log_hyperparameters_to_json()
KNNWithMeansUserPearson.log_hyperparameters_to_json()
KNNWithMeansUserPearsonBaseline.log_hyperparameters_to_json()

KNNWithMeansItemCosine.log_hyperparameters_to_json()
KNNWithMeansItemMsd.log_hyperparameters_to_json()
KNNWithMeansItemPearson.log_hyperparameters_to_json()
KNNWithMeansItemPearsonBaseline.log_hyperparameters_to_json()
