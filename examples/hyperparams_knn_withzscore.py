from KNN_WithZScore import KNN_WithZScore

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

KNNWithZScoreUserCosine = KNN_WithZScore(sim_name="cosine", user_based=True)
KNNWithZScoreUserMsd = KNN_WithZScore(sim_name="msd", user_based=True)
KNNWithZScoreUserPearson = KNN_WithZScore(sim_name="pearson", user_based=True)
KNNWithZScoreUserPearsonBaseline = KNN_WithZScore(sim_name="pearson_baseline", user_based=True)

KNNWithZScoreItemCosine = KNN_WithZScore(sim_name="cosine", user_based=False)
KNNWithZScoreItemMsd = KNN_WithZScore(sim_name="msd", user_based=False)
KNNWithZScoreItemPearson = KNN_WithZScore(sim_name="pearson", user_based=False)
KNNWithZScoreItemPearsonBaseline = KNN_WithZScore(sim_name="pearson_baseline", user_based=False)


KNNWithZScoreUserCosine.log_hyperparameters_to_json()
KNNWithZScoreUserMsd.log_hyperparameters_to_json()
KNNWithZScoreUserPearson.log_hyperparameters_to_json()
KNNWithZScoreUserPearsonBaseline.log_hyperparameters_to_json()


KNNWithZScoreItemCosine.log_hyperparameters_to_json()
KNNWithZScoreItemMsd.log_hyperparameters_to_json()
KNNWithZScoreItemPearson.log_hyperparameters_to_json()
KNNWithZScoreItemPearsonBaseline.log_hyperparameters_to_json()
