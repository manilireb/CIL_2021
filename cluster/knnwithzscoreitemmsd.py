from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreItemMsd = KNN_WithZScore(sim_name="msd", user_based=False)

KNNWithZScoreItemMsd.log_hyperparameters_to_json()
