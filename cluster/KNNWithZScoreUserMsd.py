from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreUserMsd = KNN_WithZScore(sim_name="msd", user_based=True)
KNNWithZScoreUserMsd.log_hyperparameters_to_json()
