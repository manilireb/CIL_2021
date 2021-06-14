from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansUserMsd = KNN_WithMeans(sim_name="msd", user_based=True)

KNNWithMeansUserMsd.log_hyperparameters_to_json()
