from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansUserPearson = KNN_WithMeans(sim_name="pearson", user_based=True)

KNNWithMeansUserPearson.log_hyperparameters_to_json()