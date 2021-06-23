from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansUserCosine = KNN_WithMeans(sim_name="cosine", user_based=True)

KNNWithMeansUserCosine.log_hyperparameters_to_json()
