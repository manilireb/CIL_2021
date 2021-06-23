from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansItemCosine = KNN_WithMeans(sim_name="cosine", user_based=False)

KNNWithMeansItemCosine.log_hyperparameters_to_json()
