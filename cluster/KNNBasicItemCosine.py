from KNN_Methods.KNN_Basics import KNN_Basic

KNNBasic_item_cosine = KNN_Basic(sim_name="cosine", user_based=False)
KNNBasic_item_cosine.log_hyperparameters_to_json()
