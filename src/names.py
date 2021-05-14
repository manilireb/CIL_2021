def get_log_file_name_knn(algo_name, user_based, sim_name):
    """
    This function returns the name of the json file where the hyperparameters are logged.
    Its only purpose is to make the constructors more readable

    Parameters
    ----------
    algo_name : str
        Name of the algorithm e.g. KNNBasic.
    user_based : bool
        Whether similarities will be computed between users or between items.
    sim_name : str
        Name of similarity measure method

    Returns
    -------
    log_file_name : str
        Name of the log file.

    """

    log_file_name = algo_name
    sim = "User" if user_based else "Item"
    log_file_name += sim
    if sim_name == "pearson_baseline":
        sim_name = "PearsonBaseline"
        log_file_name += sim_name
    else:
        log_file_name += sim_name.capitalize()
    log_file_name += ".json"
    return log_file_name


def get_log_file_name_mf(algo_name, biased):
    """
    This function returns the name of the json file where the hyperparameters are logged for the matrix factorization based method.
    Its only purpose is to make the constructor more readable.

    Parameters
    ----------
    algo_name : str
        Name of the algorithm. E.g. SVD
    biased : bool
        Wheter to use baslines or bias.

    Returns
    -------
    log_file_name : str
        Name of the logg file.

    """
    log_file_name = algo_name
    method = "Biased" if biased else "UnBiased"
    log_file_name += method
    log_file_name += ".json"
    return log_file_name
