def get_log_file_name(algo_name, user_based, sim_name):
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
    log_file_name += sim_name.capitalize()
    log_file_name += ".json"
    return log_file_name
