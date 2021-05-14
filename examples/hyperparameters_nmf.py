import sys

sys.path.append("../src/")

from MF_NMF import MFNMF

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

NMFBiased = MFNMF(biased=True)
NMFUnBiased = MFNMF(biased=False)

NMFBiased.log_hyperparameters_to_json()
NMFUnBiased.log_hyperparameters_to_json()
