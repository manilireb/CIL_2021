import sys

sys.path.append("../")

from MF_Methods.MF_SVD import MFSVD

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

SVDBiased = MFSVD(biased=True)
SVDUnBiased = MFSVD(biased=False)

SVDBiased.log_hyperparameters_to_json()
SVDUnBiased.log_hyperparameters_to_json()
