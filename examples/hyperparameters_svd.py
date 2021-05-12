import sys

sys.path.append("../src/")

from MF_SVD import MFSVD

SVDBiased = MFSVD(biased=True)
SVDUnBiased = MFSVD(biased=False)

SVDBiased.log_hyperparameters_to_json()
SVDUnBiased.log_hyperparameters_to_json()
