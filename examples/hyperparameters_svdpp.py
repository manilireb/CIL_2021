import sys

sys.path.append("../src/")

from MF_SVDpp import MFSVDpp

SVDpp = MFSVDpp()
SVDpp.log_hyperparameters_to_json()
