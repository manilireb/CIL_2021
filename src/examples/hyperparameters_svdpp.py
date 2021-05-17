import sys

sys.path.append("../")

from MF_Methods.MF_SVDpp import MFSVDpp

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

SVDpp = MFSVDpp()
SVDpp.log_hyperparameters_to_json()
