import sys

sys.path.append("../src/")

from slope_one import Slope_One

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

slopeOne = Slope_One();
slopeOne.log_hyperparameters_to_json()