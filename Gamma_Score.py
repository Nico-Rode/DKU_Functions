import numpy as np
from statsmodels.genmod.families.family import Gamma

def score(y_valid, y_pred):
    n = y_valid.count()
    gamma = Gamma()
    return gamma.deviance(endog=y_valid, mu=y_pred) / n
