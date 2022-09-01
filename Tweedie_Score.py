import numpy as np
from statsmodels.genmod.families.family import Tweedie

def score(y_valid, y_pred):
    n = y_valid.count()
    tweedie = Tweedie(var_power=1.5)
    return tweedie.deviance(endog=y_valid, mu=y_pred) / n
