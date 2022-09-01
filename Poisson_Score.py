import numpy as np
from statsmodels.genmod.families.family import Poisson

def score(y_valid, y_pred):
    n = y_valid.count()
    poisson = Poisson()
    return poisson.deviance(endog=y_valid, mu=y_pred) / n