from sklearn.ensemble import RandomForestClassifier 
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class CustomAdaBoost(BaseEstimator):
    def __init__(self, base_estimator, num_estimator, learning_rate):
        self.n_estimators = base_estimator
        self.max_depth = num_estimator
        self.random_state = learning_rate

    def fit(self, X, y, sample_weight=None):
        """
        takes in training data and fits a model
        """
        clf = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=self.random_state)
        self.model = clf.fit(X,y)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = list(set(y))
        
    def predict(self, X):
        """
        Returns the target as 1D array
        """
        y_pred = pd.Series(self.model.predict(X))
        return y_pred
    
    def predict_proba(self, X):
        """
        Returns the target as 1D array
        """
        y_pred_proba = np.array(self.model.predict_proba(X))
        return y_pred_proba
