# Please ensure Proc. wants matrix is unticked (the tick box below this window)
# The user input is required to define key variables below

DEGREES_OF_FREEDOM = 1
KNOTS = (22 , 60)
NEW_COLUMN_PREFIX = "DriverAge"
KEEP_ORIGINAL_COLUMN = False 














# This code runs the spline creation
# Users should NOT make changes to this
import dataiku
from dataiku import pandasutils as pdu
import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices

class RegressionSplines:

    def __init__(self,degree_freedom, knots, new_col_prefix, keep_original):
        self.degree_freedom = degree_freedom
        self.knots = knots
        self.new_col_prefix = new_col_prefix
        self.formula_string = f"bs(train, knots={self.knots}, degree={self.degree_freedom}, include_intercept=False)"
        self.original_col = None
        self.keep_original = keep_original
    
    def fit(self, series):
        self.original_col = series.name
        self.bounds = [series.min(), series.max()]
    
    def rename_columns(self, df):
        num_cols = len(df.columns)
        new_cols = [i + '_Spline_' + str(j) for i, j in zip([self.new_col_prefix] * num_cols, range(num_cols))]
        df.columns = new_cols
        return df
    
    def generate_splines(self, design_info, train_x):
        transformed_x = build_design_matrices([design_info], {'train': train_x}, return_type='dataframe')[0]
        transformed_x.drop('Intercept', axis=1, inplace=True)
        return transformed_x
    
    def concatenate(self, original_df, feature_splines):
        if not self.keep_original:
            original_df = None
        return pd.concat([original_df, feature_splines], axis=1)
    
    
    def transform(self, series):
        design_info = dmatrix(self.formula_string, {"train": self.bounds}, return_type='dataframe').design_info
        feature_splines = self.generate_splines(design_info, series)
        feature_splines = self.rename_columns(feature_splines)
        
        new_df = self.concatenate(series, feature_splines)

        return new_df    

processor = RegressionSplines(DEGREES_OF_FREEDOM, KNOTS, NEW_COLUMN_PREFIX,KEEP_ORIGINAL_COLUMN)






