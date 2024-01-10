import pandas as pd  
import numpy as np
from scipy.io import arff
import os

def code_nominal(df: pd.DataFrame):
    nominal_columns = df.columns[df.dtypes == 'object']

    df_encoded = pd.get_dummies(df, columns=nominal_columns, dtype=float)

    return df_encoded

def remove_nans(df: pd.DataFrame):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def split_labels_from_features(df: pd.DataFrame, target_column: str):
    labels = df.loc[:, target_column]
    features = df.drop(columns=target_column)

    return features, labels
        
def load_dataframe(filepath, target_column=None):

    _, extension = os.path.splitext(filepath)

    if extension == '.csv':

        df = pd.read_csv(filepath)
        return df
    
    elif extension == '.arff':

        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        
        if target_column is not None:
            df[target_column] = df[target_column].astype('str')

        return df
    
    return pd.DataFrame([])