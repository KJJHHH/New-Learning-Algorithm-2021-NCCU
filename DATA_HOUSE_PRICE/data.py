import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def preprocess(random_seed):
    # load
    train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")

    # check data type
    train.dtypes.unique()
    object_cols = list(train.select_dtypes(include=['object']).columns)

    """
    # all use dummies for convenient. maybe later can use ordinals
    # ordinal
    ordinal_cols = ['LotShape', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                    'HouseStyle', ...]
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].astype('category')
            train[col] = train[col].cat.codes
    # dummies
    dummy_cols = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofMatl', 'RoofStyle', ...]
    """
    # dummies
    def get_dummies(data, col):
        dummies = pd.get_dummies(data[col], prefix=col).astype(int)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)
        return data

    for col in object_cols:
        train = get_dummies(train, col)
        test = get_dummies(test, col)
    

    # drop missing
    train = train.dropna()
    test = test.dropna()

    # train, val
    train, val = train_test_split(train, test_size=0.2, random_state=random_seed)

    # x, y
    index_train = train['Id']
    index_val = test['Id']
    X_train = train.drop(['SalePrice', 'Id'], axis=1)
    y_train = train['SalePrice']/10000
    X_val = val.drop(['SalePrice', 'Id'], axis=1)
    y_val = val['SalePrice']/10000

    return X_train, y_train, X_val, y_val, index_train, index_val

def cpu_to_gpu(X_train, y_train, X_test, y_test):
    dtype = torch.float64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(np.array(X_train), dtype=dtype).to(device)
    X_test = torch.tensor(np.array(X_test), dtype=dtype).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=dtype).reshape(-1,1).to(device)
    y_test = torch.tensor(np.array(y_test), dtype=dtype).reshape(-1,1).to(device)
    return X_train, y_train, X_test, y_test


