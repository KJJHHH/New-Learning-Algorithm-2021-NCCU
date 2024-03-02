from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch

class MyDataset(Data.Dataset):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __getitem__(self, index):
        X_, y_ = self.X[index], self.y[index]
        return X_, y_

    def __len__(self):
        return len(self.X)
    
def preprocess\
        (
        data,
        dtype = torch.float64,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ):

    data = pd.read_csv('Copper_forecasting_data.csv')

    sc = StandardScaler()
    X = data.drop(["y"], axis = 1)
    X = sc.fit_transform(X)
    y = data["y"] / 1000

    train_size = int(len(X)*0.8)

    X_train = X[:train_size, :]
    y_train = y[:train_size]
    X_test = X[train_size:, :]
    y_test = y[train_size:] 

    X_train = torch.tensor(np.array(X_train), dtype=dtype).to(device)
    X_test = torch.tensor(np.array(X_test), dtype=dtype).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=dtype).reshape(-1,1).to(device)
    y_test = torch.tensor(np.array(y_test), dtype=dtype).reshape(-1,1).to(device)
    
    return X_train, X_test, y_train, y_test

def loader(
    X, y, batch_size = 30, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    return \
        torch.utils.data.DataLoader(
            MyDataset(X.to(device), y.to(device)), 
            batch_size = batch_size, 
            shuffle=False, 
            drop_last = False)

