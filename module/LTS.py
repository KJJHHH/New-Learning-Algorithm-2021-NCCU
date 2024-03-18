import torch
from module.Data import MyDataset
import numpy as np
from module.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_lts(out_file, model, X_train, y_train, learning_goal):    
    n = 0
    n_not_fit = 0
    write(out_file, f"-------> Initislising lts")
    train_loader, indices, X_train_lts, y_train_lts, n = \
            lts(model, X_train, y_train, learning_goal, n_not_fit, out_file)
    n_not_fit = torch.ceil(torch.tensor((X_train.shape[0] - n)/2)).to(torch.int)
    return n, n_not_fit

# 2. obtaining_LTS / selecting_LTS
def lts(model, X_train, y_train, lr_goal, n_not_fit, out_file, dtype = torch.float64, batch_size=100):
    """
    X_train, y_trian, lr_goal, n_not_fit
    ---
    # output: train_loader, indices_lts, n
    train_loader: with lts
    n of train size 
    """

    # predict and residuals
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
    y_train = y_train.reshape(-1 ,1)
    resid_square = torch.square(y_pred - y_train).reshape(-1)

    # obtaining
    # prompt: find the indices of tensor < k only shape 1 tensor
    resid_square, sorted_indices = torch.sort(resid_square) # default ascending
    indices_lts = sorted_indices[resid_square < lr_goal]
    X_train_lts, y_train_lts = X_train[indices_lts], y_train[indices_lts]

    # check if obtaining is true. 0 is correct
    out_file.write(f"Total obtaining n: {len(indices_lts)}\n")
    out_file.write(f"obtaining n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts) > lr_goal).sum()}\n")
    print(f"Total obtaining n: {len(indices_lts)}")
    print(f"obtaining n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts) > lr_goal).sum()}")

    # selecting
    n = len(indices_lts) + n_not_fit
    indices_lts = sorted_indices[:n]
    X_train_lts, y_train_lts = X_train[indices_lts], y_train[indices_lts]

    # check if the selected is true. 1 is correct
    out_file.write(f"Total select n: {len(indices_lts)}\n")
    out_file.write(f"select n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts)>lr_goal).sum()}\n")
    print(f"Total select n: {len(indices_lts)}")
    print(f"select n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts)>lr_goal).sum()}")


    # train loader of lts
    train_loader = torch.utils.data.DataLoader(
        MyDataset(X_train_lts, y_train_lts), 
        batch_size = batch_size, 
        shuffle=False, 
        drop_last = False)
    

    return train_loader, indices_lts, X_train_lts, y_train_lts, len(X_train_lts)