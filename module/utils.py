import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate_loss(model, iterator, criterion = nn.MSELoss()):
    """
    model: model for evaluate
    iterator: loader
    criterion: loss function        
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(iterator):
            preds = model(X)
            loss = criterion(preds, y)
            val_loss += loss.item()
        val_loss /= len(iterator)
    return val_loss 

def acceptable_eps_ypred(train_loader, model, lr_goal):
    """
    train_loader: train_loader
    model: model
    eps_bound: learning goal
    ---
    output: acceptable, eps, y_pred
    max eps
    acceptable
    y_pred 
    """
    eps_square = []
    y_pred = []
    for _, (X, y) in enumerate(train_loader):
        y = y
        preds = model(X)
        eps_square.append(torch.square(y-preds))
        y_pred.append(preds)
    eps_square, y_pred = torch.cat(eps_square, dim = 0), torch.cat(y_pred, dim = 0)
    if max(eps_square) < lr_goal:
        return True, eps_square, y_pred
    else:
        return False, eps_square, y_pred

def write(out_file, log, print_ = True):
    out_file.write(log)
    if print_:
        print(log)