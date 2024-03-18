import torch
import torch.nn as nn
import torch.optim as optim
import gc, os, random
from module.Weight_tune import *
from module.Model import TwoLayerNet
from module.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class reorganising():
    def __init__(
            self, previous_module, train_loader, test_loader, out_file,
            criterion = nn.MSELoss(), 
            epochs = 50, 
            lr_rate = .001, 
            lr_bound = .00001,  
            lr_goal = .0001, 
            print_reg = False,
            print_w_tune = False,
            validate_run = False):
        super(reorganising, self).__init__()
        # Reorganising module
        # regular module
        #   model => self.model: main model (final model) in regualrising module. 
        # wieght tune module
        #   model => create new model when process  
        """
        previous_module: "wt" or "Cram"
        hidden_dim: hidden dimension of model
        epochs: number of epochs
        criterion: loss function
        lr_reg: learning rate for regularization
        lr_w_tune: learning rate for weight tuning
        lr_bound_reg: learning rate lower bound for regularization
        lr_bound_w_tune: learning rate lower bound for weight tuning      
        eps_reg: early stopping epsilon for regularization
        eps_w_tune: early stopping epsilon for weight tuning
        validate_run: use model with test loader and check loss if True
        """        

        # Initialise: check if accetable SLFN (wt.pth) exist  
        self.out_file = out_file
        self.acceptable_path = f'acceptable/{previous_module}.pth'
        if os.path.exists(self.acceptable_path):
            self.model = torch.load(self.acceptable_path)
        else:
            assert False, f"Acceptable SLFN not exist in '{self.acceptable_path}'"
                         
        self.input_dim = self.model.layer_1.weight.data.shape[1]
        self.hidden_dim = self.model.layer_1.weight.data.shape[0]
        self.train_loader = train_loader
        self.test_loader = test_loader


        self.epochs = epochs
        self.criterion = criterion
        self.lr_rate = lr_rate

        self.lr_bound = lr_bound
        self.lr_goal = lr_goal

        self.validate_run = validate_run
    

    """
    need model, optimizer
    """
    def module_weight_EU_LG_UA(self, model):
        """
        Train the trimmed model, if acceptable, set the reorg model to the trimmed model, else resotre 
        the model before trim
        1. model: trimm model
        ---
        # NOTE: do not have to save model after this module
        """
        
        # SOmethine to create
        temp_save_path = "_temp/reorg_wt.pth"
        model.to(device).train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr_rate)
        loss_old = 5e+7
        train_loss_list = []
        test_loss_list = []
        
        # Train porcess
        for epoch in range(self.epochs):
            gc.collect()

            train_loss = 0

            # forward operation
            for _, (X, y) in enumerate(self.train_loader):
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                preds = model(X)
                loss = self.criterion(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # train loss for epoch
            train_loss /= len(self.train_loader)
            """
            write(self.out_file, f"train_loss: {train_loss}", print_ = False)
            """
            
            # validate with validate data
            if self.validate_run:
                test_loss = validate_loss(model, self.test_loader, self.criterion)
                test_loss_list.append(test_loss)

            
            # check acceptable and eps for each case
            acceptable, eps, y_pred = acceptable_eps_ypred(self.train_loader, model, self.lr_goal)
            """
            write(self.out_file, f"max eps sqaure: {max(eps)}", False)
            """
            
            # train loss < last epoch train loss ?
            """
            if train loss < train loss of last epoch:
                store model, update lr
            else:
                if lr < lr_bound:
                    break
                restore model, update lr
            """
            if train_loss < loss_old:
                optimizer.param_groups[0]["lr"] *= 1.2
                torch.save(model, temp_save_path)
                train_loss_list.append(train_loss)
                loss_old = train_loss
                write(self.out_file, "Save model and lr increase", False)
            else:
                if optimizer.param_groups[0]['lr'] < self.lr_bound:
                    write(self.out_file, f"non acceptable module at max eps {max(eps)}", False)
                    return acceptable, model, train_loss_list, test_loss_list            
                else:
                    write(self.out_file, "Restore model and lr decrease", False)
                    model = torch.load(temp_save_path)
                    optimizer.param_groups[0]["lr"] *= 0.8
        

        return acceptable, model, train_loss_list, test_loss_list
    
    """
    model: self.model, optimizer: define in function
    """
    def regularising_EU_LG_UA(self):  
        """
        From accpetable to acceptable.
        1. If acceptable: stop
        2. If train loss < last epoch train loss but not acceptable, restore model and stop
        3. If train loss > last epoch loss, lr < lr_bound, restore model and stop
        """
        # define regularising
        # L7 p 47
        # Classmate p 13?

        # some set
        self.model.to(device).train()     # Enter Train Mode
        temp_save_path = "_temp/reg.pth"
        torch.save(self.model, temp_save_path)
        train_loss_list = [] 
        test_loss_list = []
        loss_old = validate_loss(self.model, self.train_loader, self.criterion)
        acceptable, eps, y_pred = acceptable_eps_ypred(self.train_loader, self.model, self.lr_goal)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr_rate)

        # train 
        for epoch in range(self.epochs):

            train_loss = 0

            # forward operation
            for _, (X, y) in enumerate(self.train_loader):                              
                X, y = X.to(device), y.to(device)  
                optimizer.zero_grad()
                preds = self.model(X)
                loss = self.criterion(preds, y) + self.reg(self.hidden_dim, X.shape[0])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # train loss             
            train_loss /= len(self.train_loader)
            train_loss_list.append(train_loss)

            # print
            """
            write(self.out_file, f"train_loss: {train_loss}", False)
            write(self.out_file, f"regular term: {self.reg(self.hidden_dim, X.shape[0])}", False)
            """

            # Validate
            if self.validate_run:
                test_loss = validate_loss(self.model, self.test_loader, self.criterion)
                test_loss_list.append(test_loss)
            
            # NOTE: stopping criteria 1, if acceptable, stop
            acceptable, eps, y_pred = acceptable_eps_ypred(self.train_loader, self.model, self.lr_goal)
            
            # train loss and loss old
            if train_loss < loss_old:
                """
                write(self.out_file, "Save model and lr increase", False) 
                """               
                # stop 2: if max eps > lr goal 
                # NOTE: also acceptable, since restore from previous acceptable model
                if max(eps) >= self.lr_goal:
                    self.model = torch.load(temp_save_path)
                    return train_loss_list, test_loss_list
                else:
                    optimizer.param_groups[0]["lr"] *= 1.2
                    torch.save(self.model, temp_save_path)
                    loss_old = train_loss
                
            else:
                self.model = torch.load(temp_save_path)
                # stop 3. if lr too small
                # NOTE: also acceptable, since restore from previous acceptable model
                if optimizer.param_groups[0]['lr'] < self.lr_bound:
                    return train_loss_list, test_loss_list            
                else:
                    write(self.out_file, "Restore model and lr decrease", False)
                    optimizer.param_groups[0]["lr"] *= 0.8

        return train_loss_list, test_loss_list  

    def reg(self, p, m):
        # reg term
        """
        Get the reg term
        1. p: first hidden node size
        2. m: input sample size (batch size)
        """
        reg_lambda = torch.tensor((0.001 / (p+1 + p*(m+1)))).to(device)
        reg = 0

        names = []
        for name, tensor in self.model.named_parameters():
            names.append(name)
        for i in names:
            weights = self.model.state_dict()[i]
            reg += weights.square().sum()
        return reg*reg_lambda

    def trim_model_nodes(self, k):
        # del nodes k
        """
        Trim the model of kth node
        1. model: model to trimmed
        2. k: node to be trimmed
        """
        trim_model = TwoLayerNet(self.input_dim, self.hidden_dim-1, 1).to(device)
        param = self.model.state_dict()
        delete_status = []
        # Update nodes (del)
        for name in param:
            weights = param[name]

            # k start from 1, the k-1 row in weights will be delete
            if name == "layer_1.weight":
                trim_model.layer_1.weight.data  = torch.cat([weights[:k-1, :], weights[k:, :]], dim=0)
                if False in (trim_model.layer_1.weight.data == torch.cat([weights[:k-1, :], weights[k:, :]], dim=0)):
                    delete_status.append(False)
                else:
                    delete_status.append(True)
            elif name == "layer_1.bias":
                trim_model.layer_1.bias.data = torch.cat([weights[:k-1], weights[k:]], dim=0)
                if False in (trim_model.layer_1.bias.data == torch.cat([weights[:k-1], weights[k:]], dim=0)):
                    delete_status.append(False)
                else:
                    delete_status.append(True)

            elif name == "layer_2.weight":
                trim_model.layer_2.weight.data = torch.cat([weights[:, :k-1], weights[:, k:]], dim=1)
                if False in (trim_model.layer_2.weight.data == torch.cat([weights[:, :k-1], weights[:, k:]], dim=1)):
                    delete_status.append(False)
                else:
                    delete_status.append(True)
            elif name == "layer_2.bias":
                pass
                
        """
        if False not in delete_status:
            write(self.out_file, "Try trim model: Copy model and delete nodes success", False)
        else:
            write(self.out_file, "Try trim model: Copy model and delete nodes error", False)
        """
            
        return trim_model
    
    def reorganising(self):
        # 1. Check nodes: if delete the node, the model trained by weight tuning module 
        #   still can be accepted, delete the node
        # 2. Store the final model in: final_model/Reorg
        """
        self.model: model to be reorganised. main model.
        trim_model: model trimmed from regular module model. 
                    If acceptable, self.model = trim_model (no trained with weight tune).
        """
        # Initialise: check if accetable SLFN (wt.pth) exist               
        write(self.out_file, '-----> reorganising')
        if self.hidden_dim == 1:
            print('only 1 hidden node')
            return 

        # the k node to be check
        # k = 1 
        # while k<=self.hidden_dim:
        random_skip = True if self.hidden_dim > 50 else False
        for k in tqdm(range(self.hidden_dim)):
            if random_skip:
                options = [True, False]
                probabilities = [0.6, 0.4]
                skip = random.choices(options, probabilities)[0]
                if skip:
                    continue
            k += 1
            """
            write(self.out_file, f"{str(self.model)}", False)            
            write(self.out_file, f"    --> Start regularising_EU_LG_UA", False)
            """

            # --------------------
            # regularizing NOTE: from accpetable to acceptable
            train_loss_list, test_loss_list = self.regularising_EU_LG_UA()  # train self.model          
            acceptable, eps, y_pred = acceptable_eps_ypred(self.train_loader, self.model, self.lr_goal)
            assert acceptable, 'Wrong in regularise'
            # --------------------
            
            # --------------------
            # A new model trimmed from regualr module model
            # Use the trim model in weight tuning
            trim_model = self.trim_model_nodes(k)
            acceptable, trim_model, train_loss_list, test_loss_list = \
                self.module_weight_EU_LG_UA(trim_model)            
            if acceptable == True: # and train_loss_list[-1] < loss_reg:
                write(self.out_file, "!!!Trim node from the model, hidden nodes decrease by 1!!!\n")
                self.hidden_dim-=1
                self.model = trim_model
                if self.hidden_dim == 1:
                    break
            # --------------------
        
        # Store all model, not just state dict
        torch.save(self.model, 'acceptable/Reorg.pth')


    