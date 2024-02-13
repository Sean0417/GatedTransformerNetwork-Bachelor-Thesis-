import torch
import torch.nn as nn
from early_stopping import EarlyStopping
import numpy as np
import time
import os
import wandb
from module.loss import Myloss
import torch.optim as optim
from evaluation import evaluation
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from plot import plot_Confusion_Matrix

class opt_and_cri_functions:
    def __init__(self,model,learningRate,optimizer_name):
        self.criterion = Myloss()
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr = learningRate)
        elif optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad(model.parameters(), lr = learningRate)
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(model.parameters(), lr = learningRate)



def training_validation(model,epoch_sum,train_loader,val_loader,test_loader,learning_rate,patience,exp_index,model_folder_directory, DEVICE, optimizer_name,file_name,num_head,num_encoder,d_model,attn_type:str):
    # time_start = time.time()
    
    ocfunction = opt_and_cri_functions(model,learning_rate, optimizer_name)
    optimizer = ocfunction.optimizer
    criterion = ocfunction.criterion

    # all_epoch_train_losses = []
    # all_epoch_val_losses = []
    # all_epoch_train_accs = []
    # all_epoch_val_accs = []

    exp_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    full_param_name = file_name+"_"+f"{attn_type}_d{d_model}_N{num_encoder}_h{num_head}_"+exp_timestamp
    file_name = file_name+"/"+file_name+"_"+f"d_model{d_model}_num_encoder{num_encoder}_num_head{num_head}_"+exp_timestamp+"_"+"checkpoint.pth"
    best_model_path = os.path.join("./saved_models",file_name)
    early_stopping = EarlyStopping(patience=patience,
                                path=best_model_path,
                                verbose=True)

    for epoch in tqdm(range(epoch_sum), desc = "Training and validation", unit = 'epoch'):

        all_batch_train_losses = []
        all_batch_val_losses = []
        all_batch_train_accs = []
        all_batch_val_accs = []

        # ==============training loop==================
        model.train()
        for i, (x, y) in tqdm(enumerate(train_loader), desc = "Trainning_loader", unit='batch'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'train')

            # train_batch_loss
            batch_loss = criterion(y_pre, y.to(DEVICE))
            all_batch_train_losses.append(batch_loss.item())

            # train_batch_acc
            _, label_index = torch.max(y_pre.data, dim=-1)
            correct = (label_index == y.long()).sum().item()
            total = label_index.shape[0]
            batch_train_acc = round((100 * correct / total), 2)
            all_batch_train_accs.append(batch_train_acc)

            batch_loss.backward()
            optimizer.step()
            
        epoch_train_loss = np.average(all_batch_train_losses)
        epoch_train_acc = np.average(all_batch_train_accs)

        # validation==========================
        model.eval()
        for i, (x, y) in tqdm(enumerate(val_loader), desc = 'validation', unit= 'batch'):
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_pre, _, score_input, score_channel, _, _, _ = model(x.to(DEVICE), 'test')
            # print("score_input:",score_input.shape)
            # print("score_channel:", score_channel.shape)
            

            # val_bacth_loss
            batch_loss = criterion(y_pre, y.to(DEVICE))
            all_batch_val_losses.append(batch_loss.item())
            
            # val_bacth_acc
            _, label_index = torch.max(y_pre.data, dim=-1)
            correct = (label_index == y.long()).sum().item()
            total = label_index.shape[0]
            batch_val_acc = round((100 * correct / total), 2)
            all_batch_val_accs.append(batch_val_acc)
           
        
        epoch_val_loss = np.average(all_batch_val_losses)
        epoch_val_acc = np.average(all_batch_val_accs)

        # print epoch loss, acc
        epoch_len = len(str(epoch_sum))
        print_msg = (f'round:{exp_index+1}:[{epoch:>{epoch_len}}/{epoch_sum::>{epoch_len}}   learning_rate={learning_rate}]'+
                    f'train_loss:{epoch_train_loss:.5f}' + " " + 'validation_loss'+f':{epoch_val_loss:.5f}',
                    f'train_acc:{epoch_train_acc:.5f}' + " " + 'validation_acc'+f':{epoch_val_acc:.5f}',
                     )
        print(print_msg)

        wandb.log({"train_loss": epoch_train_loss,
                "val_loss":epoch_val_loss, 
                "train_acc":epoch_train_acc, 
                "val_acc":epoch_val_acc})

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(best_model_path))

    return model,full_param_name