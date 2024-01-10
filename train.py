import torch
import torch.nn as nn
from early_stopping import EarlyStopping
import numpy as np
import time
import os
import wandb
from module.loss import Myloss
import torch.optim as optim
class opt_and_cri_functions:
    def __init__(self,model,learningRate):
        self.criterion = Myloss()
        self.optimizer = optim.Adagrad(model.parameters(), lr = learningRate)

def training_validation(model,epoch_sum,train_loader,val_loader,learning_rate,patience,exp_index,model_folder_directory, DEVICE):
    time_start = time.time()
    
    ocfunction = opt_and_cri_functions(model,learning_rate)
    optimizer = ocfunction.optimizer
    criterion = ocfunction.criterion

    all_batch_train_losses = []
    all_batch_val_losses = []
    all_epoch_train_losses = []
    all_epoch_val_losses = []

    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in range(epoch_sum):
        # ==============training mode==================
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'train')

            batch_loss = criterion(y_pre, y.to(DEVICE))

            batch_loss.backward()
            optimizer.step()
            all_batch_train_losses.append(batch_loss.item())
        
        epoch_train_loss = np.average(all_batch_train_losses)

        # validation==========================
        model.eval()
        for i, (x, y) in enumerate(val_loader):
            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'train')

            batch_loss = criterion(y_pre, y.to(DEVICE))
            all_batch_val_losses.append(batch_loss.item())
        
        epoch_val_loss = np.average(all_batch_val_losses)

        wandb.log({"train_loss": epoch_train_loss,"val_loss":epoch_val_loss})
        all_epoch_train_losses.append(epoch_train_loss)
        all_epoch_val_losses.append(epoch_val_loss)

        epoch_len = len(str(epoch_sum))

        print_msg = (f'round:{exp_index+1}:[{epoch:>{epoch_len}}/{epoch_sum::>{epoch_len}}]'+
                     f'train_loss:{epoch_train_loss:.5f}' + ' '
                     f'valid_loss:{epoch_val_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        all_batch_train_losses = []
        all_batch_val_losses = []

        # early _stopping needs the validation loss to check if if has decreased
        # and if it has, it will make a checkpoint of the current model. Note that early stopping will only store the model with the best validation loss in checkpoint.pt
        early_stopping(epoch_val_loss, model)
        # when it reaches the requirements of stopping,early——stop will be set as True
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # time consumed in one experiment
    time_end = time.time()
    duaration  = time_end - time_start
    print("The training took %.2f"%(duaration/60)+ "mins.")
    time_start = time.asctime(time.localtime(time_start))
    time_end = time.asctime(time.localtime(time_end))
    print("The starting time was ", time_start)
    print("The finishing time was ", time_end)


    if os.path.exists(model_folder_directory) == False:
        os.makedirs(model_folder_directory)
    else:
        pass

    # save the model with the best validation loss
    save_model_time = time.strftime("%Y%m%d_%H%M%S")
    model_name = 'model_params'+save_model_time+"_"+str(exp_index+1)
    torch.save(model.state_dict(),model_folder_directory+'/'+model_name+'.pkl')


    return model_name, model, all_epoch_train_losses, all_epoch_val_losses