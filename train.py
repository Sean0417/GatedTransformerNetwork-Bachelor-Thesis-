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
from evaluation import get_acc

class opt_and_cri_functions:
    def __init__(self,model,learningRate,optimizer_name):
        self.criterion = Myloss()
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr = learningRate)
        elif optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad(model.parameters(), lr = learningRate)
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(model.parameters(), lr = learningRate)



def training_validation(model,epoch_sum,train_loader,val_loader,test_loader,learning_rate,patience,exp_index,model_folder_directory, DEVICE, optimizer_name):
    time_start = time.time()
    
    ocfunction = opt_and_cri_functions(model,learning_rate, optimizer_name)
    optimizer = ocfunction.optimizer
    criterion = ocfunction.criterion

    all_batch_train_losses = []
    all_batch_val_losses = []
    all_batch_train_accs = []
    all_batch_val_accs = []
    all_epoch_train_losses = []
    all_epoch_val_losses = []
    
    all_epoch_train_accs = []
    all_epoch_val_accs = []

    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in tqdm(range(epoch_sum), desc = "Training and validation", unit = 'epoch'):
        # ==============training mode==================
        model.train()
        for i, (x, y) in tqdm(enumerate(train_loader), desc = "Trainning_loader", unit='batch'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'train')
            _, label_index = torch.max(y_pre.data, dim=-1)
            batch_loss = criterion(y_pre, y.to(DEVICE))
            
            batch_loss.backward()
            optimizer.step()
            #  ===calculate acc===
            correct = (label_index == y.long()).sum().item()
            total = label_index.shape[0]
            batch_train_acc = round((100 * correct / total), 2)
            
            all_batch_train_accs.append(batch_train_acc)
            all_batch_train_losses.append(batch_loss.item())
        
        epoch_train_loss = np.average(all_batch_train_losses)
        epoch_train_acc = np.average(all_batch_train_accs)

        # validation==========================
        model.eval()
        for i, (x, y) in tqdm(enumerate(val_loader), desc = 'validation', unit= 'batch'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'test')

            _, label_index = torch.max(y_pre.data, dim=-1)
             #  ===calculate acc===
            correct = (label_index == y.long()).sum().item()
            total = label_index.shape[0]
            batch_val_acc = round((100 * correct / total), 2)
            all_batch_val_accs.append(batch_val_acc)

            batch_loss = criterion(y_pre, y.to(DEVICE))
            all_batch_val_losses.append(batch_loss.item())
        
        epoch_val_loss = np.average(all_batch_val_losses)
        epoch_val_acc = np.average(all_batch_val_accs)

        wandb.log({"train_loss": epoch_train_loss,"val_loss":epoch_val_loss, "train_acc":epoch_train_acc, "val_acc":epoch_val_acc})
        all_epoch_train_losses.append(epoch_train_loss)
        all_epoch_train_accs.append(epoch_train_acc)
        all_epoch_val_losses.append(epoch_val_loss)
        all_epoch_val_accs.append(epoch_val_acc)


        # epoch_train_acc, epoch_train_precision, epoch_train_recall, epoch_train_F1, epoch_train_label_pred, epoch_train_label_true = evaluation(model=model,dataloader=train_loader,DEVICE=DEVICE,flag='train_set')
        # epoch_val_acc,epoch_val_precision, epoch_val_recall, epoch_val_F1, epoch_val_label_pred, epoch_val_label_true = evaluation(model=model,dataloader=val_loader,DEVICE=DEVICE, flag='val_set')
        
        # model_name_during_training = "during_epoch_"+"exp_"+str(exp_index)+"_"+"epoch_"+str(epoch)+"_"

        # plot_Confusion_Matrix(epoch_train_label_true, epoch_train_label_pred, model_name=model_name_during_training,flag="train_set")
        # plot_Confusion_Matrix(epoch_val_label_true, epoch_val_label_pred, model_name=model_name_during_training,flag="test_set")
        
       

        # epoch_test_acc, epoch_test_precision, epoch_test_recall, epoch_test_F1, epoch_test_label_pred, epoch_test_label_true = evaluation(model=model, dataloader=test_loader,DEVICE=DEVICE)
        
        # plot_Confusion_Matrix(epoch_test_label_true, epoch_test_label_pred, model_name=model_name_during_training,flag="test_set")
       
        # print("y_pred:",epoch_test_label_pred)
        # print("y_true:", epoch_test_label_true)
        
        # wandb.log({"test_acc:": epoch_test_acc, 'test_precision':epoch_test_precision, "test_recall": epoch_test_recall, "test_F1": epoch_test_F1})
        all_epoch_train_accs.append(epoch_train_acc)
        all_epoch_val_accs.append(epoch_val_acc)



        epoch_len = len(str(epoch_sum))

        print_msg = (f'round:{exp_index+1}:[{epoch:>{epoch_len}}/{epoch_sum::>{epoch_len}}]'+
                     f'train_loss:{epoch_train_loss:.5f}' + " " + 'validation_loss'+f':{epoch_val_loss:.5f}')
        print(print_msg)



        # clear lists to track next epoch
        all_batch_train_losses = []
        all_batch_val_losses = []
        all_batch_train_accs = []
        all_batch_val_accs = []

        # early _stopping needs the validation loss to check if if has decreased
        # and if it has, it will make a checkpoint of the current model. Note that early stopping will only store the model with the best validation loss in checkpoint.pt
        early_stopping(epoch_val_loss, model)
        # when it reaches the requirements of stopping,early——stop will be set as True
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print("=================")

    # time consumed in one experiment
    time_end = time.time()
    duaration  = time_end - time_start
    print("The training took %.2f"%(duaration/60)+ "mins.")
    time_start = time.asctime(time.localtime(time_start))
    time_end = time.asctime(time.localtime(time_end))
    print("The starting time was ", time_start)
    print("The finishing time was ", time_end)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    if os.path.exists(model_folder_directory) == False:
        os.makedirs(model_folder_directory)
    else:
        pass

    # save the model with the best validation loss
    save_model_time = time.strftime("%Y%m%d_%H%M%S")
    model_name = 'after_earlyStopping_'+save_model_time+"_"+str(exp_index+1)
    torch.save(model.state_dict(),model_folder_directory+'/'+model_name+'.pkl')




    return model_name, model, all_epoch_train_losses, all_epoch_val_losses