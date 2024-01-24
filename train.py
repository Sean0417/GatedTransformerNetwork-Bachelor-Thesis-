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

class opt_and_cri_functions:
    def __init__(self,model,learningRate,optimizer_name):
        self.criterion = Myloss()
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr = learningRate)
        elif optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad(model.parameters(), lr = learningRate)
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(model.parameters(), lr = learningRate)

def plot_Confusion_Matrix(y_true, y_pred, model_name, flag="test_set"):
    cm = confusion_matrix(y_true, y_pred)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    confusion_matrix_plt_dir = "confusion_matrix/"
    if flag == "test_set":
        if os.path.exists(confusion_matrix_plt_dir):
            plt.savefig(confusion_matrix_plt_dir+"Test_set_confusionMatrix_"+model_name+'.png',format='png',dpi= 200)
        else:
            os.makedirs(confusion_matrix_plt_dir)
            plt.savefig(confusion_matrix_plt_dir+"Test_set_confusionMatrix_"+model_name+'.png',format='png',dpi= 200)
    elif flag == "train_set":
        if os.path.exists(confusion_matrix_plt_dir):
            plt.savefig(confusion_matrix_plt_dir+"Train_set_confusionMatrix_"+model_name+'.png',format='png',dpi= 200)
        else:
            os.makedirs(confusion_matrix_plt_dir)
            plt.savefig(confusion_matrix_plt_dir+"Train_set_confusionMatrix_"+model_name+'.png',format='png',dpi= 200)
    elif flag == "val_set":
        if os.path.exists(confusion_matrix_plt_dir):
            plt.savefig(confusion_matrix_plt_dir+"Val_set_confusionMatrix_"+model_name+'.png',format='png',dpi= 200)
        else:
            os.makedirs(confusion_matrix_plt_dir)
            plt.savefig(confusion_matrix_plt_dir+"Val_set_confusionMatrix_"+model_name+'.png',format='png',dpi= 200)


            
        

def training_validation(model,epoch_sum,train_loader,val_loader,test_loader,learning_rate,patience,exp_index,model_folder_directory, DEVICE, optimizer_name):
    time_start = time.time()
    
    ocfunction = opt_and_cri_functions(model,learning_rate, optimizer_name)
    optimizer = ocfunction.optimizer
    criterion = ocfunction.criterion

    all_batch_train_losses = []
    all_batch_val_losses = []
    all_epoch_train_losses = []
    all_epoch_val_losses = []
    
    all_epoch_train_accs = []
    all_epoch_val_accs = []

    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in tqdm(range(epoch_sum), desc = "Training and validation", unit = 'epoch'):
        # ==============training mode==================
        model.train()
        for i, (x, y) in tqdm(enumerate(train_loader), desc = "Trainning_loader", unit='batch'):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'train')

            batch_loss = criterion(y_pre, y.to(DEVICE))

            batch_loss.backward()
            optimizer.step()
            all_batch_train_losses.append(batch_loss.item())
        
        epoch_train_loss = np.average(all_batch_train_losses)

        # validation==========================
        model.eval()
        for i, (x, y) in tqdm(enumerate(val_loader), desc = 'validation', unit= 'batch'):
            y_pre, _, _, _, _, _, _ = model(x.to(DEVICE), 'test')

            batch_loss = criterion(y_pre, y.to(DEVICE))
            all_batch_val_losses.append(batch_loss.item())
        
        epoch_val_loss = np.average(all_batch_val_losses)

        wandb.log({"train_loss": epoch_train_loss,"val_loss":epoch_val_loss})
        all_epoch_train_losses.append(epoch_train_loss)
        all_epoch_val_losses.append(epoch_val_loss)

        epoch_train_acc, epoch_train_precision, epoch_train_recall, epoch_train_F1, epoch_train_label_pred, epoch_train_label_true = evaluation(model=model,dataloader=train_loader,DEVICE=DEVICE,flag='train_set')
        epoch_val_acc,epoch_val_precision, epoch_val_recall, epoch_val_F1, epoch_val_label_pred, epoch_val_label_true = evaluation(model=model,dataloader=val_loader,DEVICE=DEVICE, flag='val_set')
        epoch_test_acc, epoch_test_precision, epoch_test_recall, epoch_test_F1, epoch_test_label_pred, epoch_test_label_true = evaluation(model=model, dataloader=test_loader,DEVICE=DEVICE)
        model_name_during_training = "during_epoch_"+"epoch_"+"exp_"+str(exp_index)+"_"+str(epoch)+"_"
        plot_Confusion_Matrix(epoch_test_label_true, epoch_test_label_pred, model_name=model_name_during_training,flag="test_set")
        plot_Confusion_Matrix(epoch_train_label_true, epoch_train_label_pred, model_name=model_name_during_training,flag="train_set")
        print("y_pred:",epoch_test_label_pred)
        print("y_true:", epoch_test_label_true)
        
        wandb.log({"train_acc:": epoch_train_acc, 'train_precision':epoch_train_precision, "train_recall": epoch_train_recall, "train_F1": epoch_train_F1})
        wandb.log({"val_acc:": epoch_val_acc, 'val_precision':epoch_val_precision, "val_recall": epoch_val_recall, "val_F1": epoch_val_F1})
        wandb.log({"test_acc:": epoch_test_acc, 'test_precision':epoch_test_precision, "test_recall": epoch_test_recall, "test_F1": epoch_test_F1})
        all_epoch_train_accs.append(epoch_train_acc)
        all_epoch_val_accs.append(epoch_val_acc)



        epoch_len = len(str(epoch_sum))

        print_msg = (f'round:{exp_index+1}:[{epoch:>{epoch_len}}/{epoch_sum::>{epoch_len}}]'+
                     f'train_loss:{epoch_train_loss:.5f}' + " " + 'validation_loss'+f':{epoch_val_loss:.5f}')
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
        print("=================")

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
    model_name = 'after_earlyStopping_'+save_model_time+"_"+str(exp_index+1)
    torch.save(model.state_dict(),model_folder_directory+'/'+model_name+'.pkl')
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    # confusion matrix for test set with best accuracy.
    test_acc, test_precision, test_recall, test_F1, test_label_pred, test_label_true = evaluation(model=model, dataloader=test_loader,DEVICE=DEVICE)
    train_acc, train_precision, train_recall, train_F1, train_label_pred, train_label_true = evaluation(model=model, dataloader=train_loader, DEVICE=DEVICE)
    plot_Confusion_Matrix(test_label_true, test_label_pred, model_name, flag="test_set")
    plot_Confusion_Matrix(train_label_true, train_label_pred, model_name, flag="train_set")


    return model_name, model, all_epoch_train_losses, all_epoch_val_losses