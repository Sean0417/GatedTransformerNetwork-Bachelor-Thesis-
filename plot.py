import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
def plot_learning_curve(train_loss, val_loss, plot_folder_dir, model_name):
    # visualize the loss as the network trained
    plt.figure()
    plt.plot(range(1,len(train_loss)+1),train_loss, label= 'Train Loss')
    plt.plot(range(1,len(val_loss)+1),val_loss,label='Validation Loss')

    # find postion of lowest validation loss
    minposs = val_loss.index(min(val_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.title("Learning_curve")
    plt.legend(loc='best')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid(True)

    if os.path.exists(plot_folder_dir):
        plt.savefig(plot_folder_dir+'/'+"learning_curve_"+model_name+'.png',format='png',dpi= 200)
    else:
        os.makedirs(plot_folder_dir)
        plt.savefig(plot_folder_dir+'/'+"learning_curve_"+model_name+'.png',format='png',dpi= 200)

    # wandb.log({"best_train_validation_curve":wandb.Plotly(plt.gcf())}) # print the learning curve on wandb
    plt.close()

def plot_prediction_curve(y, y_predict, test_loss,plot_folder_dir,is_train,test_model_directory=""):
    plt.figure()
    plt.plot(y[500:600], 'b', label='ground truth')
    plt.plot(y_predict[500:600],'r',label = 'prediction')
    plt.title('Ozone predictions with test loss='+str(test_loss))
    plt.xlabel('time')
    plt.ylabel('Ozone')
    plt.xticks(np.arange(0, 100, step = 10))
    plt.legend(loc='best')
    
    if is_train == True:
        if os.path.exists(plot_folder_dir):
            plt.savefig(plot_folder_dir+'/best_result.png',format='png',dpi=200)
        else:
            os.makedirs(plot_folder_dir)
            plt.savefig(plot_folder_dir+'/best_result.png',format='png',dpi=200)
    else:
        if os.path.exists(plot_folder_dir):
            plt.savefig(plot_folder_dir+'/'+"prediction_curve_"+test_model_directory.split('.')[0]+'.png',format='png',dpi=200)
        else:
            os.makedirs(plot_folder_dir)
            plt.savefig(plot_folder_dir+'/'+"prediction_curve_"+test_model_directory.split('.')[0]+'.png',format='png',dpi=200)

    # wandb.log({"plot_prediction_curve":wandb.Plotly(plt.gcf())}) # print the plot of the prediction curve on wandb
    plt.close()

def plot_Confusion_Matrix(y_true, y_pred, full_param_name, flag="test_set"):
    cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    plt.title(full_param_name)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plot_folder_dir = 'Confusion_Matrix/'+full_param_name+ "_confusion_matrix/"

    plot_time = time.strftime("%Y%m%d_%H%M%S")
    if os.path.exists(plot_folder_dir):
        plt.savefig(plot_folder_dir+'/'+flag+"_Confusion_Matrix_"+plot_time+'.png',format='png',dpi= 200)
    else:
        os.makedirs(plot_folder_dir)
        plt.savefig(plot_folder_dir+'/'+flag+"_Confusion_Matrix_"+plot_time+'.png',format='png',dpi= 200)

def define_type(dataloader, model, DEVICE, prediction_type):
    score_input = np.zeros([8,8,8])
    score_channel = np.zeros([8,8,8])
    if prediction_type == "TP":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre)
                
                if label_index[0] == int(y[0]) and label_index[0] == 1:
                        break
            return score_input, score_channel
    elif prediction_type == "TN":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre)
                
                if label_index[0] == int(y[0]) and label_index[0] == 0:
                    break
            return score_input, score_channel
    elif prediction_type == "FP":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre)
                
                if label_index[0] != int(y[0]) and label_index[0] == 1:
                    break
            return score_input, score_channel
    elif prediction_type == "FN":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre)
                
                if label_index[0] != int(y[0]) and label_index[0] == 0:
                    break
            return score_input, score_channel
    else:
        print("please enter the correct form of the prediction_type")
            
def plot_heat_map(dataloader,model,file_name,DEVICE, prediction_type):
    score_input, score_channel = define_type(dataloader=dataloader, 
                                                   model=model, 
                                                   DEVICE=DEVICE,
                                                   prediction_type=prediction_type)
    if score_channel != np.zeros([8,8,8]):
        plot_time = time.strftime("%Y%m%d_%H%M%S")
        score_input = score_input.detach().cpu().numpy()
        score_channel = score_channel.detach().cpu().numpy()

        # plot score_input_
        fig_input, axes_input = plt.subplots(4, int(score_input.shape[0]/4), figsize=(20, 20))
        for i in range(score_input.shape[0]):
            ax = axes_input[i // 2, i % 2]
            sns.heatmap(score_input[i], ax=ax, cmap='Blues')
            ax.set_title(f'Stepwise Attention Heatmap for {prediction_type} on Head {i+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
        folder_name = "Heat_Map/heatmap_score_input"+file_name
        plt.tight_layout()
        if os.path.exists(folder_name):
            plt.savefig(folder_name+'/'+"Heatmap_"+prediction_type+"_"+plot_time+'.png',format='png',dpi= 200)
        else:
            os.makedirs(folder_name)
            plt.savefig(folder_name+'/'+"Heatmap_"+prediction_type+"_"+plot_time+'.png',format='png',dpi= 200)
        # plot score_channel
        fig_channel, axes_channel = plt.subplots(4, int(score_channel.shape[0]/4), figsize=(20, 20))
        for i in range(score_channel.shape[0]):
            ax = axes_channel[i // 2, i % 2]
            sns.heatmap(score_channel[i], ax=ax, cmap='Blues')
            ax.set_title(f'Channelwise Attention Heatmap for {prediction_type} on Head {i+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
        folder_name = "Heat_Map/heatmap_score_channel"+file_name
        if os.path.exists(folder_name):
            plt.savefig(folder_name+'/'+"Heatmap_"+prediction_type+"_"+plot_time+'.png',format='png',dpi= 200)
        else:
            os.makedirs(folder_name)
            plt.savefig(folder_name+'/'+"Heatmap_"+prediction_type+"_"+plot_time+'.png',format='png',dpi= 200)
    
    else:
        print(f"There is no heatmap under the {prediction_type} circumstance")


    