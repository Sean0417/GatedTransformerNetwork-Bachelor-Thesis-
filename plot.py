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

def plot_Confusion_Matrix(y_true, y_pred, file_name, full_param_name, flag="test_set"):
    cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    # plt.title(f"confusion matrix on {file_name}")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size":20},cbar_kws={"shrink": 0.5})
    plt.xlabel('Predicted',fontsize=12)
    plt.ylabel('True',fontsize=12)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plot_folder_dir = 'Confusion_Matrix/'+file_name+ "_confusion_matrix/"

    if os.path.exists(plot_folder_dir):
        plt.savefig(plot_folder_dir+'/'+flag+'_'+full_param_name+'.png',format='png',dpi= 200)
    else:
        os.makedirs(plot_folder_dir)
        plt.savefig(plot_folder_dir+'/'+flag+full_param_name+'.png',format='png',dpi= 200)

def define_type(dataloader, model, DEVICE, prediction_type):
    score_input=np.zeros([8,8,8])
    score_channel=np.zeros([8,8,8])
    flag=0
    if prediction_type == "TP":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] == int(y[0]) and label_index[0] == 1:
                        flag=1
                        break
            return score_input, score_channel,flag
    elif prediction_type == "TN":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] == int(y[0]) and label_index[0] == 0:
                    flag=1
                    break
            return score_input, score_channel,flag
    elif prediction_type == "FP":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] != int(y[0]) and label_index[0] == 1:
                    flag=1
                    break
            return score_input, score_channel,flag
    elif prediction_type == "FN":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] != int(y[0]) and label_index[0] == 0:
                    flag=1
                    break
            return score_input, score_channel, flag
    else:
        print("please enter the correct form of the prediction_type")
            
def plot_heat_map(dataloader,model,file_name,full_param_name,DEVICE, prediction_type):
    score_input, score_channel, flag= define_type(dataloader=dataloader,model=model, DEVICE=DEVICE,prediction_type=prediction_type)
    num_heads = score_input.shape[0]
    # calculate the rows, two
    num_rows = np.ceil(num_heads / 2).astype(int)
    plot_time = time.strftime("%Y%m%d_%H%M%S")
    score_input = score_input.detach().cpu().numpy()
    score_channel = score_channel.detach().cpu().numpy()

    if flag==1:
        if num_heads >= 4:

            # plot score_input_
            fig_input, axes_input = plt.subplots(num_rows, 2, figsize=(30, num_rows*15))
           
            for i in range(score_input.shape[0]):
                # print(score_input[i].shape)
                ax = axes_input[i // 2, i % 2]
                # h=sns.heatmap(score_input[i],vmax=0.018, vmin=0.002, ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                h=sns.heatmap(score_input[i], ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                ax.set_title(f'{prediction_type} on Head {i+1}', fontsize = 40)
                cb=h.figure.colorbar(h.collections[0])
                cb.ax.tick_params(labelsize=40) #set the size of colorbar
                ax.tick_params(labelsize=20)
            
            folder_name = "Heat_Map/heatmap_score_input"+file_name
            plt.tight_layout()
            if os.path.exists(folder_name):
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
            else:
                os.makedirs(folder_name)
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
                
            
            
            
            # plot score_channel
            
            fig_channel, axes_channel= plt.subplots(num_rows, 2, figsize=(30, num_rows*15))

            for i in range(score_channel.shape[0]):
                ax = axes_channel[i // 2, i % 2]
                # h2=sns.heatmap(score_channel[i], vmax=0.6, vmin=0.4,ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                h2=sns.heatmap(score_channel[i], ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                ax.set_title(f'{prediction_type} Head {i+1}',fontsize=50)
                cb=h2.figure.colorbar(h2.collections[0])
                cb.ax.tick_params(labelsize=45) # set the size of the color bar
                ax.tick_params(labelsize=45) # set the size of the label
                

            folder_name = "Heat_Map/heatmap_score_channel"+file_name
            if os.path.exists(folder_name):
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
            else:
                os.makedirs(folder_name)
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
        
        
        elif num_heads == 2:
            # plot score_input_
            fig_input, axes_input = plt.subplots(num_rows, 2, figsize=(30, num_rows*15))
            if num_heads == 1:
               axes_input = [axes_input]

            for i in range(score_input.shape[0]):
                ax = axes_input[i]
                # sns.heatmap(score_input[i],vmax=0.018, vmin=0.002, ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                sns.heatmap(score_input[i],ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                ax.set_title(f'Stepwise Attention Heatmap for {prediction_type} on Head {i+1}')
                ax.set_xlabel('Key')
                ax.set_ylabel('Query') 
                        
            folder_name = "Heat_Map/heatmap_score_input"+file_name
            plt.tight_layout()
            if os.path.exists(folder_name):
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
            else:
                os.makedirs(folder_name)
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
 


            fig_channel, axes_channel = plt.subplots(num_rows, 2, figsize=(20, num_rows*5))
            # if the axes is not 2-dimensionl（whennum_heads < 2），change it into 2 dimensions
            for i in range(score_channel.shape[0]):
                ax = axes_channel[i]
                # sns.heatmap(score_channel[i], vmax=0.6, vmin=0.4,ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                sns.heatmap(score_channel[i], ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
                ax.set_title(f'Channelwise Attention Heatmap for {prediction_type} on Head {i+1}')
                ax.set_xlabel('Key')
                ax.set_ylabel('Query')

            plt.tight_layout()
            folder_name = "Heat_Map/heatmap_score_channel"+file_name
            if os.path.exists(folder_name):
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
            else:
                os.makedirs(folder_name)
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)

        elif num_heads == 1:

            # plot step-wise
            plt.figure(figsize=(10, 8))
            # sns.heatmap(score_input[0],vmax=0.018, vmin=0.002, ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
            sns.heatmap(score_input[0], ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
            plt.title(f'Stepwise Attention Heatmap for {prediction_type}')
            plt.xlabel("Key")
            plt.ylabel("Query")

            folder_name = "Heat_Map/heatmap_score_input"+file_name
            if os.path.exists(folder_name):
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
            else:
                os.makedirs(folder_name)
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
                
            

            #plot channel wise
            plt.figure(figsize=(10, 8))
            # sns.heatmap(score_channel[0], vmax=0.6, vmin=0.4,ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
            sns.heatmap(score_channel[0], ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
            plt.title(f'Stepwise Attention Heatmap for {prediction_type}')
            plt.xlabel("Key")
            plt.ylabel("Query")

            folder_name = "Heat_Map/heatmap_score_channel"+file_name
            if os.path.exists(folder_name):
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)
            else:
                os.makedirs(folder_name)
                plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'.png',format='png',dpi= 200)





    else:
        print(f"There is no heatmap under the {prediction_type} circumstance")


def define_type_3(dataloader, model, DEVICE, prediction_type):
    score_inputs=[]
    score_channels=[]
    flag=0
    if prediction_type == "TP":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] == int(y[0]) and label_index[0] == 1:
                        score_input = score_input.cpu().data.numpy()
                        score_channel = score_channel.cpu().data.numpy()
                        score_inputs.append(score_input)
                        score_channels.append(score_channel)
                        flag += 1
                if flag == 3:
                        break
            return np.array(score_inputs), np.array(score_channels),flag
    elif prediction_type == "TN":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] == int(y[0]) and label_index[0] == 0:
                        score_input = score_input.cpu().data.numpy()
                        score_channel = score_channel.cpu().data.numpy()
                        score_inputs.append(score_input)
                        score_channels.append(score_channel)
                        flag += 1
                if flag == 3:
                        break
            return np.array(score_inputs), np.array(score_channels),flag
    elif prediction_type == "FP":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] != int(y[0]) and label_index[0] == 1:
                        score_input = score_input.cpu().data.numpy()
                        score_channel = score_channel.cpu().data.numpy()
                        score_inputs.append(score_input)
                        score_channels.append(score_channel)
                        flag += 1
                if flag == 3:
                        break
            return np.array(score_inputs), np.array(score_channels),flag
    elif prediction_type == "FN":
        with torch.no_grad():
            model.eval()
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pre, _, score_input, score_channel, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
                _, label_index = torch.max(y_pre.data, dim=-1)
                
                if label_index[0] != int(y[0]) and label_index[0] == 0:
                        score_input = score_input.cpu().data.numpy()
                        score_channel = score_channel.cpu().data.numpy()
                        score_inputs.append(score_input)
                        score_channels.append(score_channel)
                        flag += 1
                if flag == 3:
                        break
            return np.array(score_inputs), np.array(score_channels),flag
    else:
        print("please enter the correct form of the prediction_type")

def test_plot_heat_map(dataloader,model,file_name,full_param_name,DEVICE, prediction_type):
    score_inputs, score_channels, flag= define_type_3(dataloader=dataloader,model=model, DEVICE=DEVICE,prediction_type=prediction_type)
    num_heads = score_inputs.shape[1]
    print("score_inputs_shape", score_inputs.shape)
    # calculate the rows, two
    num_rows = np.ceil(num_heads / 2).astype(int)
    print("num_rows:",num_rows)
    plot_time = time.strftime("%Y%m%d_%H%M%S")
    # score_input = score_input.detach().cpu().numpy()
    # score_channel = score_channel.detach().cpu().numpy()
    

    # plot score_input_
    j = 0
    for score_input in score_inputs:
        j+=1
        fig_input, axes_input = plt.subplots(num_rows, 2, figsize=(30, num_rows*15))
        for i in range(score_input.shape[0]):
            ax = axes_input[i // 2, i % 2]
            # h=sns.heatmap(score_input[i],vmax=0.018, vmin=0.002, ax=ax, cmap='Blues', square=True, cbar=False)
            h=sns.heatmap(score_input[i],vmax=0.018, vmin=0.002, ax=ax, cmap='Blues', square=True, cbar=False)
            ax.set_title(f'{prediction_type} on Head {i+1}', fontsize = 40)
            # ax.set_xlabel('Key')
            # ax.set_ylabel('Query')
            cb=h.figure.colorbar(h.collections[0])
            cb.ax.tick_params(labelsize=40)
            ax.tick_params(labelsize=20)
            print(str(j)+','+str(i))
        folder_name = "Heat_Map/heatmap_score_input"+file_name
        plt.tight_layout()
        if os.path.exists(folder_name):
            plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'inferred_times_'+str(j)+'.pdf',format='pdf',dpi= 200)
        else:
            os.makedirs(folder_name)
            plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'inferred_times_'+str(j)+'.pdf',format='pdf',dpi= 200)
                
        
    # plot score_channel
            
    j = 0
    for score_channel in score_channels:
        j+=1
        fig_channel, axes_channel= plt.subplots(num_rows, 2, figsize=(30, num_rows*15))
        for i in range(score_channel.shape[0]):
            ax = axes_channel[i // 2, i % 2]
            # h2=sns.heatmap(score_channel[i], vmax=0.6, vmin=0.4,ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
            h2=sns.heatmap(score_channel[i], ax=ax, cmap='Blues',square=True,annot=False,cbar=False)
            ax.set_title(f'{prediction_type} Head {i+1}',fontsize=50)
            # ax.set_xlabel('Key')
            # ax.set_ylabel('Query')
            cb=h2.figure.colorbar(h2.collections[0])
            cb.ax.tick_params(labelsize=45)
            ax.tick_params(labelsize=45)
            print(str(j)+','+str(i))

        folder_name = "Heat_Map/test_heatmap_score_channel"+file_name
        if os.path.exists(folder_name):
            plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'inferred_times_'+str(j)+'.pdf',format='pdf',dpi= 200)
        else:
            os.makedirs(folder_name)
            plt.savefig(folder_name+'/'+"Heatmap_"+full_param_name+'_'+prediction_type+'inferred_times_'+str(j)+'.pdf',format='pdf',dpi= 200)