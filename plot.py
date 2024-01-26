import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_Confusion_Matrix(y_true, y_pred, flag="test_set"):
    cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plot_folder_dir = "3_1confusion_matrix/"

    plot_time = time.strftime("%Y%m%d_%H%M%S")
    if os.path.exists(plot_folder_dir):
        plt.savefig(plot_folder_dir+'/'+flag+"_Confusion_Matrix_"+plot_time+'.png',format='png',dpi= 200)
    else:
        os.makedirs(plot_folder_dir)
        plt.savefig(plot_folder_dir+'/'+flag+"_Confusion_Matrix_"+plot_time+'.png',format='png',dpi= 200)