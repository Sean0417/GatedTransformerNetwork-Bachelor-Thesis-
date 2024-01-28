import torch
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
import os
import wandb
import argparse

from module.transformer import Transformer
from module.loss import Myloss
# from processEEG import EEGDataset
from train import training_validation
from evaluation import evaluation
from utils.random_seed import setup_seed
from evaluation import evaluation
from plot import plot_Confusion_Matrix
from dataset_process import MyDataset
# from test import EEGDatase
# setup_seed(30)

def main(args):
    # 1. data preprocessing
    path = args.path
    plot_folder_dir= args.plot_folder_dir
    model_folder_dir= args.model_folder_dir
    is_train = args.is_train
    print(args.is_train)
    # test_model = args.test_model

    draw_key = 1  # 大于等于draw_key才会保存图像
    file_name = path.split('/')[-1][0:path.split('/')[-1].index('.')]  # 获得文件名字
    # 超参数设置
    EPOCH = args.EPOCH
    BATCH_SIZE = args.BATCH_SIZE
    LR = args.learning_rate
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
    patience = args.patience
    print(f'use device: {DEVICE}')

    train_percentage = args.train_percentage
    validate_percentage = args.validate_percentage
    
    # hyperparameters inside model
    d_model = args.d_model
    d_hidden = args.d_hidden
    q = args.q
    v = args.v
    h = args.head
    N = args.N
    dropout = args.dropout
    pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
    mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask

    # training_hyperparameters
    sliding_window_length = args.sliding_window_length
    num_exps = args.num_exps
    
    # 优化器选择
    optimizer_name = args.optimizer_name

    # split the data into train, validate and test
    # train_dataset = EEGDataset(path=path, dataset='train', train_percentage=train_percentage,validate_percentage=validate_percentage,sliding_window_length=sliding_window_length)
    # test_dataset = EEGDataset(path=path, dataset='test', train_percentage=train_percentage,validate_percentage=validate_percentage,sliding_window_length=sliding_window_length)
    # # validate_dataset = EEGDataset(path=path, dataset='validate',train_percentage=train_percentage,validate_percentage=validate_percentage,sliding_window_length=sliding_window_length)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # =================datasets in GTN=========================================
    train_dataset = MyDataset(path, 'train')
    test_dataset = MyDataset(path, 'test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    # -------------------------------
    DATA_LEN = train_dataset.train_len  # 训练集样本数量
    d_input = train_dataset.input_len  # 时间部数量
    d_channel = train_dataset.channel_len  # 时间序列维度
    d_output = train_dataset.output_len  # 分类类别

    # 维度展示
    print('data structure: [lines, timesteps, features]')
    print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
    # print(f'validation data size:[{train_dataset.validate_len, d_input, d_channel}]')
    print(f'test data size: [{train_dataset.test_len, d_input, d_channel}]')
    print(f'Number of classes: {d_output}')
    
    # 2. training and validation
    # 创建Transformer模型
    # net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
    #                 q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

    if is_train == True:
        print("===================train and validation====================")
        # experiments loop
        for exp_idx in range(num_exps):
                # wandb initialization
            config = dict(learningRate = LR, batch_size = BATCH_SIZE, num_of_epochs = EPOCH,
                        sliding_window_length=sliding_window_length,num_of_experiments = num_exps,
                        optimizer = optimizer_name, head_of_multi_attention=h)
            print(config)
            wandb.init(project=file_name,
                    job_type="training",
                    config=config,
                    reinit=True,
                    )
            # model initialization
            model = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                        q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
            wandb.watch(model,log="all")
            model = training_validation(model=model, epoch_sum=EPOCH, train_loader=train_loader, 
                                                                                            val_loader=val_loader, test_loader=test_loader, learning_rate=LR, patience=patience, exp_index=1, 
                                                                                            model_folder_directory=model_folder_dir, DEVICE=DEVICE,optimizer_name=optimizer_name)
        # execute testing
        test_label_pred,test_label_true = evaluation(model=model, dataloader=test_loader,DEVICE=DEVICE)
        plot_Confusion_Matrix(test_label_true, test_label_pred, file_name, flag="test_set")
            # confusion matrix for test set with best accuracy.
            # test_acc, test_precision, test_recall, test_F1, test_label_pred, test_label_true = evaluation(model=model, dataloader=test_loader,flag="test_set",DEVICE=DEVICE)
            # train_acc, train_precision, train_recall, train_F1, train_label_pred, train_label_true = evaluation(model=model, dataloader=train_loader,flag="train_set", DEVICE=DEVICE)
            # wandb.log({"exp_test_acc":test_acc, "exp_test_precision": test_precision, "exp_text_recall": test_recall, "exp_test_F1":test_F1})
            # wandb.log({"exp_train_acc":test_acc, "exp_train_precision": test_precision, "exp_train_recall": test_recall, "exp_train_F1":test_F1})
            # plot the learning curve
            # plot_learning_curve(train_loss=all_epoch_train_loss,val_loss=all_epoch_val_loss,plot_folder_dir=plot_folder_dir,model_name=model_name)
            # print("round"+str(exp_idx+1)+" has been done")
    else:
        config = dict(learningRate = LR, batch_size = BATCH_SIZE, num_of_epochs = EPOCH,
                        sliding_window_length=sliding_window_length,num_of_experiments = num_exps,
                        optimizer = optimizer_name, head_of_multi_attention=h)
        wandb.init(project=file_name,
        job_type="test",
        config=config,
        reinit=True,
        )
        model = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden, q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(args.given_best_model_path))
        # execute testing
        test_label_pred,test_label_true = evaluation(model=model, dataloader=test_loader,DEVICE=DEVICE)
        plot_Confusion_Matrix(test_label_true, test_label_pred, file_name, flag="test_set")
    
    

    # model =  

    # best_test_loss= float('inf')
    # best_test_loss_model=''

    
    # test_acc, test_precision, test_recall, test_F1, test_label_pred, test_label_true = evaluation(model=model, dataloader=test_loader,flag="test_set",DEVICE=DEVICE)
    # plot_Confusion_Matrix(test_label_true, test_label_pred, model_name, flag="test_set")

    # # use tqdm to beautify the output
    # for file in tqdm(os.listdir(args.model_folder_dir),unit="model file"):
    #     targets, predictions, test_loss = evaluation(model=model,test_loader=test_loader,lossfunction=args.lossfunction, model_filepath=args.model_folder_dir+'/'+file)
    #     if test_loss < best_test_loss:
    #         best_test_loss = test_loss
    #         best_test_loss_model = file.split('.')[0]
    #         targets_best = targets
    #         predictions_best =predictions
    #     else:
    #         pass


    # # 3. evaluation test sets with accuracy, precision, F1 score and AUC
    # if is_train == True:
    #     # first set the best loss to infinity
       
    # else:
    #     model.load_state_dict(test_model)
    #     test_acc, test_precision, test_recall, test_F1, test_label_pred, test_label_true = evaluation(model=model, dataloader=test_loader,flag="test_set",DEVICE=DEVICE)
    #     plot_Confusion_Matrix(test_label_true, test_label_pred, model_name, flag="test_set")
    #     plot_Confusion_Matrix(train_label_true, train_label_pred, model_name, flag="train_set")
    
    # # 4. plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set hyper parameters for the training.')
    parser.add_argument('--path',type=str, required=True, help="the path of the dataset")
    parser.add_argument('--plot_folder_dir', type=str,required=True,help="the folder directory where plot results are stored")
    parser.add_argument('--model_folder_dir', type=str, required=True,help="the directory of the model folder")
    parser.add_argument('--EPOCH',type=int, required=True, default=100, help="the number of epochs")
    parser.add_argument('--BATCH_SIZE', type=int, required=True, help="batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=True,help='the learning rate')
    parser.add_argument('--patience', type=int, required=True, help='the counter of the early_stopping')
    parser.add_argument('--train_percentage',type=float, required=True,help="the training percentage of the model")
    parser.add_argument('--validate_percentage',type=float,required=True,help="the valiation set percentage")
    parser.add_argument('--d_model', type=int, required=True, default=512, help="the dimension of the model")
    parser.add_argument('--d_hidden', type=int, required=True, default=1024, help="The dimmension of the hidden layers in Position-wise Feedforward network")
    parser.add_argument('-q', type=int, required=True, help='the dimension of the linear layer in the Multi-Head Attention')
    parser.add_argument('-v', type=int, required=True, help='the dimension of the linear layer in the Multi-Head Attention')
    parser.add_argument('-head', type=int, required=True, help='the head number of the Multi-Head Attention')
    parser.add_argument('-N',  type=int, required=True, help='the number of the encoders')
    parser.add_argument('--dropout', type=float, required=True, help="the random dropout")
    parser.add_argument('--sliding_window_length', type=int, required=True, help="The length of sliding window when spliting the data")
    parser.add_argument('--optimizer_name',type=str, required=True, default='Adagrad', help="The name of the optimizer")
    parser.add_argument('--is_train',action="store_true",help="parameter to determine whether run training cycle or not")
    parser.add_argument('--given_best_model_path', type=str, required=False, help="the model directory used for testing")
    parser.add_argument('--num_exps', type=int, required=True, help="The quantity of the experiments")
    args = parser.parse_args()
    main(args)