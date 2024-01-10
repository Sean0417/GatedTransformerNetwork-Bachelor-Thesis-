import torch
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
import os
import wandb

from module.transformer import Transformer
from module.loss import Myloss
from processEEG import EEGDataset
from train import training_validation
from plot import plot_learning_curve

from utils.random_seed import setup_seed

def main(args):
    # 1. data preprocessing
    path = 'EEG_Eye_State_Classification.csv'

    draw_key = 1  # 大于等于draw_key才会保存图像
    file_name = path.split('/')[-1][0:path.split('/')[-1].index('.')]  # 获得文件名字
    # 超参数设置
    EPOCH = 10
    BATCH_SIZE = 50
    LR = 1e-4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
    print(f'use device: {DEVICE}')

    train_percentage = 0.6
    validate_percentage = 0.2

    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    N = 8
    dropout = 0.2
    pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
    mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
    # 优化器选择
    optimizer_name = 'Adagrad'

    # split the data into train, validate and test
    train_dataset = EEGDataset(path=path, dataset='train', train_percentage=train_percentage,validate_percentage=validate_percentage)
    test_dataset = EEGDataset(path=path, dataset='test', train_percentage=train_percentage,validate_percentage=validate_percentage)
    validate_dataset = EEGDataset(path=path, dataset='validate',train_percentage=train_percentage,validate_percentage=validate_percentage)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # -------------------------------
    DATA_LEN = train_dataset.train_len  # 训练集样本数量
    d_input = train_dataset.input_len  # 时间部数量
    d_channel = train_dataset.channel_len  # 时间序列维度
    d_output = train_dataset.output_len  # 分类类别

    # 维度展示
    print('data structure: [lines, timesteps, features]')
    print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
    print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
    print(f'Number of classes: {d_output}')
    
    # 2. training and validation
    # 创建Transformer模型
    net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                    q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
    


    print("===================train and validation====================")
    # experiments loop
    for exp_idx in range(args.num_exps):
            # wandb initialization
        wandb.init(project='GTNforEEG',
                job_type="training",
                reinit=True,
                )
        # model initialization
        model = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
        wandb.watch(model,log="all")
        model_name, model, all_epoch_train_loss, all_epoch_val_loss = training_validation(model=model,
                                                        epoch_sum=args.num_of_epochs,
                                                        train_loader=train_loader,
                                                        val_loader=val_loader,
                                                        patience=args.patience,
                                                        learning_rate=args.learning_rate,
                                                        exp_index=exp_idx,
                                                        model_folder_directory=args.model_folder_dir)
        # plot the learning curve
        plot_learning_curve(train_loss=all_epoch_train_loss,val_loss=all_epoch_val_loss,plot_folder_dir=args.plot_folder_dir,model_name=model_name)

        print("round"+str(exp_idx+1)+" has been done")
    

    # 3. evaluation test sets with accuracy, precision, F1 score and AUC


    # 4. plot