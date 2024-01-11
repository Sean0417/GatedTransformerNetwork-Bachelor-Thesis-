from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from scipy import stats
from torch.utils.data import Dataset # Dataset is an abstract class, can only be herited
from torch.utils.data import DataLoader

class EEGDataset(Dataset):
    def __init__(self,
                 path: str,
                 dataset: str,
                 train_percentage:float,
                 validate_percentage:float):
        """
        训练数据集与测试数据集的Dataset对象
        :param path: 数据集路径
        :param dataset: 区分是获得训练集还是测试集
        """
        super(EEGDataset, self).__init__()
        self.dataset = dataset  # 选择获取测试集还是训练集
        self.train_len, \
        self.test_len, \
        self.validate_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.validate_dataset, \
        self.validate_label, \
        self.test_dataset, \
        self.test_label, \
        self.max_length_sample_inTest, \
        self.train_dataset_with_no_paddding = self.pre_option(path,train_percentage, validate_percentage)
    
    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index]
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index]
        elif self.dataset == 'validate':
            return self.validate_dataset[index], self.validate_label[index]

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len
        elif self.dataset == 'validate':
            return self.validate_len
    
    def pre_option(self, path: str, train_percentage: float, validate_percentage: float):
        train_percentage = train_percentage
        validate_percentage = validate_percentage
        sliding_window_length = 128
        df = pd.read_csv(path)

        # remove the outliers
        z_scores = stats.zscore(df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 10).all(axis=1)
        df_filtered1 = df[filtered_entries]

        #reset index
        df_filtered1 = df_filtered1.reset_index(drop=True)

        arr_filtered1 = np.array(df_filtered1)
        arr_len = arr_filtered1.shape[0]
        
        # min-max normalization
        min_values = arr_filtered1[:, :-1].min(axis=0, keepdims=True)
        max_values = arr_filtered1[:, :-1].max(axis=0, keepdims=True)

        normalized_array = (arr_filtered1[:, :-1] - min_values) / (max_values - min_values)

        # print("Max Values in Each Column:", max_values)
        # print("Min Values in Each Column:", min_values)
        
        normalized_array = np.hstack((normalized_array, arr_filtered1[:, -1:]))
        arr_filtered1 = normalized_array

        X = []
        y = []
        
        flag = 0
        record_row = 0
        
        while(record_row != arr_len-1):
            if arr_len - flag > sliding_window_length:
                for row in range(flag, arr_len - sliding_window_length):
                    _X = arr_filtered1[row:row+sliding_window_length,0:14]
                    _y = arr_filtered1[row:row+sliding_window_length,14]
                    if sum(_y) == sliding_window_length or sum(_y) == 0:
                        X.append(_X)
                        y.append(_y[0])
                        record_row = row
                    else:
                        flag = row + sliding_window_length
                        break
            else:
                for row in range(flag, arr_len):
                    _X = arr_filtered1[row,0:14]
                    _y = arr_filtered1[row,14]

                    # padding 0
                    zero_array = np.zeros(14)
                    zero_array_list = np.tile(zero_array, (sliding_window_length-1,1))
                    zero_array_list = np.vstack(zero_array_list)
                    _X = np.array([_X])
                    _X = np.concatenate((_X, zero_array_list), axis=0)
                    
                    X.append(_X)
                    y.append(_y)

                    record_row = row


        print(record_row)
        X = np.array(X)
        y = np.array(y)
        # X = torch.Tensor(X)
        # y = torch.Tensor(y)

        # split the data into training, validation and testset 
        train_size = int(len(X)*train_percentage)
        validate_size = int(len(X)*validate_percentage)
        train_dataset, train_label = X[:train_size], y[:train_size]
        validate_dataset, validate_label = X[train_size:train_size+validate_size],y[train_size:train_size+validate_size]
        test_dataset, test_label = X[train_size+validate_size:],y[train_size+validate_size:]

        output_len = 2
        train_dataset_with_no_paddding = train_dataset


        # transfer the datatype into tensor
        train_dataset = torch.Tensor(train_dataset)
        train_label = torch.Tensor(train_label)
        test_dataset = torch.Tensor(test_dataset)
        test_label = torch.Tensor(test_label)
        validate_dataset = torch.Tensor(validate_dataset)
        validate_label = torch.Tensor(validate_label)

        channel = train_dataset[0].shape[-1]
        input = test_dataset[0].shape[-2]
        train_len = train_dataset.shape[0]
        test_len = test_dataset.shape[0]
        validate_len = validate_dataset.shape[0]
        max_length_sample_inTest = 128


        return train_len, test_len, validate_len, input, channel, output_len, train_dataset, train_label, validate_dataset, validate_label, test_dataset, test_label, max_length_sample_inTest, train_dataset_with_no_paddding

