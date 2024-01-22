from torch.utils.data import Dataset
import torch
import random
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
                 validate_percentage:float,
                 sliding_window_length:int):
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
        self.train_dataset_with_no_paddding = self.pre_option(path,train_percentage, validate_percentage,sliding_window_length)
    
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
    def min_max_normalization(self,min_val,max_val,data):
        # select the 14 features
        selected_features = data[:, :14]

        # Min-Max normalization
        normalized_features = (selected_features - min_val) / (max_val - min_val)

        array_normalized = np.hstack((normalized_features, data[:, 14:]))
    
        return array_normalized
    
    def slide_data_with_slidingWindow(self, data, sliding_window_length):
        X = []
        y = []
        flag = 0
        record_row = 0
        arr_len = data.shape[0]

        while(record_row < arr_len-sliding_window_length-1):
            if arr_len - flag -1 > sliding_window_length:
                for row in range(flag, arr_len - sliding_window_length):
                    _X = data[row:row+sliding_window_length,0:14]
                    _y = data[row:row+sliding_window_length,14]
                    if sum(_y) == sliding_window_length or sum(_y) == 0:
                        X.append(_X)
                        y.append(_y[0])
                        record_row = row
                    elif sum(_y) == sliding_window_length-1 or sum(_y) == 1:
                        flag = row + sliding_window_length
                        break
                    else:
                        if _y[0] == 0:
                            boundary_indices = np.where(np.diff(_y) == 1)[0] + 1
                            _X = _X[0:boundary_indices[0]]
                            _y = _y[0]
                            
                            zero_array = np.zeros(14)
                            zero_array_list = np.tile(zero_array, (sliding_window_length-_X.shape[0],1))
                            zero_array_list = np.vstack(zero_array_list)

                            _X = np.array(_X)
                            _X = np.concatenate((_X, zero_array_list), axis=0)

                            X.append(_X)
                            y.append(_y)

                            flag = row + boundary_indices[0]

                            break
                        
                        elif _y[0] == 1:
                            boundary_indices = np.where(np.diff(_y) == -1)[0] + 1
                            _X = _X[0:boundary_indices[0]]
                            _y = _y[0]

                            zero_array = np.zeros(14)
                            zero_array_list = np.tile(zero_array, (sliding_window_length-_X.shape[0],1))
                            zero_array_list = np.vstack(zero_array_list)

                            _X = np.array(_X)
                            _X = np.concatenate((_X, zero_array_list), axis=0)

                            X.append(_X)
                            y.append(_y)

                            flag = row + boundary_indices[0]

                            break




            else:
                for row in range(flag, arr_len):
                    _X = data[row,0:14]
                    _y = data[row,14]

                    # padding 0
                    zero_array = np.zeros(14)
                    zero_array_list = np.tile(zero_array, (sliding_window_length-1,1))
                    zero_array_list = np.vstack(zero_array_list)
                    _X = np.array([_X])
                    _X = np.concatenate((_X, zero_array_list), axis=0)
                    
                    X.append(_X)
                    y.append(_y)

                    record_row = row
                break
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y

    def balance_data(self, X, y):
        # get the number of the 1s and 0s
        ones = 0
        zeros = 0
        _X_label_zero = []
        _y1 = []
        _y0 = []
        _X_label_one = []
        for i in range(y.shape[0]):
            if y[i] == 0:
                zeros += 1
                _X_label_zero.append(X[i])
                _y0.append(int(y[i]))
            elif y[i] == 1:
                ones += 1
                _X_label_one.append(X[i])
                _y1.append(int(y[i]))
        
        if zeros > ones :
            diff = len(_X_label_zero) - len(_X_label_one)
            random_elements = random.choices(_X_label_one,k=diff)
            _y11 = np.ones(len(random_elements)).tolist()
            _X_label_one.extend(random_elements)
            _y1.extend(_y11)
        elif ones > zeros:
            diff = len(_X_label_one) - len(_X_label_zero)
            random_elements = random.choices(_X_label_zero, k=diff)
            _y00 = np.zeros(len(random_elements)).tolist()
            _X_label_zero.extend(random_elements)
            _y0.extend(_y00)

        _y1.extend(_y0)
        _X_label_one.extend(_X_label_zero)

        y = _y1
        X = _X_label_one

        y = np.array(y)
        X = np.array(X)

        return X,y
    
    def pre_option(self, path: str, train_percentage: float, validate_percentage: float, sliding_window_length:int):
        train_percentage = train_percentage
        validate_percentage = validate_percentage
        sliding_window_length = sliding_window_length
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

        train_size = int(train_percentage*arr_len)
        val_size = int(validate_percentage*arr_len)
        test_size = arr_len - train_size - val_size

        train_data = arr_filtered1[:train_size,:]
        val_data = arr_filtered1[train_size:train_size+val_size,:]
        test_data = arr_filtered1[train_size+val_size:,:]

        # calculate the max and min value of the train data
        min_train_val = train_data[:,0:14].min(axis=0)
        max_train_val = train_data[:,0:14].max(axis=0)

        min_test_val = test_data[:,0:14].min(axis=0)
        max_test_val = test_data[:,0:14].max(axis=0)
        # print("min_test:",min_test_val)
        # print("max_test:",max_test_val)

        min_validate_val = val_data[:,0:14].min(axis=0)
        max_validate_val = val_data[:,0:14].max(axis=0)

        normalized_train_arr = self.min_max_normalization(min_train_val, max_train_val, train_data)
        normalized_validate_arr = self.min_max_normalization(min_train_val, max_train_val, val_data)
        normalized_test_arr = self.min_max_normalization(min_train_val, max_train_val, test_data)

        train_dataset, train_label = self.slide_data_with_slidingWindow(normalized_train_arr, sliding_window_length)
        # balance the output of labels of 1 and zeros
        train_dataset, train_label = self.balance_data(train_dataset, train_label)
        validate_dataset, validate_label = self.slide_data_with_slidingWindow(normalized_validate_arr, sliding_window_length=sliding_window_length)
        test_dataset, test_label = self.slide_data_with_slidingWindow(normalized_test_arr,sliding_window_length=sliding_window_length)

        # split the data into training, validation and testset 

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

