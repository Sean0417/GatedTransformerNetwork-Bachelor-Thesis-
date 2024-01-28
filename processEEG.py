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
                        flag = row + sliding_window_length
                        break



            else:
                # for row in range(flag, arr_len):
                #     _X = data[row,0:14]
                #     _y = data[row,14]

                #     # padding 0
                #     zero_array = np.zeros(14)
                #     zero_array_list = np.tile(zero_array, (sliding_window_length-1,1))
                #     zero_array_list = np.vstack(zero_array_list)
                #     _X = np.array([_X])
                #     _X = np.concatenate((_X, zero_array_list), axis=0)
                    
                #     X.append(_X)
                #     y.append(_y)

                #     record_row = row
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
    def get_labels(self, X, y, flg = "train"):
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
        print("There are "+str(zeros)+" zeros. on "+flg+" set.")
        print("There are " + str(ones)+" ones. on"+ flg +" set.")
        return zeros, ones
    def spllit_the_data_by_3_1(self, X:np.array,y:np.array, train_ratio):
        arr_len = X.shape[0]
        train_len = int(arr_len * train_ratio)

        train_X = X[:train_len]
        train_y = y[:train_len]

        val_X = X[train_len:]
        val_y = y[train_len:]

        return train_X, train_y, val_X, val_y
    def get_ones_zeros(self, X:np.array,y:np.array):
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
        _X_label_zero = np.array(_X_label_zero)
        _y0 = np.array(_y0)
        _X_label_one = np.array(_X_label_one)
        _y1 = np.array(_y1)
        return _X_label_zero, _y0, _X_label_one, _y1, zeros, ones
    def standardize_except_last_dimension(self,mean, std, data):
        # 提取除最后一个维度之外的所有维度
        features = data[:, :-1]
        
        # # 计算均值和标准差
        # mean = np.mean(features, axis=0)
        # std = np.std(features, axis=0)
        
        # 标准化除最后一个维度之外的所有维度
        standardized_features = (features - mean) / std
        
        # 将标准化后的特征和原始的最后一个维度重新组合
        standardized_data = np.column_stack((standardized_features, data[:, -1]))
        
        return standardized_data

    
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


        # ==============================
        # dataset, label = self.slide_data_with_slidingWindow(arr_filtered1, sliding_window_length=sliding_window_length)
        # X0, y0, X1, y1, zeros, ones= self.get_ones_zeros(dataset, label)
        # x0_train, y0_train, x0_test, y0_test = self.spllit_the_data_by_3_1(X0,y0, train_percentage)
        # x1_train, y1_train, x1_test, y1_test = self.spllit_the_data_by_3_1(X1,y1, train_percentage)
        # print(x0_train.shape)
        # print(x1_train.shape)
        # X_train = np.concatenate((x0_train,x1_train))
        # y_train = np.concatenate((y0_train, y1_train))
        # X_train, y_train = self.balance_data(X_train, y_train)
        # X_test = np.concatenate((x0_test,x1_test))
        # y_test = np.concatenate((y0_test,y1_test))


        # # min-max normalization
        # # X_train = np.reshape(X_train, (-1, 14))
        # # X_test = np.reshape(X_test, (-1, 14))
        # # min_train = np.min(X_train, axis=0)
        # # max_train = np.max(X_train, axis=0)
        # # print(min_train)
        # # print(max_train)
        # # normalized_train_data = (X_train - min_train) / (max_train - min_train)
        # # normalized_test_data = (X_test - min_train) / (max_train - min_train)
        
        # # normalized_test_data = np.reshape(normalized_test_data, (-1,sliding_window_length,14))
        # # normalized_train_data = np.reshape(normalized_train_data, (-1,sliding_window_length, 14))
        
        # # Reshape 数据为二维数组
        # X_train_flat = np.reshape(X_train, (-1, 14))
        # X_test_flat = np.reshape(X_test, (-1, 14))

        # # 计算训练集中每个特征的最小值和最大值
        # min_train = np.min(X_train_flat, axis=0)
        # max_train = np.max(X_train_flat, axis=0)

        # # 将为零的行排除在标准化计算之外
        # nonzero_rows_train = np.all(X_train_flat != 0, axis=1)
        # nonzero_rows_test = np.all(X_test_flat != 0, axis=1)

        # # 进行 min-max 标准化，仅对非零行进行标准化
        # normalized_train_data = np.zeros_like(X_train_flat)
        # normalized_train_data[nonzero_rows_train, :] = (X_train_flat[nonzero_rows_train, :] - min_train) / (max_train - min_train)

        # normalized_test_data = np.zeros_like(X_test_flat)
        # normalized_test_data[nonzero_rows_test, :] = (X_test_flat[nonzero_rows_test, :] - min_train) / (max_train - min_train)

        # # 将数据恢复为原始形状
        # normalized_train_data = np.reshape(normalized_train_data, X_train.shape)
        # normalized_test_data = np.reshape(normalized_test_data, X_test.shape)

        # # 输出最小值和最大值以及标准化后的数据
        # print("Min values:", min_train)
        # print("Max values:", max_train)
        # print("Normalized train data:", normalized_train_data)
        # print("Normalized test data:", normalized_test_data)

        # train_dataset = normalized_train_data
        # train_label = y_train
        # test_dataset = normalized_test_data
        # test_label = y_test
        # print("train_Data",train_dataset)
        # print("test_Data:",test_dataset)
        # ==============================
        # ==============method 1=========

        train_size = int(train_percentage*arr_len)
        val_size = int(validate_percentage*arr_len)
        test_size = arr_len - train_size - val_size

        train_data = arr_filtered1[:train_size,:]
        val_data = arr_filtered1[train_size:train_size+val_size,:]
        test_data = arr_filtered1[train_size+val_size:,:]

        # calculate the max and min value of the train data
        min_train_val = train_data[:,0:14].min(axis=0)
        max_train_val = train_data[:,0:14].max(axis=0)



        # # 提取除最后一个维度之外的所有维度
        # features = train_data[:, :-1]
        
        # # 计算均值和标准差
        # mean = np.mean(features, axis=0)
        # std = np.std(features, axis=0)
        # train_data = self.standardize_except_last_dimension(mean, std, train_data)
        # test_data = self.standardize_except_last_dimension(mean, std, test_data)



        train_data = self.min_max_normalization(min_train_val, max_train_val, train_data)
        # normalized_validate_arr = self.min_max_normalization(min_train_val, max_train_val, val_data)
        test_data = self.min_max_normalization(min_train_val, max_train_val, test_data)

        train_dataset, train_label = self.slide_data_with_slidingWindow(train_data, sliding_window_length)
        self.get_labels(train_dataset, train_label)
        # balance the output of labels of 1 and zeros
        train_dataset, train_label = self.balance_data(train_dataset, train_label)
        self.get_labels(train_dataset, train_label)
        # validate_dataset, validate_label = self.slide_data_with_slidingWindow(normalized_validate_arr, sliding_window_length=sliding_window_length)
        # validate_dataset, validate_label = self.balance_data(validate_dataset, validate_label)
        test_dataset, test_label = self.slide_data_with_slidingWindow(test_data,sliding_window_length=sliding_window_length)
        test_zeros, test_ones = self.get_labels(test_dataset, test_label, flg="test")
        # test_dataset, test_label = self.balance_data(test_dataset, test_label)
        print("normalized_train_arr:",train_label)
        print("normalized_train_arr.shape",train_dataset.shape)
        print("normalized_test_arr:", test_label)
        print("normalized_test_data.shape:", test_dataset.shape)
        # ===========================================
   
        output_len = 2
        train_dataset_with_no_paddding = train_dataset


        # transfer the datatype into tensor
        train_dataset = torch.Tensor(train_dataset)
        train_label = torch.Tensor(train_label)
        test_dataset = torch.Tensor(test_dataset)
        test_label = torch.Tensor(test_label)
        # validate_dataset = torch.Tensor(validate_dataset)
        # validate_label = torch.Tensor(validate_label)
        validate_dataset = torch.Tensor(test_dataset)
        validate_label = torch.Tensor(test_label)

        channel = train_dataset[0].shape[-1]
        input = test_dataset[0].shape[-2]
        train_len = train_dataset.shape[0]
        test_len = test_dataset.shape[0]
        validate_len = validate_dataset.shape[0]
        max_length_sample_inTest = 128


        return train_len, test_len, validate_len, input, channel, output_len, train_dataset, train_label, validate_dataset, validate_label, test_dataset, test_label, max_length_sample_inTest, train_dataset_with_no_paddding