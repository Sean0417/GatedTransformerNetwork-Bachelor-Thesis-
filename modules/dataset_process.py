from torch.utils.data import Dataset
import torch

from scipy.io import loadmat


class MyDataset(Dataset):
    def __init__(self,
                 path: str,
                 dataset: str):
        """
        :param path: directory of the file
        :param dataset: define whether it's training data or test data
        """
        super(MyDataset, self).__init__()
        self.dataset = dataset  # trainset or test set
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.test_dataset, \
        self.test_label, \
        self.max_length_sample_inTest, \
        self.train_dataset_with_no_paddding = self.pre_option(path)

    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index] - 1
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index] - 1

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len

    # data preprocessing
    def pre_option(self, path: str):
        """
        datapreprocessing, we use the longest timesteps as our timesteps, and padding with zeros
        :param path: the directory of the files
        return: the amount of training data, the amount of the test data, the length of the timesteps, the number of channels, the number of classes, train data, train label,
        testdata, testlabel, datasample with the longest timesteps, data without paddings
        """
        m = loadmat(path)

        # m is a dictionary, four keys are in m, the last key-value sets stores the data
        x1, x2, x3, x4 = m
        data = m[x4]

        data00 = data[0][0]
        # print('data00.shape', data00.shape)  # ()  data00 reached the dimension of the data

        index_train = str(data.dtype).find('train\'')
        index_trainlabels = str(data.dtype).find('trainlabels')
        index_test = str(data.dtype).find('test\'')
        index_testlabels = str(data.dtype).find('testlabels')
        list = [index_test, index_train, index_testlabels, index_trainlabels]
        list = sorted(list)
        index_train = list.index(index_train)
        index_trainlabels = list.index(index_trainlabels)
        index_test = list.index(index_test)
        index_testlabels = list.index(index_testlabels)

        # [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]   numpy.object
        train_label = data00[index_trainlabels]
        train_data = data00[index_train]
        test_label = data00[index_testlabels]
        test_data = data00[index_test]

        train_label = train_label.squeeze()
        train_data = train_data.squeeze()
        test_label = test_label.squeeze()
        test_data = test_data.squeeze()

        train_len = train_data.shape[0]
        test_len = test_data.shape[0]
        output_len = len(tuple(set(train_label)))

        # the longest timesteps
        max_lenth = 0  # 93
        for item in train_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]
                # max_length_index = train_data.tolist().index(item.tolist())

        for item in test_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        # Padding with zeros
        # train_data, test_data are numpy.object type，which can't directly handle numpy.ndarray data.
        train_dataset_with_no_paddding = []
        test_dataset_with_no_paddding = []
        train_dataset = []
        test_dataset = []
        max_length_sample_inTest = []
        for x1 in train_data:
            train_dataset_with_no_paddding.append(x1.transpose(-1, -2).tolist())
            x1 = torch.as_tensor(x1).float()
            if x1.shape[1] != max_lenth:
                padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
                x1 = torch.cat((x1, padding), dim=1)
            train_dataset.append(x1)

        for index, x2 in enumerate(test_data):
            test_dataset_with_no_paddding.append(x2.transpose(-1, -2).tolist())
            x2 = torch.as_tensor(x2).float()
            if x2.shape[1] != max_lenth:
                padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
                x2 = torch.cat((x2, padding), dim=1)
            else:
                max_length_sample_inTest.append(x2.transpose(-1, -2))
            test_dataset.append(x2)

        # last dimension[the total length of the data, timesteps, channels]
        # train_dataset_with_no_paddding = torch.stack(train_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
        # test_dataset_with_no_paddding = torch.stack(test_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
        train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
        test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
        train_label = torch.Tensor(train_label)
        test_label = torch.Tensor(test_label)
        channel = test_dataset[0].shape[-1]
        input = test_dataset[0].shape[-2]

        return train_len, test_len, input, channel, output_len, train_dataset, train_label, test_dataset, test_label, max_length_sample_inTest, train_dataset_with_no_paddding