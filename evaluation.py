import torch
from module.loss import Myloss
# return the accuracy
# todo return the F1 score, precision, and so on
def evaluation(model,dataloader, DEVICE, flag = 'test_set'):
    correct = 0
    total = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    loss_function = Myloss()
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # print(f'y on {flag}: ',y)
            # print('y\'s shape ',y.shape)
            y_pre, _, _, _, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
            # y_prep.shape: torch.Size([batchsize, 2])  the datatype of the last dimension is float
            # print("y_pre:",y_pre)
            # print('y_prep.shape:',y_pre.shape)
            _, label_index = torch.max(y_pre.data, dim=-1) # label_index is a 2-dim tensor. label_index.shape: torch.Size([batchsize])
            # the data type of the last dimension is int
            # print("label_index:",label_index)
            # print('label_index.shape:',label_index.shape)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            for i in range(label_index.shape[0]):
                if label_index[i] == int(y[i]):
                    if label_index[i] == 1:
                        TP += 1
                    elif label_index[i] == 0:
                        TN += 1
                elif label_index[i] != int(y[i]):
                    if label_index[i] == 1:
                        FP += 1
                    elif label_index[i] == 0:
                        FN += 1
        precision = round((100 * (TP / (TP + FP))),2)
        recall = round((100*(TP / (FN + TP))),2)
        F1 = round((100*((2*precision*recall)/(precision+recall))),2)

        # if flag == 'test_set':
        #     # correct_on_test.append(round((100 * correct / total), 2))
        # elif flag == 'train_set':
        #     # correct_on_train.append(round((100 * correct / total), 2))
        print(f'Metrix on {flag}, accuracy: %.2f %%' % (100 * correct / total) +" " + f'precision:  %.2f %%' % precision + ' ' + f'recall: %.2f %%' % recall +
              ' ' + f'F1 score: %.2f %%' % F1)

        return round((100 * correct / total), 2), precision, recall, F1


