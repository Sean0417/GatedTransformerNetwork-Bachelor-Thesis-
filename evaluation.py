import torch
from module.loss import Myloss
import numpy as np
import wandb

 # testing
 
def evaluation(model, dataloader, DEVICE):

    correct = 0
    total = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    label_pred = []
    label_true = []

    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = model(x, 'test') # y_pre is a tensor with a dimension of batchsize*2(200*2 for instance if the batchsize is 200),
            
            # acc
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            
            if label_index[0] == int(y[0]):
                if label_index[0] == 1:
                    TP += 1
                elif label_index[0] == 0:
                    TN += 1
            elif label_index[0] != int(y[0]):
                if label_index[0] == 1:
                    FP += 1
                elif label_index[0] == 0:
                    FN += 1
            
            # label_true = np.concatenate((label_true, y.cpu().numpy()))
            # label_pred = np.concatenate((label_pred, label_index.cpu().numpy()))
            label_pred.append(label_index.cpu().numpy()[0])
            label_true.append(y.cpu().numpy()[0])
        # print(label_pred)
        # print(label_true)

        if TP+FP != 0:
            precision = TP / (TP + FP)
        else:
            print("the denominator of the precision is 0, precision is set to 0.")
            precision = 0
        if FN+TP != 0:
            recall =TP / (FN + TP)
        else:
            print("the denominator of the recall is 0, recall is set to 0")
            recall = 0
        if(precision+recall != 0):
            F1 = (2*precision*recall)/(precision+recall)
        else:
            print("since the denominator is zero, F1 is set to 0")
            F1 = 0

        test_acc = round((100 * correct / total), 2)
        test_precision = round(100*precision,2)
        test_recall = round(100*recall,2)
        test_f1_score = round(100*F1,2)

        print(f'Test accuracy: %.2f %%' % (test_acc) +" " + f'precision:  %.2f %%' % (test_precision) + ' ' + f'recall: %.2f %%' % (test_recall) +
              ' ' + f'F1 score: %.2f %%' % (test_f1_score))
        

        wandb.log({"test_acc":test_acc,
                    "test_precision":test_precision,
                    "test_recall":test_recall,
                    "test_f1_score":test_f1_score})
        return test_acc, test_precision, test_recall, test_f1_score, label_pred, label_true