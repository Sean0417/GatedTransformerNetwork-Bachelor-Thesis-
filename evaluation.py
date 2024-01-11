import torch
from module.loss import Myloss
# return the accuracy
# todo return the F1 score, precision, and so on
def evaluation(model,dataloader, DEVICE, flag = 'test_set'):
    correct = 0
    total = 0

    loss_function = Myloss()
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = model(x, 'test') # y_pre 是一个二维数组分别对应0， 1
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        # if flag == 'test_set':
        #     # correct_on_test.append(round((100 * correct / total), 2))
        # elif flag == 'train_set':
        #     # correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


