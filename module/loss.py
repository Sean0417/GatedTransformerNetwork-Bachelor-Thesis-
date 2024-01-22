from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss

class Myloss(Module):
    def __init__(self):
        super(Myloss, self).__init__()
        # self.loss_function = CrossEntropyLoss()
        self.loss_function = BCELoss()

    def forward(self, y_pre, y_true):
        y_true = y_true.long()
        loss = self.loss_function(y_pre, y_true)

        return loss