import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, lambda_main, lambda_user, lambda_item):
        super().__init__()
        self.lambda_main = lambda_main
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, pred_user, pred_item, label, label_user, label_item):
        loss_main = self.criterion(pred, label)
        loss_user = self.criterion(pred_user, label_user)
        loss_item = self.criterion(pred_item, label_item)
        loss = self.lambda_main * loss_main + self.lambda_user * loss_user\
            + self.lambda_item * loss_item
        return loss, loss_main, loss_user, loss_item