import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, factor=0.1):
        super(LabelSmoothing, self).__init__()

        self.factor = factor
        self.smooth = 1 - factor

    def forward(self, input, target):
        input = input.log_softmax(dim=1)

        fill_value = self.factor / (input.size(1) - 1)
        y = torch.full_like(input, fill_value, requires_grad=False)
        y.scatter_(1, target.unsqueeze(1), self.smooth)

        return -(y * input).sum(1).mean()