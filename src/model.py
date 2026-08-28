import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # conv
        self.initial_conv = nn.Conv2d(
            3, 128, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.initial_bn = nn.BatchNorm2d(128)

        # resnet block 1
        self.res1_conv1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res1_bn1 = nn.BatchNorm2d(128)
        self.res1_conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res1_bn2 = nn.BatchNorm2d(128)

        # rresnet block 2
        self.res2_conv1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res2_bn2 = nn.BatchNorm2d(128)

        # value head
        self.value_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1, bias=True)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc = nn.Linear(3 * 6 * 7, 32)
        self.value_head = nn.Linear(32, 1)

        # policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_head = nn.Linear(32 * 6 * 7, 7)
        self.policy_ls = nn.LogSoftmax(dim=1)

        self.to(device)

    def forward(self, x):
        # game=(3, 6, 7)
        x.view(-1, 3, 6, 7)

        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # resnet - save original x to preserve gradients
        # res 1
        res = x
        x = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = F.relu(self.res1_bn2(self.res1_conv2(x)))
        x += res
        x = F.relu(x)

        # res 2
        res = x
        x = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = F.relu(self.res2_bn2(self.res2_conv2(x)))
        x += res
        x = F.relu(x)

        # value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 3 * 6 * 7)  # flatten for linear layer
        v = F.relu(self.value_fc(v))
        v = F.tanh(v)

        # policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 6 * 7)  # flatten for linear layer
        p = F.relu(self.policy_head(p))
        p = self.policy_ls(p).exp()

        return v, p


if __name__ == "__main__":
    from torchinfo import summary

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = Connect4Model(device)

    architecture_summary = summary(model, input_size=(16, 3, 6, 7), verbose=0)
    print(architecture_summary)
