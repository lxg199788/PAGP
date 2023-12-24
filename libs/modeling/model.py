import torch
import torch.nn as nn


# this code is implemented after referring to https://github.com/BangguWu/ECANet
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, k=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden_size, batch_first=True) for _ in range(input_size)])
        self.fc = nn.Sequential(nn.Linear(input_size * hidden_size, 2))
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False, groups=1))
        self.simgoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, index):
        outs = []
        for num in range(self.input_size):
            lstm = self.lstms[num]
            out = []
            for i, data in enumerate(x):
                data = data[index[i][0]: index[i][1]][:, num]
                data = data.view(1, data.shape[0], 1)
                data, _ = lstm(data)
                out.append(data[:, -1, :])
            out = torch.cat(out)
            outs.append(out[:, None, :])
        outs = torch.cat(outs, dim=1)
        cofs = self.avgPool(outs)
        cofs = self.conv(cofs.transpose(-1, -2)).squeeze()
        cofs = self.simgoid(cofs)
        outs = outs * cofs.unsqueeze(-1).expand_as(outs)
        return self.fc(outs.view(outs.shape[0], -1))
