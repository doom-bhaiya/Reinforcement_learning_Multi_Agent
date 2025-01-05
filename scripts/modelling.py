import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(16, 8)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.part1 = nn.Linear(8, 1)
        self.part2 = nn.Linear(8, output_size)
        self.fc_final = nn.Linear(output_size + 1, output_size)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        part1_output = self.part1(x)
        part2_output = self.part2(x)
        combined = torch.cat((part1_output, part2_output), dim=1)
        x = self.fc_final(combined)
        x = self.softmax(x)
        x = (x - 0.5) * 2
        return x
