import torch.nn as nn
import torch.nn.functional as functional


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 6)
        self.fc4 = nn.Linear(6, output_dim)


    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LargeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)


    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x