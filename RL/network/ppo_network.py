import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=256):
        """
        Critic输出的是对策略的评价，是一个数值
        :param input_dim:
        :param output_dim:
        :param hidden_dim:
        """
        super(Critic,self).__init__()
        assert output_dim == 1 # critic must output a single value
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value