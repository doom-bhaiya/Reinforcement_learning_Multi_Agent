import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super(FullyConnectedModel, self).__init__()


        self.fc1 = nn.Linear(input_size, 64)
        # self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 16)
        # self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(16, 8)
        # self.dropout3 = nn.Dropout(dropout_rate)
        
        self.part1 = nn.Linear(8, 1)
        self.part2 = nn.Linear(8, output_size)
        self.fc_final = nn.Linear(output_size + 1, output_size)
        self.softmax = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.dropout1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        # x = self.dropout3(x)
        
        part1_output = self.part1(x)
        part2_output = self.part2(x)
        combined = torch.cat((part1_output, part2_output), dim=1)
        x = self.fc_final(combined)
        x = self.softmax(x)
        return x



class Actor_Critic(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super(Actor_Critic, self).__init__()

        self.fc_activation = nn.LeakyReLU(negative_slope=0.2)


        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        
        self.actor_fc1 = nn.Linear(16, 8)
        self.actor_fc2 = nn.Linear(8, output_size)
        self.actor_final = nn.Tanh()

        self.critic_fc1 = nn.Linear(16, 8)
        self.critic_fc2 = nn.Linear(8, 1)
        self.critic_final = nn.Sigmoid()

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc_activation(x)

        x = self.fc2(x)
        x = self.fc_activation(x)

        x = self.fc3(x)
        x = self.fc_activation(x)

        actor_output = self.actor_fc1(x)
        actor_output = self.fc_activation(actor_output)
        actor_output = self.actor_fc2(actor_output)
        actor_output = self.fc_activation(actor_output)
        actor_output = self.actor_final(actor_output)

        critic_output = self.critic_fc1(x)
        critic_output = self.fc_activation(critic_output)
        critic_output = self.critic_fc2(critic_output)
        critic_output = self.fc_activation(critic_output)
        critic_output = self.critic_final(critic_output) 

        return actor_output, critic_output
