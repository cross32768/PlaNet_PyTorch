from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, obs):
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        hidden = F.relu(self.cv4(hidden))
        hidden = hidden.view(hidden.size(0), -1)
        return hidden


class ObservationModel(nn.Module):
    def __init__(self, rnn_hidden_dim, state_dim):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(rnn_hidden_dim + state_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, rnn_hidden, state):
        hidden = self.fc(torch.cat([rnn_hidden, state], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        recon_obs = self.dc4(hidden)
        return recon_obs


class RewardModel(nn.Module):
    def __init__(self, rnn_hidden_dim, state_dim, hidden_dim=200):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(rnn_hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, rnn_hidden, state):
        hidden = F.relu(self.fc1(torch.cat([rnn_hidden, state], dim=1)))
        hidden = F.relu(self.fc2(hidden))
        reward = self.fc3(hidden)
        return reward
