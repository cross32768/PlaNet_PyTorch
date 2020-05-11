from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.fc = nn.Linear(256*2*2, embed_dim)

    def forward(self, obs):
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        hidden = F.relu(self.cv4(hidden))
        hidden = hidden.view(hidden.size(0), -1)
        return self.fc(hidden)
