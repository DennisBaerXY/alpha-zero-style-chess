import torch
import torch.nn as nn
import torch.nn.functional as F

from aki_chess_ai import utils


class ChessValueNetwork(nn.Module):
    def __init__(self):
        super(ChessValueNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def predict(self, state, to_play):
        if type(state) == str:
            state = utils.getStateFromFEN(state, to_play=to_play)

        state = torch.tensor(state).float()
        state = state.view(1, -1)
        value = self.forward(state)
        return value.item()
