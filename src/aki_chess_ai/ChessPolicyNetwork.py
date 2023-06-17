import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aki_chess_ai import utils


class ChessPolicyNetwork(nn.Module):
    def __init__(self):
        super(ChessPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 4096)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def predict(self, state):
        if isinstance(state, str):
            state = utils.getStateFromFEN(state)
        state = torch.tensor(state).float().view(1, -1)
        logits = self.forward(state).detach().numpy()
        logits = logits.flatten()  # Flatten the array to 1D
        return logits

    def action_probabilities(self, state, valid_moves):
        logits = self.predict(state)

        if len(valid_moves) == 1:
            print("Only one move available")

        valid_probs = np.array([logits[self._move_to_index(move)] for move in valid_moves])
        valid_probs = utils.softmax(valid_probs)
        combined = zip(valid_moves, valid_probs)
        return list(combined)

    def action_probabilitiesThreaded(self, state, valid_moves):
        logits = self.predict(state)

        if len(valid_moves) == 1:
            print("Only one move available")

        valid_probs = []
        try:
            valid_probs = np.array([logits[self._move_to_index_threaded(move)] for move in valid_moves])
        except Exception as e:
            print("Error in doofer valid_probs calculation:", e)
        valid_probs = utils.softmax(valid_probs)
        return valid_probs

    def select_move(self, state, valid_moves):
        probs = self.action_probabilities(state, valid_moves)
        move = np.random.choice(valid_moves, p=probs)
        return move

    def _move_to_index(self, move):
        from_square = chess.parse_square(move[:2])
        to_square = chess.parse_square(move[2:4])
        return from_square * 64 + to_square

    def _move_to_index_threaded(self, move):
        try:
            from_square = move.from_square
            to_square = move.to_square
        except Exception as e:
            print(e)

        return from_square * 64 + to_square
