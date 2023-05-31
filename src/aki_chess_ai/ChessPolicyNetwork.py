import chess
import numpy as np
import tensorflow as tf

from aki_chess_ai import utils


class ChessPolicyNetwork:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            # Input shape for the chessboard state
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            # Output logits for each legal move
            tf.keras.layers.Dense(4096)
        ])
        self.model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.build((None, 64))

    def predict(self, state):
        if type(state) == str:
            state = utils.getStateFromFEN(state)
        # Reshape the state to match the input shape of the model
        state = tf.reshape(state, (1, 64))
        # Predict the logits for each legal move
        logits = self.model.predict(state, verbose=0)[0]

        return logits

    # Returns the probabilities of each legal move
    def action_probabilities(self, state, valid_moves):
        logits = self.predict(state)

        if(len(valid_moves) == 1):
            print("Only one move available")

        valid_probs = np.array([logits[self._move_to_index(move)] for move in valid_moves])

        valid_probs = utils.softmax(valid_probs)
        # Return a tuple of the legal moves and their probabilities
        combined = zip(valid_moves, valid_probs)
        return list(combined)
    def action_probabilitiesThreaded(self, state, valid_moves):
        logits = self.predict(state)

        # if(len(valid_moves) == 1):
            # print("Only one move available")

        valid_probs = np.array([logits[self._move_toIndexThreaded(move)] for move in valid_moves])

        valid_probs = utils.softmax(valid_probs)
        # Return a tuple of the legal moves and their probabilities

        return valid_probs

    def select_move(self, state, valid_moves):
        # Get the probabilities of each legal move
        probs = self.action_probabilities(state, valid_moves)
        # Select a move according to the probability distribution
        move = np.random.choice(valid_moves, p=probs)
        return move

    def _move_to_index(self, move):
        # Convert UCI move to corresponding index
        from_square = chess.parse_square(move[:2])  # e2e4 -> e2
        to_square = chess.parse_square(move[2:4])  # e2e4 -> e4
        return from_square * 64 + to_square
    def _move_toIndexThreaded(self,move: chess.Move):
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square
