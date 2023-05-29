import chess
import chess.svg
import numpy as np
import tensorflow as tf

white = 1
black = -1

# Board State
# 8x8 board
board = chess.Board()


class ChessEnv:
    def __init__(self) -> None:
        # Init chess board
        self.board = chess.Board()

    # Color Property for better feedability of the network
    # ensures that the network can learn from both sides in self play
    # https://www.youtube.com/watch?v=62nq4Zsn8vc
    def get_state(self, color=chess.WHITE):
        # chess board is 8x8
        state = np.zeros(64)
        for i in range(64):
            piece: chess.Piece = self.board.piece_at(i)
            # 0 for a blank square
            if piece is not None:
                state[i] = self._encode_piece(piece, color=color)

        # Return the current board state for a nn to use
        return state
    def get_next_state(self, state, to_play, action):
        pass
    def get_canonical_board(self):
        pass


    def _encode_piece(self, piece: chess.Piece, color=white):
        # 0.1 for white pawn, 0.2 for white knight, etc.
        # -0.1 for black pawn, -0.2 for black knight, etc.
        return piece.piece_type / 10 * self._encode_piece_color(piece, color=color)

    def _encode_piece_color(self, piece: chess.Piece, color=chess.WHITE):
        # 1 for white, -1 for black

        if (piece.color == color):
            return 1
        else:
            return -1

    def get_valid_moves(self):
        # Return a list of valid moves
        return [move.uci() for move in self.board.legal_moves]

    def make_move(self, move):
        self.board.push(move)


chessEnv = ChessEnv()
print("\n")
for i in range(64):
    if (i % 8 == 0 and i != 0):
        print("\n")
    print(chessEnv.get_state(color=chess.BLACK)[i], end=" ")

print("\n")
print(chessEnv.get_valid_moves())


# Architecture
# 1. Value Network -> predicts who will win the game
# 2. Policy Network -> predicts the best move to make in a given state
# 3. MCTS -> Monte Carlo Tree Search -> uses the value and policy network to search for the best move

# Value Network
# 1. Input: Board State
# 2. Output: Value of the board state
# 3. Loss: MSE between predicted value and actual value
# 4. Optimizer: Adam
# On Win get 1 reward for all moves leading to win
# On Loss get -1 reward for all moves leading to loss


# Policy Network
# 1. Input: Board State
# 2. Output: Probabilities of all possible moves
# 3. Loss: Cross Entropy Loss between predicted probabilities and actual probabilities
# 4. Optimizer: Adam

# MCTS
# 1. Input: Board State
# 2. Output: Probabilities of all possible moves
# 3. Loss: Cross Entropy Loss between predicted probabilities and actual probabilities
# 4. Optimizer: Adam


# learns through self play


class ChessValueNetwork:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            # Input shape for the chessboard state
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            # Output a single value between -1 and 1
            tf.keras.layers.Dense(1, activation="tanh")
        ])
        self.model.compile(optimizer="adam", loss="mse")

    def predict(self, state):
        # Reshape the state to match the input shape of the model
        state = tf.reshape(state, (1, 64))
        # Predict the value of the state
        value = self.model.predict(state)[0][0]
        return value


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

    def predict(self, state):
        # Reshape the state to match the input shape of the model
        state = tf.reshape(state, (1, 64))
        # Predict the logits for each legal move
        logits = self.model.predict(state)[0]
        return logits

    # Returns the probabilities of each legal move
    def action_probabilities(self, state,valid_moves):
        logits = self.predict(state)
        valid_probs = np.array([logits[self._move_to_index(move)] for move in valid_moves])
        valid_probs = valid_probs - np.min(valid_probs)  # Shift probabilities to make them non-negative
        valid_probs_sum = np.sum(valid_probs)
        if valid_probs_sum > 0:
            valid_probs = valid_probs / valid_probs_sum
        # Return a tuple of the legal moves and their probabilities
        combined = zip(valid_moves, valid_probs)
        return list(combined)


    def select_move(self, state, valid_moves):
# Get the probabilities of each legal move
        probs = self.action_probabilities(state, valid_moves)
        # Select a move according to the probability distribution
        move = np.random.choice(valid_moves, p=probs)
        return move


    def _move_to_index(self, move):
        # Convert UCI move to corresponding index
        from_square = chess.parse_square(move[:2])  # e2e4 -> e2
        to_square = chess.parse_square(move[2:])  # e2e4 -> e4
        return from_square * 64 + to_square


valueNetwork = ChessValueNetwork()

# NOT TRAINED YET

# How likely is it to win for white in this position -> for learning purposes -1 means opponent wins, 1 means we win
print(valueNetwork.predict(chessEnv.get_state(color=chess.WHITE)))

# What move to make in this position
policyNetwork = ChessPolicyNetwork()
actionProbs = policyNetwork.action_probabilities(chessEnv.get_state(color=chess.WHITE), chessEnv.get_valid_moves())
print(actionProbs)

# Print the sum of the probabilities
print(np.sum([prob for _, prob in actionProbs]))




