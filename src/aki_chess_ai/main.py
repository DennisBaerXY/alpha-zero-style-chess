import chess
import chess.svg
import utils
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork


class ChessEnv:
    def __init__(self) -> None:
        # Init chess board
        self.board = chess.Board()

    # Color Property for better feedability of the network
    # ensures that the network can learn from both sides in self play
    # https://www.youtube.com/watch?v=62nq4Zsn8vc
    def get_state(self):
        return self.board.fen()
    def get_action_size(self):
        return 4096
    def get_next_state(self, action):
        boardcopy = self.board.copy()
        boardcopy.push(chess.Move.from_uci(action))
        return boardcopy.fen()

    def get_next_state_for_game(self, state, action):
        boardcopy = chess.Board(state)
        if isinstance(action,chess.Move):
            boardcopy.push(action)
        else:
            boardcopy.push(chess.Move.from_uci(action))
        return boardcopy.fen(), 1 if boardcopy.turn == chess.WHITE else -1

    def action_to_index(self, action):
        return utils.move_to_index(action)

    def _encode_piece_color(self, piece: chess.Piece, color=chess.WHITE):
        # 1 for white, -1 for black

        if (piece.color == color):
            return 1
        else:
            return -1

    def get_valid_moves(self):
        # Return a list of valid moves
        return [move.uci() for move in self.board.legal_moves]

    def get_valid_moves_from_state(self, state, player=00):
        tmpBoard = chess.Board(state)

        if(tmpBoard.legal_moves.count() == 0):
            print("No legal moves", state , player)
        return [move.uci() for move in tmpBoard.legal_moves]

    def make_move(self, move):
        self.board.push(move)

    def get_reward_for_player(self, state, player):
        boardcopy = chess.Board(state)
        outcome = boardcopy.outcome(claim_draw=True)
        if outcome is None:
            return None

        if outcome.winner == chess.WHITE:
            print("Checkmate white")
            return 1 if player == 1 else -1
        elif outcome.winner == chess.BLACK:
            print("Checkmate black")
            return -1 if player == 1 else 1
        else:
            print("Draw")
            return 0
        # if boardcopy.is_checkmate():
        #     print("Checkmate")
        #     if player == 1:
        #         return 1
        #     else:
        #         return -1
        # if boardcopy.is_stalemate():
        #     print("Stalemate")
        #     return 0.1
        # if boardcopy.is_seventyfive_moves():
        #     print("Seventyfive moves")
        #     return 0
        # if boardcopy.is_fivefold_repetition():
        #     print("Fivefold repetition")
        #     return 0
        # if boardcopy.is_insufficient_material():
        #     print("Insufficient material")
        #     return 0
        # if boardcopy.is_game_over():
        #     print("Game over")
        #     return 0
        # if boardcopy.is_variant_end():
        #     print("Variant end")
        #     return 0
        # if boardcopy.is_variant_draw():
        #     print("Variant draw")
        #     return 0
        # if boardcopy.is_variant_loss():
        #     print("Variant loss")
        #     return 0

        # return None
    def get_canonical_board(self, state, player):
        boardcopy = chess.Board(state)
        boardcopy.turn = player == 1
        return boardcopy.fen()




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




