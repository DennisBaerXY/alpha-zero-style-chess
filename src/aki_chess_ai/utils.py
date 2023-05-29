# Helper Functions to encode the board state for the neural network
import chess
import numpy as np



def getStateFromChessBoard(board: chess.Board, color=chess.WHITE):
    # chess board is 8x8
    state = np.zeros(64)
    for i in range(64):
        piece: chess.Piece = board.piece_at(i)
        # 0 for a blank square
        if piece is not None:
            state[i] = _encode_piece(piece, color=color)

    # Return the current board state for a nn to use
    return state


# Fen representation to save the state of the board
def getStateFromFEN(fen: str, to_play=1):
    # chess board is 8x8
    color = chess.WHITE if to_play == 1 else chess.BLACK

    board = chess.Board(fen)
    state = np.zeros(64)
    for i in range(64):
        piece: chess.Piece = board.piece_at(i)
        # 0 for a blank square
        if piece is not None:
            state[i] = _encode_piece(piece, color=color)

    # Return the current board state for a nn to use
    return state


def _encode_piece(piece: chess.Piece, color=chess.WHITE):
    # 0.1 for white pawn, 0.2 for white knight, etc.
    # -0.1 for black pawn, -0.2 for black knight, etc.
    return piece.piece_type / 10 * _encode_piece_color(piece, color=color)


def _encode_piece_color(piece: chess.Piece, color=chess.WHITE):
    # 1 for white, -1 for black

    if piece.color == color:
        return 1
    else:
        return -1




def move_to_index(move):
    # Convert UCI move to corresponding index
    from_square = chess.parse_square(move[:2])  # e2e4 -> e2
    to_square = chess.parse_square(move[2:4])  # e2e4 -> e4
    return from_square * 64 + to_square


def softmax(valid_probs):
    # Softmax the probabilities
    exp_valid_probs = np.exp(valid_probs)
    sum_valid_probs = np.sum(exp_valid_probs)
    valid_probs = exp_valid_probs / sum_valid_probs
    return valid_probs