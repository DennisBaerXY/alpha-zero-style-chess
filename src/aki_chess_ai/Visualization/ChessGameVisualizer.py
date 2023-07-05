import glob
import os
import time

import torch
import chess.engine

from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.agents.ChessPlayer import ChessPlayer
from aki_chess_ai.config.Config import Config
from aki_chess_ai.umgebung.ChessEnv import ChessEnv
from chessboard import display


class ChessGameVisualizer:
    def __init__(self, config: Config, policy_model: ChessPolicyNetwork, value_model: ChessValueNetwork):
        self.env = ChessEnv().reset()
        self.white = ChessPlayer(value_model, policy_model)
        self.black = ChessPlayer(value_model, policy_model)
        self.config = config
        self.board = display.start(self.env.board.fen())
        self.score = {"white": 0, "black": 0}
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(config.demonstration.stockfish_path)
        except:
            print("Stockfish not found. Please specify the path to the engine in the config file.")
            print("Download it from here: https://stockfishchess.org/download/\n"
                  "And configurate your path like it is descriped here https://python-chess.readthedocs.io/en/latest/engine.html")

            exit()

    def play_game(self):
        self.reset()
        while not self.env.done:
            if self.env.white_to_move:
                action = self.white.select_move(self.env)
            else:
                action = self.black.select_move(self.env)

            print(f"Move: {action}")
            self.env.step(action)
            self.evaluation()
            self.update_chess_board()

            time.sleep(self.config.demonstration.time_between_moves)  # add this line to wait for 1 second
            if self.env.num_halfmoves >= self.config.demonstration.max_moves:
                self.env.adjudicate()
        print("Game over.")
        if self.env.winner == 1:
            print("White wins!")
            self.score["white"] += 1

        elif self.env.winner == -1:
            print("Black wins!")
            self.score["black"] += 1
        else:
            print("Draw!")

    def reset(self):
        self.env = ChessEnv().reset()

    def update_chess_board(self):
        display.update(self.env.board.fen(), self.board)
        display.check_for_quit()

    def evaluation(self, time_limit=0.01):
        result = self.engine.analyse(self.env.board, chess.engine.Limit(time=time_limit))
        print(result["score"])


def load_model() -> (ChessPolicyNetwork, ChessValueNetwork):
    policy_model = ChessPolicyNetwork()
    value_model = ChessValueNetwork()

    policy_model, value_model = load_latest_checkpoint("../", policy_model, value_model)

    return policy_model, value_model


def load_latest_checkpoint(folder, policy_model, value_model):
    print("Loading latest checkpoint...")
    policy_folder = os.path.join(folder, "policy_training_models")
    value_folder = os.path.join(folder, "value_training_models")

    value_checkpoints = glob.glob(os.path.join(value_folder, 'model_*.pt'))
    policy_checkpoints = glob.glob(os.path.join(policy_folder, 'model_*.pt'))

    # Find the latest checkpoint (highest number in the filename)
    latest_value_checkpoint = max(value_checkpoints, key=os.path.getctime)
    latest_policy_checkpoint = max(policy_checkpoints, key=os.path.getctime)

    value_model.load_state_dict(torch.load(latest_value_checkpoint))
    policy_model.load_state_dict(torch.load(latest_policy_checkpoint))

    print("Loaded latest value model from:", latest_value_checkpoint)
    print("Loaded latest policy model from:", latest_policy_checkpoint)
    print("Loading checkpoint done.")

    return policy_model, value_model


if __name__ == '__main__':
    policy_model, value_model = load_model()
    config = Config()

    visualizer = ChessGameVisualizer(Config(), policy_model, value_model)
    for i in range(10):
        visualizer.play_game()
    print(visualizer.score)