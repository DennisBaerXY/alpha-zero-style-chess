import datetime
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

import chess
import chess.pgn


import pyperclip
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork

import utils
import time
import glob

import gc

from aki_chess_ai.agnets.ChessPlayer import ChessPlayer
from aki_chess_ai.config.Config import Config
from aki_chess_ai.env.ChessEnv import ChessEnv


def execute_episode_func(config: Config,policy_model: ChessPolicyNetwork, value_model: ChessValueNetwork):
    env = ChessEnv().reset()

    white = ChessPlayer(value_model, policy_model)
    black = ChessPlayer(value_model, policy_model)

    while not env.done:

        if env.white_to_move():
            action = white.select_move(env)
        else:
            action = black.select_move(env)

        env.step(action)
        if env.num_halfmoves >= config.max_game_length:
            env.adjudicate()

    if env.winner == 1:
        white_win = 1
    elif env.winner == -1:
        white_win = -1

    else:
        white_win = 0

    white.finish_game(white_win)
    black.finish_game(-white_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])
    return env, data


def pretty_print(env, colors):
    new_pgn = open("test3.pgn", "at")
    game = chess.pgn.Game.from_board(env.board)
    game.headers["Result"] = env.result
    game.headers["White"], game.headers["Black"] = colors
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    new_pgn.write(str(game) + "\n\n")
    new_pgn.close()
    pyperclip.copy(env.board.fen())


class Trainer:
    def __init__(self, game: ChessEnv, value_model: ChessValueNetwork, policy_model: ChessPolicyNetwork, args):
        self.game = game
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_model = value_model.to(self.device)
        self.policy_model = policy_model.to(self.device)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=5e-4)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=5e-4)

        # Initialize TensorBoard writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/gradient_tape/{current_time}/train'
        self.writer = SummaryWriter(train_log_dir)
        self.buffer = []


        self.config = Config()

        self.global_step = 0



    def learn(self):
        print("Starting training...")
        self.load_latest_checkpoint(folder=".")

        self.buffer = []
        for i in range(self.args["iterations"]):
            print("ITERATION ::: " + str(i + 1))
            iteration_train_examples = deque()

            with ProcessPoolExecutor(max_workers=self.config.max_processes) as executor:
                for _ in range(self.config.max_processes * 2):
                    iteration_train_examples.append(
                        executor.submit(execute_episode_func, self.config, self.policy_model, self.value_model))
                game_idx = 0    # Game index
                while True:
                    game_idx += 1
                    start_time = time.time()
                    env, data = iteration_train_examples.popleft().result()

                    print(f"game {game_idx:3} time={time.time() - start_time:5.1f}s "
                          f"halfmoves={env.num_halfmoves:3} {env.winner:12} "
                          f"{'by resign ' if env.resigned else '          '}")
                    pretty_print(env, ("current_model", "current_model"))
                    self.buffer += data

                    if(game_idx % self.config.max_game_before_training == 0):
                        self.flush_buffer()

                    iteration_train_examples.append(
                        executor.submit(execute_episode_func, self.config, self.policy_model, self.value_model))
                    # if len(data) > 0:
                    #     self.train(data)
                    #     self.save_checkpoint()
                    #     if len(iteration_train_examples) == 0:
                    #         break

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """
        rc = self.config.resource
        game_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)

        if not os.path.exists(rc.play_data_dir):
            os.makedirs(rc.play_data_dir)
        print(f"save play data to {path}")
        thread = Thread(target=utils.write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def save_checkpoint(self):
        folder = "."
        # Check if folder exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        policy_folder = os.path.join(folder, "policy_training_models")
        value_folder = os.path.join(folder, "value_training_models")

        if not os.path.exists(policy_folder):
            os.makedirs(policy_folder)

        if not os.path.exists(value_folder):
            os.makedirs(value_folder)

        # Get the latest checkpoint number
        value_checkpoints = glob.glob(os.path.join(value_folder, 'model_*.pt'))
        policy_checkpoints = glob.glob(os.path.join(policy_folder, 'model_*.pt'))
        checkpointNumberPolicy = 0
        checkpointNumberValue = 0

        if len(value_checkpoints) > 0:
            value_checkpoints.sort()
            checkpointNumberPolicy = int(value_checkpoints[-1].split('_')[-1].split('.')[0]) + 1
        if len(policy_checkpoints) > 0:
            policy_checkpoints.sort()
            checkpointNumberValue = int(policy_checkpoints[-1].split('_')[-1].split('.')[0]) + 1

        value_filepath = os.path.join(folder, f"value_training_models/model_{checkpointNumberValue}.pt")
        policy_filepath = os.path.join(folder, f"policy_training_models/model_{checkpointNumberPolicy}.pt")

        torch.save(self.value_model.state_dict(), value_filepath)
        torch.save(self.policy_model.state_dict(), policy_filepath)

        print("Model saved in file:", value_filepath)
        print("Model saved in file:", policy_filepath)

    def load_latest_checkpoint(self, folder):
        print("Loading latest checkpoint...")
        policy_folder = os.path.join(folder, "policy_training_models")
        value_folder = os.path.join(folder, "value_training_models")

        if not os.path.exists(policy_folder) or not os.path.exists(value_folder):
            return

        value_checkpoints = glob.glob(os.path.join(value_folder, 'model_*.pt'))
        policy_checkpoints = glob.glob(os.path.join(policy_folder, 'model_*.pt'))

        if len(value_checkpoints) == 0 or len(policy_checkpoints) == 0:
            print("No network checkpoints found")
            return

        # Find the latest checkpoint (highest number in the filename)
        latest_value_checkpoint = max(value_checkpoints, key=os.path.getctime)
        latest_policy_checkpoint = max(policy_checkpoints, key=os.path.getctime)

        self.value_model.load_state_dict(torch.load(latest_value_checkpoint, map_location=self.device))
        self.policy_model.load_state_dict(torch.load(latest_policy_checkpoint, map_location=self.device))

        print("Loaded Value checkpoint from:", latest_value_checkpoint)
        print("Loaded Policy checkpoint from:", latest_policy_checkpoint)
        print("Loading checkpoint done.")
