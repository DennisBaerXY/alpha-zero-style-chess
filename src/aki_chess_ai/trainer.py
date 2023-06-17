import datetime
import multiprocessing
import os
import numpy as np
from random import shuffle
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from aki_chess_ai.MCTSThreaded import MCTS_Threaded
from aki_chess_ai.main import ChessEnv
from aki_chess_ai.MCTS import MCTS, Node
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork

import utils
import time
import glob

import gc


def execute_episode_func(game, policy_model, value_model, args):
    logging.info(f'Worker {multiprocessing.current_process().name} is doing something...')
    train_examples = []
    current_player = 1
    state = game.get_state()

    episode_step = 0
    time_start = time.time()

    gc.collect()
    torch.cuda.empty_cache()  # Clear GPU memory
    print("Starting new episode")
    while True:
        root = MCTS_Threaded(state, itermax=4, policy_model=policy_model, value_model=value_model,
                             max_depth=10)
        action_probs = np.zeros(4096)
        for action, node in root.children.items():
            # convert action to index -> action is uci string
            action_probs[utils.move_to_index(action)] = node.visits
        # Normalize
        action_probs /= np.sum(action_probs)
        # Record training example
        train_examples.append([utils.getStateFromFEN(state, current_player), current_player, action_probs])

        # Make move
        action = root.select_action()
        if action is None:
            print("No action selected")

        state, current_player = game.get_next_state_for_game(state, action)
        if args["debug"]:
            print("Move {} has been done by {}. Current FEN {} ".format(action,
                                                                        "White" if -current_player == 1 else "Black",
                                                                        state))
        reward = game.get_reward_for_player(state, current_player)

        episode_step += 1
        if reward is not None or episode_step > 200:
            if reward is None:
                reward = 0
            print("Reward: ", reward)
            print("Episode step: ", episode_step)
            print("Time: ", time.time() - time_start)

            # Game ended
            return [(x[0], x[2], reward * ((-1) ** (x[1] != current_player))) for x in train_examples]


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

        self.global_step = 0

    def execute_episode(self):
        train_examples = []
        current_player = 1
        state = self.game.get_state()

        episode_step = 0
        time_start = time.time()

        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU memory

        while True:
            root = MCTS_Threaded(state, itermax=4, policy_model=self.policy_model, value_model=self.value_model,
                                 max_depth=10)
            action_probs = np.zeros(4096)
            for action, node in root.children.items():
                # convert action to index -> action is uci string
                action_probs[utils.move_to_index(action)] = node.visits
            # Normalize
            action_probs /= np.sum(action_probs)
            # Record training example
            train_examples.append([utils.getStateFromFEN(state, current_player), current_player, action_probs])

            # Make move
            action = root.select_action()
            if action is None:
                print("No action selected")

            state, current_player = self.game.get_next_state_for_game(state, action)
            if self.args["debug"]:
                print("Move {} has been done by {}. Current FEN {} ".format(action,
                                                                            "White" if -current_player == 1 else "Black",
                                                                            state))
            reward = self.game.get_reward_for_player(state, current_player)

            episode_step += 1
            if reward is not None or episode_step > 200:
                if reward is None:
                    reward = 0
                print("Reward: ", reward)
                print("Episode step: ", episode_step)
                print("Time: ", time.time() - time_start)

                # Game ended
                return [(x[0], x[2], reward * ((-1) ** (x[1] != current_player))) for x in train_examples]

    def learn(self):
        print("Starting training...")

        self.load_latest_checkpoint(folder=".")

        for i in range(self.args["iterations"]):
            print("ITERATION ::: " + str(i + 1))
            iteration_train_examples = []

            for episode in range(self.args["episodes"]):
                episode_result = execute_episode_func(self.game, self.policy_model, self.value_model, self.args)
                iteration_train_examples.extend(episode_result)

            print("Number of examples: ", len(iteration_train_examples))

            shuffle(iteration_train_examples)
            self.train(iteration_train_examples)

            self.save_checkpoint()

    def train_step(self, boards, target_pis, target_vs):
        boards = boards.to(self.device)
        target_pis = target_pis.to(self.device)
        target_vs = target_vs.to(self.device)

        self.policy_model.train()
        self.value_model.train()

        out_pi = self.policy_model(boards)
        out_v = self.value_model(boards)

        l_pi = self.loss_pi(target_pis, out_pi)
        l_v = self.loss_v(target_vs, out_v)

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss = l_pi + l_v
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        self.writer.add_scalar('Loss Policy', l_pi.item(), self.global_step)
        self.writer.add_scalar('Loss Value', l_v.item(), self.global_step)

        return l_pi.item(), l_v.item(), out_pi, out_v

    def train(self, examples):
        pi_losses = []
        v_losses = []

        shuffle(examples)
        for epoch in range(self.args["epochs"]):
            batch_idx = 0
            shuffle(examples)
            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.tensor(boards, dtype=torch.float32, device=self.device)
                target_pis = torch.tensor(pis, dtype=torch.float32, device=self.device)
                target_vs = torch.tensor(vs, dtype=torch.float32, device=self.device)

                l_pi, l_v, out_pi, out_v = self.train_step(boards, target_pis, target_vs)
                self.global_step += 1

                pi_losses.append(l_pi)
                v_losses.append(l_v)

                batch_idx += 1

            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))

    def loss_pi(self, targets, outputs):
        loss = -torch.sum(targets * torch.log(outputs), dim=1)
        return torch.mean(loss)

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)
        return loss

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
