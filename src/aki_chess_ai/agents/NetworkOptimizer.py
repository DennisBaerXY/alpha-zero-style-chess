import glob
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from random import shuffle

import numpy as np
import torch
from tensorboard.program import TensorBoard
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.umgebung.ChessEnv import testeval
from aki_chess_ai.utils import getStateFromFEN


def getGameDataFilenames():
    # Get all files in ./data/play_data/*
    files = glob.glob(os.path.join("../data/play_data", "*.json"))
    # Sort by creation time
    files.sort(key=os.path.getctime)
    return files


def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    return convert_to_cheating_data(data)
def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'

def convert_to_cheating_data(data):
    """
    :param data: format is SelfPlayWorker.buffer -> [(state_fen, policy, value)]
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:

        network_input = getStateFromFEN(state_fen, 1 if not is_black_turn(state_fen) else -1)
        move_number = int(state_fen.split(' ')[5])
        value_certainty = min(5, move_number)/5 # reduces the noise of the opening...

        # Test eval is the evaluation of the strength of the current position
        sl_value = value*value_certainty + testeval(state_fen, False)*(1-value_certainty)

        state_list.append(network_input)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list, dtype=np.float32)
class NetworkOptimizer:
    def __init__(self, lr=0.001, max_epochs=20):
        """
        :param lr: learning rate
        """
        self.filenames = None
        self.lr = lr
        self.max_epochs = max_epochs
        self.global_step = 0

        self.dataset_size = 100000

        self.policy_model: ChessPolicyNetwork = None
        self.value_model: ChessValueNetwork = None
        self.dataset = deque(), deque(), deque()

    def start(self):
        self.policy_model, self.value_model = self.load_model()
        self.train()

    def train(self):
        """
        Trains the policy and value network.
        :return: schmerzen im debuggen
        """

        # Config the Models for training wiht Optimizer and Loss Function
        self.config_models()
        self.filenames = deque(getGameDataFilenames())
        self.writer = SummaryWriter(log_dir="../logs")
        # Randomize the order of the files
        shuffle(self.filenames)

        total_steps = 0
        total_epochs = 0  # count the total number of epochs
        while total_epochs < self.max_epochs:  # stop when max_epochs is reached
            self.fill_queue()
            steps = self.train_epoch()
            total_steps += steps
            total_epochs += 1
            print(f"Trained {steps} steps in epoch {total_epochs}.")
            if(total_epochs % 10 == 0):
                self.save_current_model()
            a, b, c = self.dataset
            while len(a) > self.dataset_size:
                a.popleft()
                b.popleft()
                c.popleft()

    def train_epoch(self, epochs=1, batch_size=256):
        """
        Trains the model for epochs.

        :epochs: number of epochs to train
        :return: number of steps
        """
        state_array, policy_array, value_array = self.collect_all_loaded_data()

        num_batches = len(state_array) // batch_size
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                # Get the current batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                state_batch = torch.tensor(state_array[start_idx:end_idx])
                policy_batch = torch.tensor(policy_array[start_idx:end_idx])
                value_batch = torch.tensor(value_array[start_idx:end_idx])

                # Train the policy network
                self.policy_model.optimizer.zero_grad()
                policy_pred = self.policy_model(state_batch)
                policy_loss = self.policy_model.loss_function(policy_pred, policy_batch)
                policy_loss.backward()
                self.policy_model.optimizer.step()



                # Train the value network
                self.value_model.optimizer.zero_grad()
                value_pred = self.value_model(state_batch)
                value_batch = value_batch.view(-1, 1)  # reshape value_batch to match value_pred
                value_loss = self.value_model.loss_function(value_pred, value_batch)
                value_loss.backward()
                self.value_model.optimizer.step()

                # Log the losses
                self.writer.add_scalar("Loss/Policy", policy_loss.item(), global_step=self.global_step)
                self.writer.add_scalar("Loss/Value", value_loss.item(), global_step=self.global_step)
                self.global_step += 1


        steps = (state_array.shape[0] //batch_size) * epochs
        return steps


    def load_model(self) -> (ChessPolicyNetwork, ChessValueNetwork):
        policy_model = ChessPolicyNetwork()
        value_model = ChessValueNetwork()

        policy_model, value_model = self.load_latest_checkpoint("../", policy_model, value_model)

        return policy_model, value_model

    def load_latest_checkpoint(self, folder, policy_model, value_model):
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

    def save_current_model(self):
        folder = "../"
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
    def config_models(self):
        opt_policy = Adam(params=self.policy_model.parameters(), lr=self.lr)
        opt_value = Adam(params=self.value_model.parameters(),lr=self.lr)

        self.policy_model.train()
        self.policy_model.optimizer = opt_policy
        self.policy_model.loss_function = torch.nn.CrossEntropyLoss()
        self.policy_model.loss_weight = 1.5
        # loss weight

        self.value_model.train()
        self.value_model.optimizer = opt_value
        self.value_model.loss_function = torch.nn.MSELoss()
        self.value_model.loss_weight = 1

    def fill_queue(self):
        """
        Fill the queue with data from the files in the filenames queue.
        :return:
        """

        futures = deque()
        # Ram hungry thing
        with ProcessPoolExecutor(max_workers=3) as executor:
            for _ in range(3):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                print(f"loading data from {filename}")
                # Append Data to the queue
                futures.append(executor.submit(load_data_from_file, filename))

            while futures and len(self.dataset[0]) < self.dataset_size:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    print(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file, filename))

    def collect_all_loaded_data(self):
        """

        :return: a tuple containing the data in self.dataset, split into
        (state, policy, and value).
        """
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1


def main():
    optimizer = NetworkOptimizer()
    optimizer.start()

if __name__ == "__main__":
    main()