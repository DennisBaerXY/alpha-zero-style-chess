import logging

import torch

from aki_chess_ai.main import ChessEnv
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork

from trainer import Trainer
import os


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    args = {
        "batch_size": 64,
        "iterations": 500,  # Total number of training iterations
        "simulations": 10,  # Total number of MCTS simulations to run when deciding on a move to play
        "episodes": 50,  # Number of full games (episodes) to run during each iteration
        "epochs": 2,  # Number of epochs of training per iteration
        "checkpoint_path": "training_models",  # location to save latest set of weights
        "debug": True
    }

    game = ChessEnv()
    valueNetwork = ChessValueNetwork()
    policyNetwork = ChessPolicyNetwork()

    trainer = Trainer(game, value_model=valueNetwork, policy_model=policyNetwork, args=args)
    trainer.learn()


if __name__ == "__main__":
    main()
