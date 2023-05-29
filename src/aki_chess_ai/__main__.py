import tensorflow as tf

from aki_chess_ai.main import ChessEnv
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork

from trainer import Trainer
import os

def main():
    args = {
        "batch_size": 64,
        "iterations": 500,  # Total number of training iterations
        "simulations": 10,  # Total number of MCTS simulations to run when deciding on a move to play
        "episodes": 100,  # Number of full games (episodes) to run during each iteration
        "epochs": 5,  # Number of epochs of training per iteration
        "checkpoint_path": "training_models",  # location to save latest set of weights
        "debug": True
    }
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    game = ChessEnv()
    valueNetwork = ChessValueNetwork()
    policyNetwork = ChessPolicyNetwork()

    trainer = Trainer(game, value_model=valueNetwork,policy_model=policyNetwork, args=args)
    trainer.learn()
if __name__ == "__main__":
    main()
