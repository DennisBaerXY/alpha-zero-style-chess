import chess
import numpy as np

from aki_chess_ai import utils
from aki_chess_ai.MCTSThreaded import  MCTS_parallel
from aki_chess_ai.umgebung.ChessEnv import ChessEnv


class ChessPlayer:
    def __init__(self, value_model, policy_model):

        self.value_model = value_model
        self.policy_model = policy_model
        self.moves = []


    def select_move(self, env: ChessEnv) -> str:
        """
        Selects a move based on the current state of the board.
        :param env: chess umgebung to play in

        :return:
        """
        action, policy = MCTS_parallel(env.board.fen(), max_iterations=15, policy_model=self.policy_model,
                                       value_model=self.value_model)
        # Save move
        self.moves.append([env.observation, list(policy)])
        return action

    def reward_moves(self, reward):
        for move in self.moves:
            move[-1] = reward  # add reward to each move

    def reset_moves(self):
        self.moves = []

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.

        :param self:
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]