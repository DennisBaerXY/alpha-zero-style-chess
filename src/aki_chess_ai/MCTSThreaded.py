import time
from collections import defaultdict

import chess
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import math

from aki_chess_ai import utils
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
import gc




class MCTSNode:
    def __init__(self, fen, parent=None):
        self.fen = fen
        self.parent = parent
        self.children = {}
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0
        self.lock = Lock()

    def add_child(self, move, fen):
        child = MCTSNode(fen, parent=self)
        self.children[move] = child
        return child

    def update(self, value):
        with self.lock:
            self.w += value
            self.n += 1
            self.q = self.w / self.n


    def __repr__(self):
        return f"MCTSNode(n={self.n}, q={self.q}, children={self.children})"


def apply_temperature(policy, turn):
    """
    Applies a random fluctuation to probability of choosing various actions
    :param policy: list of probabilities of taking each action
    :param turn: number of turns that have occurred in the game so far
    :return: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp
        = less.
    """
    # Tau decay rate of 0.99
    tau = np.power(0.99, turn + 1)
    if tau < 0.1:
        tau = 0
    if tau == 0:
        action = np.argmax(policy)
        ret = np.zeros(4096)
        ret[action] = 1.0
        return ret
    else:
        ret = np.power(policy, 1 / tau)
        ret /= np.sum(ret)
        return ret


def MCTS_parallel(rootstate, max_iterations, policy_model: ChessPolicyNetwork, value_model, max_depth=12) -> (str,list):
    root_node = MCTSNode(fen=rootstate)

    def worker(root_node):
        with root_node.lock:
            node = root_node
            board = chess.Board(node.fen)

            # Selection and expansion
            if node.children:
                children_values = [
                    (move, child_node, child_node.q + 1.5 * node.p * np.sqrt(node.n) / (1 + child_node.n)) for
                    move, child_node in node.children.items()]
                max_q_plus_u = max(children_values, key=lambda x: x[2])[2]
                max_nodes = [item for item in children_values if item[2] == max_q_plus_u]
                _, node, _ = random.choice(max_nodes)
                board.push(node.move)
            else:
                moves = list(board.legal_moves)
                probabilities = policy_model.action_probabilitiesThreaded(fen_to_input(board.fen(), board.turn), moves)
                selected_move = np.random.choice(moves, p=probabilities)
                board.push(selected_move)
                node = node.add_child(selected_move, board.fen())
                node.p = probabilities[moves.index(selected_move)]

            # Simulation
            depth = 0
            while not board.is_checkmate() and not board.is_stalemate() and depth < max_depth:
                moves = list(board.legal_moves)
                probabilities = policy_model.action_probabilitiesThreaded(fen_to_input(board.fen(), 1 if board.turn else -1), moves)
                selected_move = np.random.choice(moves, p=probabilities)
                board.push(selected_move)
                print(depth, board.fen())
                depth += 1

            # Backpropagation
        value = value_model.predict(fen_to_input(board.fen(), 1 if board.turn else -1), 1 if board.turn else -1)
        node.update(value)
        while node.parent:
            node = node.parent
            node.update(value)

    with ThreadPoolExecutor(max_workers=20) as executor:
        for _ in range(max_iterations):
            executor.submit(worker, root_node)

    executor.shutdown(wait=True)
    policy = calc_policy(root_node)
    zero = np.zeros(4096)
    move_dict = {}
    for action, move in root_node.children.items():
        index = utils.move_to_index(action)
        move_dict[index] = action.uci()
    my_action = int(np.random.choice(range(4096), p=apply_temperature(policy,chess.Board(root_node.fen).fullmove_number)))


    return move_dict[my_action], policy
def fen_to_input(fen, turn):
    return utils.getStateFromFEN(fen, to_play=turn)
def calc_policy(rootnode: MCTSNode):
    """calc π(a|s0)
    :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
    """

    my_visitstats = rootnode
    policy = np.zeros(4096)
    for action, move in my_visitstats.children.items():
        index = utils.move_to_index(action)
        policy[index] = move.n
    policy /= np.sum(policy)


    policy /= np.sum(policy)
    return policy
def main():
    state = chess.Board()
    policy_model = ChessPolicyNetwork()
    value_model = ChessValueNetwork()
    start = time.time()
    root = MCTS_parallel(rootstate=state.fen(), max_iterations=160, policy_model=policy_model, value_model=value_model,
                         max_depth=12)
    duration = time.time() - start
    print(f"Recommended move: {root} ({duration:.2f}s)")


if __name__ == "__main__":
    main()
