import time

import chess
import tensorflow as tf
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import math

from aki_chess_ai import utils
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
import gc

class Node:
    def __init__(self, fen, parent=None):
        self.fen = fen  # fen is board state
        self.turn = 1 if chess.Board(fen).turn else -1
        self.parent = parent
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.untried_actions = list(chess.Board(fen).legal_moves)
        self.lock = Lock()
        self.prior = 0  # prior is initialized to 0 but will be updated by the policy network

    def ucb_score(self):
        prior_score = self.prior * math.sqrt(self.parent.visits) / (self.visits + 1)
        if self.visits > 0:
            # The value of the child is from the perspective of the opposing player
            value_score = -self.turn * self.value()
        else:
            value_score = 0

        return value_score + prior_score

    def select_child(self):
        s = sorted(self.children.values(), key=lambda c: c.ucb_score())[-1]
        return s

    def add_child(self, a, fen):
        n = Node(fen=fen, parent=self)
        self.untried_actions.remove(a)
        self.children[a] = n
        return n

    def update(self, result):
       with self.lock:
            # print(f"Updating node with result: {result}")
            self.visits += 1
            self.wins += result


    def value(self):
        return self.wins / self.visits if self.visits != 0 else 0
    def select_action(self):
        return sorted(self.children.items(), key=lambda c: c[1].visits)[-1][0]

    def __repr__(self):

        return f"Node(fen={self.fen}, turn={self.turn}, visits={self.visits}, wins={self.wins})\n"

def board_to_input(board,turn):
    return utils.getStateFromFEN(board.fen(),to_play=turn)

def MCTS_Threaded(rootstate, itermax, policy_model, value_model,max_depth=12):
    rootnode = Node(fen=rootstate)

    def worker(rootnode):
        # print("Starting worker")
        node = rootnode
        state: chess.Board = chess.Board(node.fen)
        # print("Created state")

        # Select
        while node.untried_actions == [] and node.children != {}:
            node = node.select_child()
            state.push(node.action)

        # Expand
        if node.untried_actions != []:
            # Use policy network to decide the action
            actions = list(state.legal_moves)
            action_probs = policy_model.action_probabilitiesThreaded(board_to_input(state,node.turn), actions)
            a = np.random.choice(actions, p=action_probs)
            state.push(a)
            node = node.add_child(a, state.fen())


        # Rollout
        depth = 0
        while not state.is_checkmate() and not state.is_stalemate() and depth < max_depth:
            # Use policy network to decide the action
            actions = list(state.legal_moves)
            action_probs = policy_model.action_probabilitiesThreaded(board_to_input(state, 1 if state.turn else -1), actions)
            a = np.random.choice(actions, p=action_probs)
            state.push(a)
            depth += 1



        # Backpropagate
        # print("About to start backpropagation")
        while node is not None:
            # print(f"Backpropagating node: {node}")
            state = chess.Board(node.fen)

            # If the game is not over, use the predicted value
            if not state.is_game_over():
                value = value_model.predict(board_to_input(state, 1 if state.turn else -1),1 if state.turn else -1)
                node.update(value)
            # If the game is over, use the actual outcome
            else:
                outcome = 1 if state.result() == '1-0' else -1 if state.result() == '0-1' else 0
                node.update(outcome)

            node = node.parent

    with ThreadPoolExecutor(max_workers=20) as executor:
        for _ in range(itermax):
            executor.submit(worker, rootnode)

    # Wait for all threads to finish
    executor.shutdown(wait=True)


    # return sorted(rootnode.children.items(), key=lambda c: c[1].visits)[-1][0]
    # Return root
    return rootnode



def main():
    state = chess.Board()
    policy_model = ChessPolicyNetwork()
    value_model = ChessValueNetwork()
    start = time.time()
    root = MCTS_Threaded(rootstate=state.fen(), itermax=5, policy_model=policy_model, value_model=value_model,max_depth=12)
    duration = time.time() - start
    print(f"Recommended move: {root.select_child()} ({duration:.2f}s)")

if __name__ == "__main__":
    main()