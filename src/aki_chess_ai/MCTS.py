# Monte Carlo Tree Search (MCTS) for chess
import math

import numpy as np

import utils

from aki_chess_ai.main import ChessEnv
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork


class Node:
    def __init__(self, prior, to_play):
        # Prior probability of taking this node (from policy network)
        self.prior = prior
        # 1 for white, -1 for black (the player who is to play)
        self.to_play = to_play

        # Number of times node was visited
        self.visit_count = 0
        # Sum of values of all visits
        self.value_sum = 0
        self.children = {}

        # Board state -> as FEN representation
        self.state = None

    def select_child(self):
        """
               Select the child with the highest UCB score.
               """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self):
        return len(self.children) > 0

    def expand(self, state, to_play, action_probs):
        self.state = state
        self.to_play = to_play

        for action, prob in action_probs:
            if prob != 0:
                self.children[action] = Node(prob, -to_play)

    def select_action(self, temperature=1):

        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if(actions is None or len(actions) == 0):
            return None
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def __repr__(self):
        return "Node: {}, {}, {}".format(self.prior, self.value(), self.visit_count)


# https://towardsdatascience.com/the-upper-confidence-bound-ucb-bandit-algorithm-c05c2bf4c13f
# https://github.com/JoshVarty/AlphaZeroSimple/blob/master/monte_carlo_tree_search.py (Josh Varty)
def ucb_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class MCTS:
    # Value Network -> predicts who will win the game
    # Policy Network -> predicts the best move to make in a given state
    def __init__(self, policy_model: ChessPolicyNetwork, value_model: ChessValueNetwork, game: ChessEnv, args):
        self.policy_model = policy_model
        self.value_model = value_model
        self.game = game
        self.args = args

    # to_play -> 1 for white, -1 for black
    # state -> FEN representation of the board
    def run(self, state, to_play):
        root = Node(0, to_play)

        # a,p -> action, probability list
        game_state = utils.getStateFromFEN(state, to_play)

        action_probs = self.policy_model.action_probabilities(game_state, self.game.get_valid_moves_from_state(state,to_play))
        # value = self.value_model.predict(game_state, to_play)

        root.expand(state, to_play, action_probs)
        # debug print print(root.children)

        for _ in range(self.args["simulations"]):
            node = root
            search_path = [node]

            while node.expanded():
                # Select the child with the highest UCB score
                action, node = node.select_child()
                # print(action, node)
                search_path.append(node)
            # print(search_path)
            if (len(search_path) < 2):
                continue
            parent = search_path[-2]
            state = parent.state

            # Expand the node

            # My Board -> do move -> action = move
            next_state, _ = self.game.get_next_state_for_game(state, action)
            # Board after my move in next_state
            # print(utils.getStateFromFEN(next_state, to_play))

            # Enemy Board
            value = self.game.get_reward_for_player(next_state, to_play)
            if value is None:
                action_probs = self.policy_model.action_probabilities(next_state,
                                                                      self.game.get_valid_moves_from_state(next_state))
                value = self.value_model.predict(next_state, to_play)
                node.expand(next_state, -parent.to_play, action_probs)
            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
