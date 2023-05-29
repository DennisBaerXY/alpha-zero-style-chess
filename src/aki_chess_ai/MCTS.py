# Monte Carlo Tree Search (MCTS) for chess
import chess

from aki_chess_ai.main import ChessEnv, ChessValueNetwork, ChessPolicyNetwork


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

        # Board state
        self.state = None

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

    def __repr__(self):
        return "Node: {}, {}, {}".format(self.prior, self.value(), self.visit_count)


class MCTS:
    # Value Network -> predicts who will win the game
    # Policy Network -> predicts the best move to make in a given state
    def __init__(self, policy_model: ChessPolicyNetwork, value_model: ChessValueNetwork, game: ChessEnv):
        self.policy_model = policy_model
        self.value_model = value_model
        self.game = game

    def run(self, state, to_play, num_simulations=100):
        root = Node(0, to_play)

        # a,p -> action, probability list
        action_probs = self.policy_model.action_probabilities(state, self.game.get_valid_moves())
        value = self.value_model.predict(state)

        root.expand(state, to_play, action_probs)
        print(root.children)

        return root


mcts = MCTS(ChessPolicyNetwork(), ChessValueNetwork(), ChessEnv())
mcts.run(ChessEnv().get_state(color=chess.WHITE), 1)
