import os
import numpy as np
from random import shuffle

import tensorflow as tf
from keras import callbacks
from keras.optimizers import Adam

from aki_chess_ai.main import ChessEnv
from aki_chess_ai.MCTS import MCTS, Node
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
import utils


class Trainer:
    def __init__(self, game: ChessEnv, value_model: ChessValueNetwork, policy_model: ChessPolicyNetwork, args):
        self.game = game
        self.value_model = value_model
        self.policy_model = policy_model
        self.args = args
        self.mcts = MCTS(self.policy_model, self.value_model, self.game, args)
        self.loss_pi = tf.keras.losses.CategoricalCrossentropy()
        self.loss_v = tf.keras.losses.MeanSquaredError()
        self.value_optimizer = Adam(learning_rate=5e-4)
        self.policy_optimizer = Adam(learning_rate=5e-4)

    def execute_episode(self):
        train_examples = []
        current_player = 1
        state = self.game.get_state()


        while True:
            self.mcts = MCTS(self.policy_model, self.value_model, self.game, self.args)
            root = self.mcts.run(state, current_player)

            action_probs = np.zeros((self.game.get_action_size()))
            for action, node in root.children.items():
                # convert action to index -> action is uci string
                action_probs[self.game.action_to_index(action)] = node.visit_count
            # Normalize
            action_probs /= np.sum(action_probs)
            # Record training example
            train_examples.append([utils.getStateFromFEN(state, current_player), current_player, action_probs])

            # Make move
            action = root.select_action()
            if action is None:
                print("No action selected")

            state, current_player = self.game.get_next_state_for_game(state, action)
            if (self.args["debug"]):
                print("Move {} has been done by {}. Current FEN {} ".format(action,
                                                                            "White" if -current_player == 1 else "Black",
                                                                            state))
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                print("Reward: ", reward)
                # Game ended
                return [(x[0], x[2], reward * ((-1) ** (x[1] != current_player))) for x in train_examples]

    def learn(self):
        print("Starting training...")

        for i in range(self.args["iterations"]):
            print("EPOCH ::: " + str(i + 1))
            iteration_train_examples = []
            for _ in range(self.args["episodes"]):
                print("Executing episode... {} / {}".format(_ + 1, self.args["episodes"]))
                iteration_train_examples.extend(self.execute_episode())
            shuffle(iteration_train_examples)
            self.train(iteration_train_examples)
            filename = self.args["checkpoint_path"] + "/model_" + str(i + 1) + ".h5"
            self.save_checkpoint(folder=".", filename=filename)
            print("Model saved in file: %s" % filename)


    @tf.function
    def train_step(self, boards, target_pis, target_vs):
        with tf.GradientTape() as tape:
            out_pi = self.policy_model.model(boards, training=True)
            out_v = self.value_model.model(boards, training=True)
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v = self.loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

        value_gradients = tape.gradient(total_loss, self.value_model.model.trainable_variables)
        policy_gradients = tape.gradient(total_loss, self.policy_model.model.trainable_variables)

        self.value_optimizer.apply_gradients(zip(value_gradients, self.value_model.model.trainable_variables))
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_model.model.trainable_variables))

        return l_pi, l_v, out_pi, out_v
    def train(self, examples):
        pi_losses = []
        v_losses = []

        for epoch in range(self.args["epochs"]):
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = np.array(boards, dtype=np.float32)
                target_pis = np.array(pis, dtype=np.float32)
                target_vs = np.array(vs, dtype=np.float32)

                l_pi, l_v, out_pi, out_v = self.train_step(boards, target_pis, target_vs)

                pi_losses.append(l_pi)
                v_losses.append(l_v)

                batch_idx += 1

            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
    def loss_pi(self, targets, outputs):
        loss = -tf.reduce_sum(targets * tf.math.log(outputs), axis=1)
        return tf.reduce_mean(loss)

    def loss_v(self, targets, outputs):
        loss = tf.reduce_sum(tf.square(targets - tf.reshape(outputs, [-1])), axis=0) / tf.shape(targets)[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        value_filepath = os.path.join(folder, f"value_{filename}")
        policy_filepath = os.path.join(folder, f"policy_{filename}")

        self.value_model.model.save_weights(value_filepath)
        self.policy_model.model.save_weights(policy_filepath)
