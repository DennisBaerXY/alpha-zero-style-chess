import datetime
import os
import numpy as np
from random import shuffle

import tensorflow as tf
from keras import callbacks
from keras.optimizers import Adam
from tensorflow.python.ops import summary_ops_v2

from aki_chess_ai.MCTSThreaded import MCTS_Threaded
from aki_chess_ai.main import ChessEnv
from aki_chess_ai.MCTS import MCTS, Node
from aki_chess_ai.ChessValueNetwork import ChessValueNetwork
from aki_chess_ai.ChessPolicyNetwork import ChessPolicyNetwork
import utils
import time
import glob

import gc


class Trainer:
    def __init__(self, game: ChessEnv, value_model: ChessValueNetwork, policy_model: ChessPolicyNetwork, args):
        self.game = game
        self.value_model = value_model
        self.policy_model = policy_model
        self.args = args

        self.loss_pi = tf.keras.losses.CategoricalCrossentropy()
        self.loss_v = tf.keras.losses.MeanSquaredError()
        self.value_optimizer = Adam(learning_rate=5e-4)
        self.policy_optimizer = Adam(learning_rate=5e-4)

        # Initialize TensorBoard callback
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        self.global_step = 0

    def execute_episode(self):
        train_examples = []
        current_player = 1
        state = self.game.get_state()

        episode_step = 0
        time_start = time.time()


        gc.collect()
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        tf.compat.v1.reset_default_graph()  # Reset TensorFlow default graph

        while True:
            root = MCTS_Threaded(state,itermax=4,policy_model=self.policy_model,value_model=self.value_model,max_depth=10)
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
            if (self.args["debug"]):
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
            print("EPOCH ::: " + str(i + 1))
            iteration_train_examples = []
            for _ in range(self.args["episodes"]):
                print("Executing episode... {} / {}".format(_ + 1, self.args["episodes"]))
                iteration_train_examples.extend(self.execute_episode())
            shuffle(iteration_train_examples)
            self.train(iteration_train_examples)

            self.save_checkpoint()


    # @tf.function
    def train_step(self, boards, target_pis, target_vs, epoch):
        with tf.GradientTape() as value_tape, tf.GradientTape() as policy_tape:
            out_pi = self.policy_model.model(boards, training=True)
            out_v = self.value_model.model(boards, training=True)
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v = self.loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

        value_gradients = value_tape.gradient(total_loss, self.value_model.model.trainable_variables)
        policy_gradients = policy_tape.gradient(total_loss, self.policy_model.model.trainable_variables)

        self.value_optimizer.apply_gradients(zip(value_gradients, self.value_model.model.trainable_variables))
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_model.model.trainable_variables))


        with self.train_summary_writer.as_default():
            tf.summary.scalar('Loss Policy', l_pi, step=self.global_step)
            tf.summary.scalar('Loss Value', l_v, step=self.global_step)

        self.global_step += 1
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

                l_pi, l_v, out_pi, out_v = self.train_step(boards, target_pis, target_vs, epoch)

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
        value_checkpoints = glob.glob(os.path.join(value_folder, 'model_*.h5'))
        policy_checkpoints = glob.glob(os.path.join(policy_folder, 'model_*.h5'))
        checkpointNumberPolicy = 0
        checkpointNumberValue = 0

        if len(value_checkpoints) > 0:
            value_checkpoints.sort()
            checkpointNumberPolicy = int(value_checkpoints[-1].split('_')[-1].split('.')[0]) + 1
        if len(policy_checkpoints) > 0:
            policy_checkpoints.sort()
            checkpointNumberValue = int(policy_checkpoints[-1].split('_')[-1].split('.')[0]) + 1



        value_filepath = os.path.join(folder, f"value_training_models/model_{checkpointNumberValue}.h5")
        policy_filepath = os.path.join(folder, f"policy_training_models/model_{checkpointNumberPolicy}.h5")

        self.value_model.model.save_weights(value_filepath)
        self.policy_model.model.save_weights(policy_filepath)

        print("Model saved in file: %s" % value_filepath)
        print("Model saved in file: %s" % policy_filepath)

    def load_latest_checkpoint(self, folder):
        print("Loading latest checkpoint...")
        policy_folder = os.path.join(folder, "policy_training_models")
        value_folder = os.path.join(folder, "value_training_models")

        if not os.path.exists(policy_folder) or not os.path.exists(value_folder):
            return

        value_checkpoints = glob.glob(os.path.join(value_folder, 'model_*.h5'))
        policy_checkpoints = glob.glob(os.path.join(policy_folder, 'model_*.h5'))

        if len(value_checkpoints) == 0 or len(policy_checkpoints) == 0:
            print("No Network checkpoints found")
            return

        # Find the latest checkpoint (highest number in the filename)
        latest_value_checkpoint = max(value_checkpoints, key=os.path.getctime)
        latest_policy_checkpoint = max(policy_checkpoints, key=os.path.getctime)

        self.value_model.model.load_weights(latest_value_checkpoint)
        self.policy_model.model.load_weights(latest_policy_checkpoint)

        print("Loaded Value checkpoint from: ", latest_value_checkpoint)
        print("Loaded Policy checkpoint from: ", latest_policy_checkpoint)
        print("Loading checkpoint done.")

