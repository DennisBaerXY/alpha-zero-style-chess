import tensorflow as tf

from aki_chess_ai import utils


class ChessValueNetwork:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            # Input shape for the chessboard state
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            # Output a single value between -1 and 1
            tf.keras.layers.Dense(1, activation="tanh")
        ])
        self.model.compile(optimizer="adam", loss="mse")


    def predict(self, state, to_play):
        if type(state) == str:
            state = utils.getStateFromFEN(state, to_play=to_play)

        # Reshape the state to match the input shape of the model
        state = tf.reshape(state, (1, 64))
        # Predict the value of the state
        value = self.model.predict(state,verbose=0)[0][0]
        return value
