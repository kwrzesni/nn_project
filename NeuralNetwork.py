from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import activations
import tensorflow as tf


class NeuralNetwork:
    def __init__(self, activation_ind):
        self.model = models.Sequential()
        self.activations_list = [activations.linear, step_function, activations.sigmoid]
        self.model.add(layers.Dense(1, activation=self.activations_list[activation_ind], input_shape=(2,)))
        self.training_history = None

    def train(self, X, y, epochs, learning_rate, momentum):
        self.model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                           loss='mse',
                           metrics=['accuracy'])
        self.training_history = self.model.fit(X, y, batch_size=len(X), epochs=epochs, verbose=0)

    def predict(self, X):
        return self.model.predict([X])[0][0]

    def get_weight(self):
        return [self.model.layers[0].get_weights()[0][0][0], self.model.layers[0].get_weights()[0][1][0]]

    def get_bias(self):
        return self.model.layers[0].get_weights()[1][0]


def step_function(x):
    return 1/(1+tf.exp(-10*x))