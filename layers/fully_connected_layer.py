from layer import WeightLayer
import numpy as np


class FullyConnectedLayer(WeightLayer):

    def __init__(self, output_size, lr, l1, l2, init_weights_params):
        self.output_size = output_size
        self.init_weights_params = init_weights_params
        self.lr = lr
        self.l1 = l1
        self.l2 = l2

    def init_weights(self, rand_state):
        if self.init_weights_params['type'] == 'gaussian':
            self.weights[:] = self.init_weights_params[
                'std'] * rand_state.randn(self.weights.shape)
            self.biases[:] = np.full(
                self.output_size, self.init_weights_params['bias'])
        else:
            raise ValueError("Unknown type for init_weights_params['type'].")

    def setup(self, rand_state, prev_layer):
        input_shape = prev_layer.output.shape
        self.batch_size = input_shape[0]
        input_size = np.mult(input_shape[1:])

        self.init_weights(rand_state, input_size)

        # input is a view to output of previous layer
        self.input_ = prev_layer.output.reshape((self.batch_size, input_size))
        self.input_grad = prev_layer.output_grad.reshape(
            (self.batch_size, input_size))
        # allocate output
        self.output = np.empty((self.batch_size, self.output_size))
        self.output_grad = np.empty(self.output.shape)
        # allocate weights
        self.weights = np.empty((input_size, self.output_size))
        self.biases = np.empty(self.output_size)
        self.weights_grad = np.zeros(self.weights.shape)
        self.biases_grad = np.zeros(self.biases.shape)
        self.init_weights(rand_state)

    def forward(self):
        self.batch_size = self.input_.shape[0]
        for i in xrange(self.batch_size):
            self.output[i, :] = (self.input_[i].dot(self.weights) +
                                 self.biases[i])

    def backward(self):
        self.batch_size = self.input_.shape[0]
        for i in xrange(self.batch_size):
            self.input_grad[i, :] = self.output_grad[i].dot(self.weights.T)
            self.weights_grad += np.outer(self.input_[i], self.output_grad[i])
            self.biases_grad += self.output_grad[i]
        self.weights_grad = (
            self.weights_grad / float(self.batch_size) + self.l2 * self.weights
            + self.l1 * np.sign(self.weights))
