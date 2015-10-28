from layer import Layer
import numpy as np


def calc_output_size(input_size, size, stride):
    return float(input_size - size) / stride + 1


class PoolingLayer(Layer):

    def __init__(self, pool_type, size, stride):
        self.pool_type = pool_type
        self.size = size
        self.stride = stride

    def get_pool_op(self):
        if self.pool_type == 'max':
            return lambda arr: np.max(arr, axis=(2, 3))
        elif self.pool_type == 'avg':
            return lambda arr: np.mean(arr, axis=(2, 3))
        else:
            raise ValueError("Unknown pool_type: %s." % self.pool_type)

    def setup(self, prev_layer):
        input_shape = prev_layer.output.shape
        batch_size, input_channels, input_h, input_w = input_shape

        output_h = calc_output_size(input_h, self.size, self.stride)
        output_w = calc_output_size(input_w, self.size, self.stride)
        if not output_h.is_integer() or not output_w.is_integer():
            raise ValueError(
                "PoolingLayer size/stride configuration doesn't fit nicely in \
                input dimensions.")

        self.input_ = prev_layer.output
        self.input_grad = prev_layer.output_grad
        self.output = np.empty((batch_size, input_channels, output_h, output_w))
        self.output_grad = np.empty(self.output.shape)

        self.pool_op = self.get_pool_op()

    def forward(self):
        oi, oj = 0, 0
        for ki in xrange(0, self.input_.shape[2] - self.size + 1, self.stride):
            for kj in xrange(0, self.input_.shape[3] - self.size + 1,
                             self.stride):
                self.output[:, :, oi, oj] = self.pool_op(
                    self.input_[:, :, ki:ki + self.size, kj:kj + self.size])
                oj += 1
            oi += 1

    def backward(self):
        pass
