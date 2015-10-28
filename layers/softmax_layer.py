from utils import softmax
import numpy as np


class SoftmaxWithLossLayer(object):

    def __init__(self):
        pass

    def setup(self, prev_layer, input_labels):
        self.input_ = prev_layer.output.squeeze()
        self.input_grad = prev_layer.output_grad.squeeze()
        self.input_labels = input_labels
        self.output = np.empty(self.input_.shape)
        batch_size = self.input_.shape[0]
        self.inc_inds = np.arange(batch_size)

    def loss(self):
        ''' should be called after forward has been called '''
        batch_size = self.input_.shape[0]
        loss = 0.0
        for b in xrange(batch_size):
            loss += -np.log(self.output[b, self.input_labels[b]])
        return loss / float(batch_size)

    def forward(self):
        ''' output produced isn't the loss as one would expect, '''
        ''' but it is the result of the softmax '''
        batch_size = self.input_.shape[0]
        for b in xrange(batch_size):
            self.output[b, :] = softmax(self.input_[b])

    def backward(self):
        impulse = np.zeros(self.output.shape)
        impulse[self.inc_inds, self.input_labels] = 1
        self.input_grad[:] = -(impulse - self.output)
