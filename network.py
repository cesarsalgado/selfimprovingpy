""" This module contains the Net class.
"""

from layers.data_layer import DataLayer
from layers.loss_layer import LossLayer
from itertools import izip

# TODO Make a option to extract a test net from a trainable net.
# Also try to create two new classes Trainable net and inference net.


class Net(object):
    """ A Net object should be used to build neural nets.
        You must specify a list of layer to be used to build the net.
    """

    def setup(self, rand_state, layers):
        """ This method connects the layers passed as argument and builds the
            net. The first layer should be a DataLayer.

            Args:
                rand_state (numpy.random.RandomState): rand_state is used to
                generate random numbers in every layer of the net. Always
                initialize the rand_state with the same seed to produce the same
                results.
                layers (List[Layer]): list of Layer objects.
        """
        self.layers = layers
        self.trainable = self.is_trainable()
        layers[0].setup(rand_state, None)
        for prev_layer, layer in izip(layers[:-1], layers[1:]):
            layer.setup(rand_state, prev_layer)

    def is_trainable(self):
        if not isinstance(self.layers[0], DataLayer):
            return False
        elif not isinstance(self.layers[-1], LossLayer):
            return False
        else:
            return True

    def loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += layer.loss()
        return loss

    def forward(self):
        for layer in self.layers:
            layer.forward()

    def backward(self):
        if self.is_trainable():
            for layer in self.layers:
                layer.backward()
