""" This file contains the definitions of the abstract Layer class and other
    derivated classes.
"""

import numpy as np


class Layer(object):
    """ This is the abstract base class for all neural nets layers. """

    def has_weights(self):
        return False

    def loss(self):
        return 0.0

    def setup(self, rand_state, prev_layer):
        """ Every subclass from Layer must have a setup method.

            Args:
                rand_state (numpy.random.RandomState): rand_state is used to
                generate random numbers. Some layers don't need it so you can
                pass None or any object and the they will ignore it.

                prev_layer (Layer): This is the previous layer of the net
                which output should be connected to this layer's input. If the
                layer shouldn't have a previous layer use None as argument.
        """
        raise NotImplementedError

    def forward(self):
        """ Every subclass from Layer must have a forward method.
            This method calculates the output of the function that the layer
            represents and stores it in self.output
        """
        raise NotImplementedError

    def backward(self):
        """ Every subclass from Layer must have a backward method.
            This method calculates the gradient of the loss function with
            respect to the layer's input using the gradient of the loss function
            with respect to the layer's output that should have already been
            computed. In addittion, if the layer has weights, the gradient with
            respect to those should also be computed.
        """
        raise NotImplementedError


class WeightLayer(Layer):
    """ This is the abstract base class that should be used for all classes that
        contain weights.
    """

    def __init__(self):
        self.l1, self.l2 = 0, 0  # l1 and l2 weight decay.

    def has_weights(self):
        return True

    def loss(self):
        """ Every weight layer will add a loss equal to the regulariation loss.
            For now l1 and l2 weight decay are accepted.
        """
        return (self.l1 * np.sum(np.abs(self.weights)) + 0.5 * self.l2 *
                np.sum(self.weights ** 2))

    def reset_weights_grad(self):
        self.weights_grad[:] = 0
        self.biases_grad[:] = 0
