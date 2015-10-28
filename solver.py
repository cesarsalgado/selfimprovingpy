class SGD_Solver(object):

    def __init__(self, params, net):
        """ This is a doc string and bla bla.
            I'm just testing shit now.
            Anther new line.
        """
        self.momentum_decay = params['momentum_decay']
        self.lr = params['learning_rate']
        self.max_epochs = params['max_epochs']

    def train(self, net):
        """ This is a doc string and bla bla.
            I'm just testing shit now.
            Anther new line.
        """
        data_layer = net.layers[0]
        max_iterations = (self.max_epochs*data_layer.get_n_samples()) \
            / data_layer.batch_size
        for i in xrange(max_iterations):
            yp = net.forward()
            loss = net.backward()
            for layer in net.layers:
                if layer.has_weights():
                    layer.weights = layer.weights - self.lr * layer.weights_grad
                    layer.biases = layer.biases - self.lr * layer.biases_grad
                    layer.reset_weights_grad()
