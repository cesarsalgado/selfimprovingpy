
class SGD_Solver(object):
    def __init__(self, params, net):
        self.momentum_decay = params['momentum_decay']
        self.lr = params['learning_rate']
        self.max_epochs = params['max_epochs']

    def train(self, net):
        max_iterations = self.max_epochs*net.layers[0].get_n_samples()
        for i in xrange():
            net.forward()
            loss = net.backward()
            for layer in net.layers:
                if layer.has_weights():
                    layer.weights = layer.weights - self.lr*layer.weights_grad
                    layer.reset_weights_grad()
                    if layer.has_bias():
                        layer.biases = layer.biases - self.lr*layer.biases_grad
                        layer.reset_weights_grad()
