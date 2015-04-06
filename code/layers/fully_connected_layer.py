from layer import WeightLayer

class FullyConnectedLayer(WeightLayer):
  def __init__(self, output_size, l1, l2, init_weights_info):
    self.output_size = output_size
    self.init_weights_info = init_weights_info
    self.l1 = l1
    self.l2 = l2

  def init_weights(self, rand_state):
    if self.init_weights_info['type'] == 'gaussian':
      self.weights[:] = self.init_weights_info['std']*rand_state.randn(self.weights.shape)
      self.biases[:] = np.full(self.output_size, self.init_weights_info['bias'])
    else:
      raise ValueError("Unknown type for init_weights_info['type'].")

  def setup(self, rand_state, prev_layer):
    input_shape = prev_layer.output.shape
    input_size = np.mult(input_shape[1:])

    self.init_weights(rand_state, input_size)

    # input is a view to output of previous layer
    self.input_       = prev_layer.output.reshape((batch_size, input_size))
    self.input_grad   = prev_layer.output_grad.reshape((batch_size, input_size))
    # allocate output
    self.output       = np.empty((batch_size, self.output_size))
    self.output_grad  = np.empty(self.output.shape)
    # allocate weights
    self.weights      = np.empty((input_size, self.output_size))
    self.biases       = np.empty(self.output_size)
    self.weights_grad = np.zeros(self.weights.shape)
    self.biases_grad  = np.zeros(self.biases.shape)
    self.init_weights(rand_state)

  def forward(self):
    batch_size = self.input_.shape[0]
    for b in xrange(batch_size):
      self.output[b,:] = self.input_[b].dot(self.weights) + self.biases[b]

  def backward(self):
    batch_size = self.input_.shape[0]
    for b in xrange(batch_size):
      self.input_grad[b,:] = self.output_grad[b].dot(self.weights.T)
      self.weights_grad += np.outer(self.input_[b], self.output_grad[b]) + 2*self.l2*self.weights + self.l1*sign(self.weights)
      self.biases_grad += self.output_grad[b]
