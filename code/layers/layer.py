class Layer(object):
  #TODO: initialize an attribute in object initialization
  # to avoid keep calling hasattr everytime. 
  # Is this overhead negligible? Inside the solver loop?
  def has_weights(self):
    return hasattr(self, 'weights')
  
  def has_bias(self):
    return hasattr(self, 'bias')


class WeightLayer(Layer):
  def has_weights(self):
    return True

  def reset_weights_grad(self):
    self.weights_grad[:] = 0

  def reset_biases_grad(self):
    self.biases_grad[:] = 0
