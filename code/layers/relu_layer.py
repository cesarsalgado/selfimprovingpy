
class ReluLayer(object):
  def setup(self, prev_layer):
    input_shape = prev_layer.output.shape
    batch_size, input_channels, input_h, input_w = input_shape

    self.input_       = prev_layer.output
    self.input_grad   = prev_layer.output_grad
    self.output       = np.empty((batch_size, self.n_filters, output_h, output_w))
    self.output_grad  = np.empty((batch_size, self.n_filters, output_h, output_w))

  def forward(self):
    self.output[:] = np.maximum(0, self.input_)

  def backward(self):
    self.input_grad[:] = (self.input_>0)*self.output_grad
