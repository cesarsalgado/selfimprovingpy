from data_layer import DataLayer
from itertools import izip

class Net(object):
  def setup(self, layers, rand_state):
    self.layers = layers
    if not isinstance(layers[0], DataLayer):
      raise ValueError("First layer must be a DataLayer.")
    layers[0].setup(rand_state)
    for prev_layer, layer in izip(layers[:-1],layers[1:]):
      layer.setup(rand_state, prev_layer)

  def forward(self):
    for layer in self.layers:
      layer.forward()

  def backward(self):
    for layer in self.layers:
      layer.backward()
