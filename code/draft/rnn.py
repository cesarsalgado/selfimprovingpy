import numpy as np
from utils import sigmoid, softmax, get_random_batches_inds
import ipdb

class RNN:
  def __init__(self, input_size, hidden_size, output_size, hidden_non_linearity=np.tanh, bias_init=1.0, lr=0.01, momen_decay=0.0, l2=0.0):
    # initial weights
    mul_wih  = np.sqrt(6)/np.sqrt(input_size+hidden_size)
    self.Wih = 2*mul_wih*np.random.rand(input_size, hidden_size)-mul_wih  # input to hidden weights
    self.hb  = bias_init*np.ones((1,hidden_size))                         # bias weights for the hidden layer.
    mul_who  = np.sqrt(6)/np.sqrt(hidden_size+output_size)
    self.Who = 2*mul_who*np.random.rand(hidden_size, output_size)-mul_who # hidden to output weights
    self.ob  = bias_init*np.ones((1,output_size))                         # bias wieghts for the output layer.
    mul_whh  = np.sqrt(6)/np.sqrt(2*hidden_size)
    self.Whh = 2*mul_whh*np.random.rand(hidden_size, hidden_size)-mul_whh # hidden to hidden weights

    # activation matrices
    self.input_act  = np.empty((1,input_size))
    self.hidden_act = np.zeros((1,hidden_size))
    self.output_act = np.empty((1,output_size))

    # learning rate
    self.lr = lr
    
    # l2 weight decay
    self.l2 = l2

    # momentum decay (normally called just momentum)
    self.momen_decay = momen_decay

    self.hidden_non_linearity = hidden_non_linearity

  def step_forward(self, x):
    self.input_act[:] = x
    pre_act = self.input_act.dot(self.Wih) + self.hidden_act.dot(self.Whh) + self.hb
    self.hidden_act[:] = self.hidden_non_linearity(pre_act)
    print self.hidden_act
    self.output_act[:] = softmax(np.dot(self.hidden_act, self.Who) + self.ob)
    return self.output_act.copy()

  def sequence_forward(self, xs):
    output_seq = []
    for x in xs:
      output_seq.append(self.step_forward(x))
    return output_seq
