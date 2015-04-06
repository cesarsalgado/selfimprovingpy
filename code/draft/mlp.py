import numpy as np
from sklearn.datasets import load_digits
from utils import sigmoid, softmax, get_random_batches_inds
import ipdb

class Sigmoidal2LayerMLP_WithSoftmax:
  def __init__(self, input_size, hidden_size, output_size, bias_init=1.0, lr=0.01, momen_decay=0.0, l2=0.0):
    # initial weights
    mul_wih  = np.sqrt(6)/np.sqrt(input_size+hidden_size)
    self.Wih = 2*mul_wih*np.random.rand(input_size, hidden_size)-mul_wih  # input to hidden weights
    self.hb  = bias_init*np.ones((1,hidden_size))                         # bias weights for the hidden layer.
    mul_who  = np.sqrt(6)/np.sqrt(hidden_size+output_size)
    self.Who = 2*mul_who*np.random.rand(hidden_size, output_size)-mul_who # hidden to output weights
    self.ob  = bias_init*np.ones((1,output_size))                         # bias wieghts for the output layer.

    # gradient wrt weights matrices
    self.Wih_grad = np.zeros(self.Wih.shape)
    self.hb_grad  = np.zeros(self.hb.shape)                     
    self.Who_grad = np.zeros(self.Who.shape)
    self.ob_grad  = np.zeros(self.ob.shape)                     

    # activation matrices
    self.input_act  = np.empty((1,input_size))
    self.hidden_act = np.empty((1,hidden_size))
    self.output_act = np.empty((1,output_size))

    # gradient wrt activations matrices
    self.output_preact_grad = np.empty(self.output_act.shape)
    self.hidden_posact_grad = np.empty(self.hidden_act.shape)
    self.hidden_preact_grad = np.empty(self.hidden_act.shape)

    # learning rate
    self.lr = lr
    
    # l2 weight decay
    self.l2 = l2

    # momentum decay (normally called just momentum)
    self.momen_decay = momen_decay

  def input_size(self):
    return self.Wih.shape[0]

  def output_size(self):
    return self.ob.size

  def forward(self, x):
    self.input_act[:] = x
    self.hidden_act[:] = sigmoid(np.dot(self.input_act, self.Wih) + self.hb)
    self.output_act[:] = softmax(np.dot(self.hidden_act, self.Who) + self.ob)

  def loss(self, yi):
    return -np.log(self.output_act[0,yi]) + self.l2*(np.sum(self.Who**2) + np.sum(self.Wih**2))

  def backward(self, yi):
    impulse = np.zeros((1,self.output_size()))
    impulse[0,yi] = 1
    self.output_preact_grad[:] = -(impulse-self.output_act)
    self.Who_grad[:] += self.output_preact_grad*self.hidden_act.T + 2*self.l2*self.Who
    self.ob_grad[:]  += self.output_preact_grad
    self.hidden_posact_grad[:] = np.dot(self.output_preact_grad, self.Who.T)
    self.hidden_preact_grad[:] = self.hidden_posact_grad*(self.hidden_act*(1-self.hidden_act))
    self.Wih_grad[:] += self.hidden_preact_grad*self.input_act.T + 2*self.l2*self.Wih
    self.hb_grad[:]  += self.hidden_preact_grad
    return self.loss(yi)

  def set_grads_wrt_weights_to_zero(self):
    self.Who_grad[:] = 0 
    self.ob_grad[:]  = 0 
    self.Wih_grad[:] = 0 
    self.hb_grad[:]  = 0 

  def validate(self):
    return self.score(self.Xv, self.yv)

  def score(self, X, y):
    correct_count = 0
    for i, x in enumerate(X):
      self.forward(x)
      yp = self.output_act.argmax()
      if yp == y[i]:
        correct_count += 1
    return float(correct_count)/y.size

  def train(self, X, y, batch_size=100, max_epochs=100, Xv=None, yv=None):
    self.Xv = Xv
    self.yv = yv

    Wih_momemtum  = np.zeros(self.Wih.shape)
    hb_momentum   = np.zeros(self.hb.shape)
    Who_momentum  = np.zeros(self.Who.shape)
    ob_momentum   = np.zeros(self.ob.shape)

    d = X.shape[1]
    if d != self.input_size():
      raise ValueError('X should have the number of column equal to the input_size.')
    global_nsamples = X.shape[0]
    for epoch in xrange(max_epochs):
      if epoch == 350:
        self.lr /= 10.0
      batches_inds = get_random_batches_inds(global_nsamples, batch_size)
      for i, batch_inds in enumerate(batches_inds):
        batch_nsamples = len(batch_inds)
        loss = np.inf
        for idx in batch_inds:
          self.forward(X[idx])
          loss = self.backward(y[idx])
        Wih_momemtum = self.momen_decay*Wih_momemtum - self.lr*(self.Wih_grad/batch_nsamples)
        hb_momentum  = self.momen_decay*hb_momentum  - self.lr*(self.hb_grad /batch_nsamples)
        Who_momentum = self.momen_decay*Who_momentum - self.lr*(self.Who_grad/batch_nsamples)
        ob_momentum  = self.momen_decay*ob_momentum  - self.lr*(self.ob_grad /batch_nsamples)
        self.Wih += Wih_momemtum
        self.hb  += hb_momentum
        self.Who += Who_momentum
        self.ob  += ob_momentum
        self.set_grads_wrt_weights_to_zero()
        #print 'epoch = %d | batch_idx = %d | loss = %.3f' % (epoch, i, loss)
      print '\nepoch = %d | val accu = %.3f | loss = %.3f | |W**2| = %.3f | lr = %.5f | momen_decay = %.3f | l2 = %.5f' % (epoch, self.validate(), loss, np.sum(self.Wih**2) + np.sum(self.Who**2), self.lr, self.momen_decay, self.l2)

def test_on_minst():
  # loading data
  digits = load_digits()
  X = digits['data']
  y = digits['target']

  # dividing in training, validation, and test set
  nsamples = X.shape[0]
  end_train_idx = int(0.5*nsamples)
  end_val_idx = int(0.7*nsamples)
  perm = np.random.permutation(nsamples)
  Xtrain  = X[perm[:end_train_idx]]
  Xval    = X[perm[end_train_idx:end_val_idx]]
  Xtest   = X[perm[end_val_idx:]]
  ytrain  = y[perm[:end_train_idx]]
  yval    = y[perm[end_train_idx:end_val_idx]]
  ytest   = y[perm[end_val_idx:]]

  # data normalization
  mean = Xtrain.mean(0)
  std  = Xtrain.std(0)
  std[std == 0] = 1
  Xtrain = (Xtrain-mean)/std
  Xval = (Xval-mean)/std
  Xtest = (Xtest-mean)/std

  # net params
  input_size = Xtrain.shape[1]
  hidden_size = 60
  output_size = np.unique(y).size
  net = Sigmoidal2LayerMLP_WithSoftmax(input_size, hidden_size, output_size, bias_init=0.0, lr=0.01, momen_decay=0.9, l2=0.0)

  # train params
  net.train(Xtrain, ytrain, batch_size=100, max_epochs=500, Xv=Xval, yv=yval)

  print net.score(Xtest, ytest)

def test_backprop():
  # loading data
  digits = load_digits()
  X = digits['data']
  y = digits['target']

  # dividing in training, validation, and test set
  nsamples = X.shape[0]
  end_train_idx = int(0.5*nsamples)
  end_val_idx = int(0.7*nsamples)
  perm = np.random.permutation(nsamples)
  Xtrain  = X[perm[:end_train_idx]]
  Xval    = X[perm[end_train_idx:end_val_idx]]
  Xtest   = X[perm[end_val_idx:]]
  ytrain  = y[perm[:end_train_idx]]
  yval    = y[perm[end_train_idx:end_val_idx]]
  ytest   = y[perm[end_val_idx:]]

  # data normalization
  mean = Xtrain.mean(0)
  std  = Xtrain.std(0)
  std[std == 0] = 1
  Xtrain = (Xtrain-mean)/std
  Xval = (Xval-mean)/std
  Xtest = (Xtest-mean)/std

  # net params
  input_size = Xtrain.shape[1]
  hidden_size = 30
  output_size = np.unique(y).size
  
  net = Sigmoidal2LayerMLP_WithSoftmax(input_size, hidden_size, output_size, bias_init=0.0, lr=0.0001, momen_decay=0.0, l2=0.1)

  x = Xtrain[0]
  yi = y[0]
  net.forward(x)
  loss = net.backward(yi)
  Wih_grad = net.Wih_grad.copy()
  Who_grad = net.Who_grad.copy()
  hb_grad  = net.hb_grad.copy()
  ob_grad  = net.ob_grad.copy()
  e = 1e-6
  for i in xrange(net.Wih.shape[0]):
    for h in xrange(net.Wih.shape[1]):
      net.Wih[i,h] += e
      net.forward(x)
      loss1 = net.loss(yi)
      net.Wih[i,h] -= 2*e
      net.forward(x)
      loss2 = net.loss(yi)
      print 'estimated grad W%d_%d = %.4f' % (i, h, (loss1-loss2)/(2*e))
      print 'backprop grad = %.4f' % Wih_grad[i,h]
      net.Wih[i,h] += e
