import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def softmax(x):
  xmax = x.max()
  exp_x = np.exp(x-xmax)
  return exp_x/exp_x.sum()

def get_random_batches_inds(nsamples, batch_size):
  nbatches = nsamples/batch_size+1
  batches = nbatches*[None]
  perm = np.random.permutation(nsamples)
  for i in xrange(nbatches-1):
    batches[i] = perm[i*batch_size:(i+1)*batch_size]
  i = nbatches-1
  batches[i] = perm[i*batch_size:nsamples]
  return batches
