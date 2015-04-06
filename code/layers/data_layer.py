import numpy as np
from skimage.io import imread
import time
import ipdb

def read_imgs_path_from_txt_file(file_path):
  with open(file_path, 'r') as f:
    paths = f.read().splitlines()
  return np.array(paths)

class DataLayer(object):
  def __init__(self, params):
    self.batch_size = params['batch_size']

  # DataLayer backward method doesn't need to do anything.
  def backward(self):
    pass

  # makes this method mandatory to implement
  def get_n_samples(self):
    pass

# Currently this layer stores all the imgs paths in memory.
# If the dataset has too many images, the paths will occupy
# a lot of memory. For example, 1 million paths ~= 100MB
# But of course there are redundant subpaths in this files.
# We can explore that fact in the future.
class ImageDataLayer(DataLayer):
  def __init__(self, params):
    super(ImageDataLayer, self).__init__(params)
    self.imgs_source = params['imgs_source']
    self.labels_source = params['labels_source']
    self.gray = params.get('gray', False)
    self.n_samples = None

  def get_n_samples(self):
    if self.n_samples == None:
      raise ValueError # actually want illegal state error
    return self.n_samples

  def read_img(self, path):
    img = imread(path, as_grey=self.gray)
    if len(img.shape) == 3:
      img = img.transpose(2,0,1)
    return img

  def permute_sample_inds(self):
    self.sample_inds = self.rand_state.permutation(self.n_samples)

  def setup(self, rand_state):
    self.rand_state = rand_state
    self.imgs_paths = read_imgs_path_from_txt_file(self.imgs_source)[:1000]
    self.labels = np.loadtxt(self.labels_source, int)[:1000]
    self.n_samples = len(self.imgs_paths)
    img0 = self.read_img(self.imgs_paths[0])
    if len(img0.shape) > 2:
      height, width = img0.shape[1:]
      n_channels = img0.shape[0]
    else:
      height, width = img0.shape
      n_channels = 1
    self.output = np.empty((self.batch_size, n_channels, height, width))
    self.output_labels = np.empty(self.batch_size, int)
    self.batch_idx = 0
    self.permute_sample_inds()

  # TODO: Make a test to really verify that this function is doing 
  # what it is supposed to.
  def get_current_imgs_inds(self):
    last_idx = self.batch_idx + self.batch_size
    current_inds = np.empty(self.batch_size, int)
    if last_idx > self.n_samples:
      first_amount = self.n_samples-self.batch_idx
      current_inds[:first_amount] = self.sample_inds[self.batch_idx:]
      self.batch_idx = last_idx - self.n_samples
      current_inds[first_amount:] = self.sample_inds[:self.batch_idx]
      self.permute_sample_inds()
      self.batch_idx = 0
    else:
      current_inds[:] = self.sample_inds[self.batch_idx:last_idx]
      self.batch_idx = last_idx
    return current_inds

  def forward(self):
    #start = time.clock()
    current_imgs_inds = self.get_current_imgs_inds()
    current_paths = self.imgs_paths[current_imgs_inds]
    self.output_labels[:] = self.labels[current_imgs_inds]
    for i, path in enumerate(current_paths):
      img = self.read_img(path)
      if len(img.shape) == 2:
        self.output[i,0,:] = img
      else:
        self.output[i,:] = img
    #end = time.clock()
    #print end-start

def test_image_data_layer():
  params = {}
  params['gray'] = True
  params['batch_size'] = 220
  params['imgs_source'] = '/media/cesar/Acer/Users/cesarsalgado/datasets/tiny-imagenet-200/train/paths.txt'
  params['labels_source'] = '/media/cesar/Acer/Users/cesarsalgado/datasets/tiny-imagenet-200/train/labels.txt'
  img_data_layer = ImageDataLayer(params)
  rand_state = np.random.RandomState(1)
  img_data_layer.setup(rand_state)
  img_data_layer.forward()
  return img_data_layer
