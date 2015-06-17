from layer import WeightLayer
import numpy as np
from scipy.ndimage.filters import convolve

def calc_output_size(input_size, filter_size, pad, stride):
    return float(input_size-filter_size+2*pad)/stride+1

def convolve_multi_channel_plus_bias(input_vol, output, kernel, bias, stride):
    kh,kw = kernel.shape[1:]
    oi,oj = 0,0
    for ki in xrange(0, input_vol.shape[1]-kh+1, stride):
        for kj in xrange(0, input_vol.shape[2]-kw+1, stride):
            output[oi,oj] = np.sum(input_vol[:,ki:ki+kh,kj:kj+kw]*kernel) + bias
            oj += 1
        oi += 1

class ConvLayer(WeightLayer):
    def __init__(self, filter_size, stride, pad, n_filters, l1, l2, init_weights_info):
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.n_filters = n_filters
        self.init_weights_info = init_weights_info
        self.l1 = l1
        self.l2 = l2

    def init_weights(self, rand_state, input_channels):
        if self.init_weights_info['type'] = 'gaussian':
            self.weights = self.init_weights_info['std']*rand_state.randn(self.n_filters, input_channels, self.filter_size, self.filter_size)
            self.biases = np.full((self.nfilters, 1), self.init_weights_info['bias'])
        else:
            raise ValueError("Unknown type for init_weights_info['type'].")
        self.weights_grad = np.zeros(self.weights.shape)
        self.biases_grad  = np.zeros(self.biases.shape)

    def setup(self, rand_state, prev_layer):
        input_shape = prev_layer.output.shape
        batch_size, input_channels, input_h, input_w = input_shape
        if self.pad > 0:
            del prev_layer.output
            del prev_layer.output_grad
            self.padded_input = np.zeros((batch_size, input_channels, input_h+2*pad, input_w+2*pad))
            self.padded_input_grad = np.zeros((batch_size, input_channels, input_h+2*pad, input_w+2*pad))
            self.input_ = self.padded_input[:,:,pad:pad+input_h,pad:pad+input_w]
            self.input_grad = self.padded_input_grad[:,:,pad:pad+input_h,pad:pad+input_w]
            prev_layer.output = self.input_
            prev_layer.output_grad = self.input_grad
        else:
            self.input_ = self.padded_input = prev_layer.output
            self.input_grad = self.padded_input_grad = prev_layer.output_grad

        self.init_weights(rand_state, input_channels)

        output_h = calc_output_size(input_h, self.filter_size, self.pad, self.stride)
        output_w = calc_output_size(input_w, self.filter_size, self.pad, self.stride)
        if not output_h.is_integer() or not output_w.is_integer():
            raise ValueError("ConvLayer filters configuration doesn't fit nicely in input dimensions.")

        self.output       = np.empty((batch_size, self.n_filters, output_h, output_w))
        self.output_grad  = np.empty(self.output.shape)

    def forward(self):
        batch_size = self.input_.shape[0]
        for b in xrange(batch_size):
            for f in xrange(self.n_filters):
                convolve_multi_channel_plus_bias(self.padded_input[b], self.output[b,f], self.weights[f], self.biases[f], self.stride)

    def backward(self):
        for b in xrange(X.shape[0]):
            for f in xrange(self.n_filters):
                input_grad_vol = self.padded_input_grad[b]
                input_vol = self.padded_input[b]
                kernel = self.weights[f]
                kh,kw = kernel.shape[1:]
                oi,oj = 0,0
                for ki in xrange(0, input_vol.shape[1]-kh+1, stride):
                    for kj in xrange(0, input_vol.shape[2]-kw+1, stride):
                        top_grad = self.output_grad[b,f,oi,oj]
                        input_grad_vol[:,ki:ki+kh,kj:kj+kw] += kernel*top_grad
                        self.weights_grad[f] += input_vol[:,ki:ki+kh,kj:kj+kw]*top_grad + 2*self.l2*self.weights + self.l1*np.sign(self.weights)
                        self.biases_grad[f] += top_grad
                        kj += 1
                    oi += 1

# keeping this imcomplete method just in case I need parts of it in the future.
def convolve_multi_channel_with_stride1_and_right_padding(input_vol, kernel, pad):
    raise NotImplementedError
    return convolve(input_vol, kernel, mode='constant').sum(axis=0)
