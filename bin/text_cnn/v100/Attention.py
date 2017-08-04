from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
import numpy as np

class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert len(input_shape) == 2
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        att_raw = K.tf.einsum('ixy,iy->ix', x[0], x[1])
        att = K.tf.nn.softmax(att_raw)
        rtn = K.tf.einsum('ixy,ix->iy', x[0], att)
        return rtn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])
