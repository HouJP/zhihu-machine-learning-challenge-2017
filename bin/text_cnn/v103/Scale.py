from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
import numpy as np

class Scale(Layer):

    def __init__(self, 
                 kernel_initializer=initializers.Constant(1.0),
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(1,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(1,),
                                      initializer=self.bias_initializer,
                                      name='bias',
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
        super(Scale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.tf.scalar_mul(self.kernel[0], x) + self.bias[0]

    def compute_output_shape(self, input_shape):
        return input_shape
