""" 
    Python implementation of AlexNet in tensorflow.
"""

import numpy as np
import tensorflow as tf

class AlexNet(object):

    def __init__ (self, x, keep_prob, num_classes, skip_layer,
                    weights_path = 'DEFAULT'):
        """
            Inputs: 
            - x:            tf.placeholder, for input images
            - keep_prob:    tf.placeholder, for dropout rate
            - num_classes:  int, no. of classes in the dataset
            - skip_layer:   list of strings, names of layers to be reinitialized
            - weights_path: path string for the stored weights
        """

        self.x = x
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.skip_layer = skip_layer
        self.is_training = is_training

        if weights_path == 'DEFAULT':
            self.weights_path = 'bvlc_alexnet.npy'
        else:
            self.weights_path = weights_path

        # Call create function to create model graph
        self.create()

    # Create the model graph
    def create(self):
        # 1st layer: conv(with ReLU) -> LRN -> Pool
        conv1 = conv(self.x, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
        norm1 = local_response_normalization(conv1, 2, 1e-05, 0.75, name = 'norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')

        # 2nd layer: conv(with ReLU) -> LRN -> Pool
        conv2 = conv(pool1, 5, 5, 256, 1, 1, name = 'conv2')
        norm2 = local_response_normalization(conv2, 2, 1e-05, 0.75, name = 'norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')

        # 3rd layer: conv(with ReLU)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name = 'conv3')

        # 4th layer: conv(with ReLU)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, name = 'conv4')

        # 5th layer: conv(with ReLU) -> Pool
        conv5 = conv(conv4, 3, 3, 256, 1, 1, name = 'conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

        # 6th layer: fully connected(with ReLU) -> Dropout
        flattened_layer = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fullu_connected(flattened_layer, 6*6*256, 4096, name = 'fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th layer: fully connected(with ReLU) -> Dropout
        fc7 = fully_connected(dropout6, 4096, 4096, name = 'fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        # 8th layer: fully connected and return unscaled activations
        self.fc8 = fully_connected(dropout7, 4096, self.num_classes, relu = False, name = 'fc8')


    # Load pre-trained weights for initialization
    def load_initial_weights(self, session):
        # Load weights into memory
        weights_dict = np.load(self.weights_path, encoding = 'bytes').item()

        # Loop over all layers stored in the dict
        for op_name in weights_dict:
            # Check if layer needs to be reinitialized
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse = True):

                    # loop over list of weights/biases and assign them to their 
                    # corresponding layers
                    for data in weights_dict[op_name]:
                        # biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable = False)
                            session.run(var.assign(data))

                        else:
                            # weights
                            var = tf.get_variable('weights', trainable = False)
                            session.run(var.assign(data))


# Helper function for convolution layer
def conv_layer(x, filter_height, filter_width, num_filters, stride_y, 
                stride_x, name, padding = 'SAME'):
    """
        Inputs:
        - x:                input image
        - filter_height:    convolution filter height
        - filter_width:     convolution filter width
        - num_filters:      no. of convolution filters

return tf.nn.dropout(x, keep_prob)

- stride_y:         no. of blocks to stride along y axis
        - stride_x:         no. of blocks to stride along x axis
        - name:             name scope
        - padding:          Whether to zero-pad or not. Defaults to same size
    """

    # Get no. of input channels
    input_channels = int(x.get_shape()[-1])

    # Lambda function for convolutions
    # strides = [batch_num. height, width, channel]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1],
                                            padding = padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for weights and biases of the conv layer
        weights = tf.get_variable('weights', shape = [filter_height, filter_width, 
                                                    input_channels, num_filters])
        biases = tf.get_variable('biases', shape = [num_filters])

        # Convolve images with the weights
        conv = convolve(x, weights)

        # Add bias
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # ReLU
        relu = tf.nn.relu(bias, name = scope.name)
        
        return relu

# Helper function for fully connected layers
def fully_connected(x, num_inputs, num_outputs, name, relu = True):
    """
        Inputs:
        - x:                input image
        - num_inputs:       no. of inputs from prev layer
        - num_outputs:      no. of outputs to the next layer
        - name:             name scope
        - relu:             whether using relu
    """

    with tf.variable_scope(name) as scope:
        # Create tf variables for weights and biases
        weights = tf.get_varibale('weights', shape = [num_inputs, num_outputs], 
                                                trainable = True)
        biases = tf.get_variable('biases', shape = [num_outputs], trainable = True)

        # Calculate activation
        act = tf.nn.xw_plus_b(x, weights, biases, name = scope.name)

        if relu == True:
            # Apply ReLU non-linearity
            relu = tf.nn.relu(act)
            
            return relu
        else:
            return act

# Helper function for Max-Pooling layer
def max_pool(x, filter_height, filter_width, stride_y, stride_x, 
                    name, padding = 'SAME'):
    """
        Inputs:
        - x:                input image
        - filter_height:    height of pooling filter
        - filter_width:     width of pooling filter
        - stride_y:         no. of blocks to stride along y axis
        - stride_x:         no. of blocks to stride along x axis
        - name:             scope name
        - padding:          whether to pad with 0s
    """
    return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
                                    strides = [1, striude_y, stride_x, 1], 
                                            name = name, padding = padding)

# Helper function for local response normalization
def local_response_normalization(x, radius, alpha, beta, name, bias = 1.0):
    """
        Inputs:
        - x:                input image
        - radius:           size of normalization window
        - alpha:            hyperparameter
        - beta:             hyperparameter
        - name:             scope name
        - bias:             neuron bias
    """
    return tf.nn.local_response_normalization(x, depth_radius = radius,  alpha = alpha,
                                                beta = beta, bias = bias, name = name)

# Dropout
def dropout(x, keep_prob):
    """
        Inputs:
        - x:                input image
        - keep_prob:        fraction of neurons to keep
    """
    return tf.nn.dropout(x, keep_prob)


