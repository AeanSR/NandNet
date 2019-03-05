import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

rev_gap = 1.

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def BinarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='BinarizedSpatialConvolution'):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None,[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            bin_x = binarize(x)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d

def BinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='SpatialConvolution'):
    def conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name,None,[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d

def Affine(nOutputPlane, bias=True, name=None, reuse=False):
    def affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer

def BinarizedAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def b_affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer

def BinarizedWeightOnlyAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def bwo_affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return bwo_affineLayer

def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)


def wrapNN(f,*args,**kwargs):
    def layer(x, scope='', is_training=True):
        return f(x,*args,**kwargs)
    return layer

def Dropout(p, name='Dropout'):
    def dropout_layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x,p)
            else:
                return x
    return dropout_layer

def ReLU(name='ReLU'):
    def layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            return tf.nn.relu(x)
    return layer

def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            return tf.clip_by_value(x,-1,1)
    return layer


def View(shape, name='View'):
    with tf.variable_scope(name,None,[x], reuse=reuse):
        return wrapNN(tf.reshape,shape=shape)

def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def max_pool(x,is_training=True):
        with tf.variable_scope(name,None,[x]):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool

def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def avg_pool(x,is_training=True):
        with tf.variable_scope(name, None, [x]):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool

def BatchNormalization(*kargs, **kwargs):
    return wrapNN(tf.contrib.layers.batch_norm, *kargs, **kwargs)


def Sequential(moduleList):
    def model(x, is_training=True):
    # Create model
        output = x
        #with tf.variable_scope(name,None,[x]):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model

def Concat(moduleList, dim=3):
    def model(x, is_training=True):
    # Create model
        outputs = []
        for i,m in enumerate(moduleList):
            name = 'layer_'+str(i)
            with tf.variable_scope(name, 'Layer', [x], reuse=reuse):
                outputs[i] = m(x, is_training=is_training)
            output = tf.concat(dim, outputs)
        return output
    return model

def Residual(moduleList, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True):
    # Create model
        with tf.variable_scope(name,None,[x]):
            output = tf.add(m(x, is_training=is_training), x)
            return output
    return model

def inputbin(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    For inputs: clip to 0/1, then concat with self-reverses.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        return x
        #x=tf.clip_by_value(x,0,1)
#        with g.gradient_override_map({"Sign": "Identity", "ClipByValue": "Identity"}):
#            return tf.sign(x)

def weightbin(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    For weights: clip to 0/1.
    """
    g = tf.get_default_graph()
#    x = tf.clip_by_value(x, -1, 1)
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Round": "Identity", "ClipByValue": "Identity"}):
            #x = tf.round(tf.clip_by_value(x,-1,1))
            return x

def crossprod(a, b):
    tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])  
    tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]) 
    return tf.multiply(tile_a, tile_b) 

def NandLayer(nOutputPlane, name=None, reuse=False):
    def b_nandLayer(x, is_training=True):
        with tf.variable_scope(name,'Nand',[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''

            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.random_uniform_initializer(-1, 1))
            bin_w = weightbin(w)
            bin_x = inputbin(reshaped)
            w_unpack = tf.unstack(bin_w, axis=0)
            x_unpack = tf.unstack(bin_x, axis=1)
            nmos = 1.
            pmos = 0.
            for w_i in w_unpack:
                for x_i in x_unpack:
                    ident = crossprod(x_i, w_i)
                    nmos = tf.multiply(nmos, tf.nn.relu(tf.negative(ident)))
                    pmos = tf.add(pmos, tf.nn.relu(ident))
            output = tf.subtract(pmos, nmos)
            #output = binarize(tf.negative(tf.matmul(bin_x, bin_w)))
        return output
    return b_nandLayer

def DenseNandLayer(nOutputPlane, name=None, reuse=False):
    def b_denseNandLayer(x, is_training=True):
        with tf.variable_scope(name,'DenseNand',[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.random_uniform_initializer(-1, 1))
            bin_w = weightbin(w)
            bin_x = inputbin(reshaped)
            w_unpack = tf.unstack(bin_w, axis=0)
            x_unpack = tf.unstack(bin_x, axis=1)
            nmos = 1.
            pmos = 0.
            for w_i in w_unpack:
                for x_i in x_unpack:
                    ident = crossprod(x_i, w_i)
                    nmos = tf.multiply(nmos, tf.nn.relu(tf.negative(ident)))
                    pmos = tf.add(pmos, tf.nn.relu(ident))
            output = tf.subtract(pmos, nmos)
            #output = binarize(tf.negative(tf.matmul(bin_x, bin_w)))
        return tf.concat([x, output], 1)
    return b_denseNandLayer

