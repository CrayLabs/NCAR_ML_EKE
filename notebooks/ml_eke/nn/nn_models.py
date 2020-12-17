import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import horovod.keras as hvd
hvd.init()

activation_def = 'relu'
kernel_initializer_def = 'glorot_normal'
L2 = 2e-3
kernel_regularizer_def = regularizers.l2(L2)
bias_regularizer_def = regularizers.l2(L2)
activity_regularizer_def = regularizers.l2(L2)
X_train_shape_1 = 10


def next_pow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def residual_block(inputs, channels_per_group=4, groups=32,
                   filters=256, size=(3, 3)):

    x = layers.Conv2D(
      64, 3, strides=(1, 1), padding='valid',
      data_format=None, dilation_rate=(1, 1), activation=activation_def,
      use_bias=True,
      kernel_initializer=kernel_initializer_def, bias_initializer='zeros',
      kernel_regularizer=kernel_regularizer_def, bias_regularizer=None,
      activity_regularizer=None, kernel_constraint=None, bias_constraint=None
    )(inputs)

    return x


# Next two functions taken from https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
def block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3

    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv',
                      kernel_regularizer=kernel_regularizer_def,
                      activity_regularizer=activity_regularizer_def)(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv',
                               kernel_regularizer=kernel_regularizer_def,
                               activity_regularizer=activity_regularizer_def)(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
    x = layers.Conv2D(filters, 1, use_bias=False, trainable=False,
                      kernel_initializer={'class_name': 'Constant',
                                          'config': {'value': kernel}},
                      name=name + '_2_gconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D((64 // groups) * filters, 1,
                      use_bias=False, name=name + '_3_conv',
                      kernel_regularizer=kernel_regularizer_def,
                      activity_regularizer=activity_regularizer_def)(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups,
               name=str(name) + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False,
                   name=str(name) + '_block' + str(i))
    return x


def trans_residual_block(inputs, filters=256, size=(3, 3)):

    x = layers.Conv2DTranspose(
      filters, size, strides=(1, 1), padding='valid',
      activation=None,
      kernel_initializer=kernel_initializer_def,
      kernel_regularizer=kernel_regularizer_def,
      activity_regularizer=activity_regularizer_def,
      bias_regularizer=bias_regularizer_def
    )(inputs)

    x = layers.BatchNormalization()(x)

    return layers.ReLU()(x)


def build_conv_gen_model(lr=0.001, loss='mse'):

    inputs = layers.Input(shape=(X_train_shape_1,))

    x = layers.Reshape(target_shape=(1, 1, X_train_shape_1,))(inputs)

    x = trans_residual_block(x,      filters=X_train_shape_1*2, size=3)
    x = trans_residual_block(x,      filters=2**(next_pow2(X_train_shape_1*2)),
                             size=3)
    x = trans_residual_block(x,      filters=2**(next_pow2(X_train_shape_1*2)),
                             size=3)

    x = stack3(x, 32, 3, stride1=1, name='stack1')
    x = layers.MaxPooling2D()(x)
    x = stack3(x, 64, 3, stride1=1, name='stack2')

    x = layers.Flatten()(x)

    x = layers.Dense(32, kernel_regularizer=kernel_regularizer_def,
                     activity_regularizer=activity_regularizer_def,
                     bias_regularizer=bias_regularizer_def)(x)
    x = layers.Dense(8,
                     kernel_regularizer=kernel_regularizer_def,
                     activity_regularizer=activity_regularizer_def,
                     bias_regularizer=bias_regularizer_def)(x)
    outputs = layers.Dense(1, kernel_regularizer=kernel_regularizer_def,
                           activity_regularizer=activity_regularizer_def,
                           bias_regularizer=bias_regularizer_def)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"GEN-INT_l2_{L2}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                         clipnorm=0.001)
    optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=['mae', 'mse'],
                    experimental_run_tf_function=False)
    return model

# Next models are less performant, but we keep them for future researches.

def build_ext_gen_model(lr=0.001, loss='mse'):

    inputs = layers.Input(shape=(1,1,X_train_shape_1,))

    x = inputs

    x = trans_residual_block(x,      filters=X_train_shape_1*2, size=3)
    x = trans_residual_block(x,      filters=2**(next_pow2(X_train_shape_1*2)), size=3)
    x = trans_residual_block(x,      filters=2**(next_pow2(X_train_shape_1*2)), size=3)

    x = stack3(x, 32, 3, stride1=1, groups=32, name='stack1')
    x = stack3(x, 64, 3, stride1=1, groups=32, name='stack2')
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(filters = 9, kernel_size = 1, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.ActivityRegularization(l2=L2)(x)

    shortcut = layers.ZeroPadding2D(padding=(1, 1))(inputs)

    outputs = x+shortcut

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"GEN-3x3_l2_{L2}")

    optimizer = tf.keras.optimizers.Adam(lr*hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=['mae', 'mse'],
                    experimental_run_tf_function=False)
    return model


def build_cnn(input_size=(3,3), lr=0.001):
    img_inputs = keras.layers.Input(shape=(input_size[0], input_size[1], 9))

    img_blowup = keras.layers.Conv2DTranspose(16, kernel_size=(3,3), padding='valid',
                                              kernel_initializer=kernel_initializer_def,
                                              activation=activation_def)(img_inputs)
    img_blowup = keras.layers.Conv2DTranspose(32, kernel_size=(3,3), padding='valid',
                                              kernel_initializer=kernel_initializer_def,
                                              activation=activation_def)(img_blowup)
    img_blowup = keras.layers.Conv2DTranspose(64, kernel_size=(6,6), padding='valid',
                                              kernel_initializer=kernel_initializer_def,
                                              activation=activation_def)(img_blowup)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',
                            kernel_initializer=kernel_initializer_def,
                            activation=activation_def)(img_blowup)
    x1 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',
                             kernel_initializer=kernel_initializer_def,
                             activation=activation_def)(x)
    x2 = x+x1
    x2 = keras.layers.MaxPooling2D()(x2)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
                            kernel_initializer=kernel_initializer_def,
                            activation=activation_def)(x2)
    x1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
                             kernel_initializer=kernel_initializer_def,
                             activation=activation_def)(x)
    x2 = x+x1
    x = keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same',
                            kernel_initializer=kernel_initializer_def,
                            activation=activation_def)(x2)
    x = keras.layers.MaxPooling2D()(x)
    latent = keras.layers.Flatten()(x)

    latent = keras.layers.Dense(16, activation=activation_def)(latent)
    outputs = keras.layers.Dense(1, activation=activation_def)(latent)

    model = keras.Model(inputs=img_inputs, outputs=outputs, name=f"cnn_{input_size[0]}x{input_size[1]}")

    optimizer = tf.keras.optimizers.Adam(lr)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'],
                    experimental_run_tf_function=False)

    return model


def build_tri_model(max_pow=6, lr=0.001):

    model = keras.Sequential(name='_'.join(["TriangularNN", str(max_pow), 'act', activation_def, 'l2', str(L2) ]))
    model.add(layers.Input(shape=(X_train_shape_1,)))

    min_pow = next_pow2(X_train_shape_1)
    pows = range(min_pow, max_pow+1)

    for power in pows:
        units = 2**power
        model.add(layers.Dense(units,
                               kernel_initializer=kernel_initializer_def,
                               kernel_regularizer=kernel_regularizer_def,
                               bias_regularizer=bias_regularizer_def))
        #model.add(tf.keras.layers.Dropout(0.5))

    revpows = range(max_pow-1, next_pow2(1), -1)
    for power in revpows:
        units = 2**power
        model.add(layers.Dense(units,
                               kernel_initializer=kernel_initializer_def,
                               kernel_regularizer=kernel_regularizer_def,
                               bias_regularizer=bias_regularizer_def))
        #model.add(tf.keras.layers.Dropout(0.5))

    model.add(layers.Dense(1,
                           kernel_initializer=kernel_initializer_def,
                           kernel_regularizer=kernel_regularizer_def,
                           bias_regularizer=bias_regularizer_def))

    optimizer = tf.keras.optimizers.Adam(lr*hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'],
                    experimental_run_tf_function=False)
    return model


