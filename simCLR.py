import io
import itertools
import random
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Bidirectional
from dataset import SignalsDataset
from tensorflow.keras.optimizers import Optimizer

from tensorflow.keras.losses import CosineSimilarity, Reduction, SparseCategoricalCrossentropy
from tensorflow.keras import regularizers
import ruamel.yaml


import re

import tensorflow.compat.v2 as tf

EETA_DEFAULT = 0.001


class LARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.
  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  """

  def __init__(self,
               learning_rate,
               momentum=0.9,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eeta=EETA_DEFAULT,
               name="LARSOptimizer"):
    """Constructs a LARSOptimizer.
    Args:
      learning_rate: A `float` for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momeuntum update in
          classic momentum, but after momentum for popular momentum.
      eeta: A `float` for scaling of learning rate when computing trust ratio.
      name: The name for the scope.
    """
    super(LARSOptimizer, self).__init__(name)

    self._set_hyper("learning_rate", learning_rate)
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.use_nesterov = use_nesterov
    self.classic_momentum = classic_momentum
    self.eeta = eeta
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for v in var_list:
      self.add_slot(v, "Momentum")

  def _resource_apply_dense(self, grad, param, apply_state=None):
    if grad is None or param is None:
      return tf.no_op()

    var_device, var_dtype = param.device, param.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    learning_rate = coefficients["lr_t"]

    param_name = param.name

    v = self.get_slot(param, "Momentum")

    if self._use_weight_decay(param_name):
      grad += self.weight_decay * param

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(grad, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
            1.0)
      scaled_lr = learning_rate * trust_ratio

      next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
      else:
        update = next_v
      next_param = param - update
    else:
      next_v = tf.multiply(self.momentum, v) + grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + grad
      else:
        update = next_v

      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        v_norm = tf.norm(update, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
            1.0)
      scaled_lr = trust_ratio * learning_rate
      next_param = param - scaled_lr * update

    return tf.group(*[
        param.assign(next_param, use_locking=False),
        v.assign(next_v, use_locking=False)
    ])

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        # TODO(srbs): Try to avoid name based filtering.
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        # TODO(srbs): Try to avoid name based filtering.
        if re.search(r, param_name) is not None:
          return False
    return True

  def get_config(self):
    config = super(LARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self.momentum,
        "classic_momentum": self.classic_momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "use_nesterov": self.use_nesterov,
    })
    return config


def get_encoder(input_shapes,
               args,
               dimension = 512,
               acc_kernel_initializer = 'he_uniform',
                regularizer = None,
                dropout = 0.0):

    inputAccShape = input_shapes


    inputAcc = keras.Input(shape = inputAccShape)

    useSpecto = args.train_args['spectograms']
    fusion = args.train_args['acc_fusion']

    X = inputAcc


    if useSpecto:
        if fusion in ['Depth','Frequency','Time']:
            _, _, channels = inputAccShape


            bnLayer = keras.layers.BatchNormalization(name = 'accBatch1')
            X = bnLayer(X)

            paddingLayer = keras.layers.ZeroPadding2D(padding=(1,1))
            conv2D = keras.layers.Conv2D(
                filters=channels * 16,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer = acc_kernel_initializer,
                name = 'accConv1',
                kernel_regularizer = regularizer
            )

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling2D((2,2),strides=2)

            X = paddingLayer(X)
            X = conv2D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            paddingLayer = keras.layers.ZeroPadding2D(padding=(1, 1))
            conv2D = keras.layers.Conv2D(
                filters=channels * 32,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name = 'accConv2',
                kernel_regularizer=regularizer
            )
            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

            X = paddingLayer(X)
            X = conv2D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)


            conv2D = keras.layers.Conv2D(
                filters=channels * 64,
                kernel_size=3,
                strides=1,
                padding='valid',
                kernel_initializer=acc_kernel_initializer,
                name = 'accConv3',
                kernel_regularizer=regularizer
            )

            X = conv2D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            flattenLayer = keras.layers.Flatten()
            dropoutLayer = keras.layers.Dropout(rate=dropout)

            X = flattenLayer(X)
            X = dropoutLayer(X)

            dnn = keras.layers.Dense(units=dimension,
                                     kernel_initializer=acc_kernel_initializer,
                                     name = 'accDense1',
                                     kernel_regularizer=regularizer)
            bnLayer = keras.layers.BatchNormalization(name = 'accBatch2')
            activationLayer = keras.layers.ReLU()
            dropoutLayer = keras.layers.Dropout(rate=dropout)

            X = dnn(X)
            X = bnLayer(X)
            X = activationLayer(X)
            X = dropoutLayer(X)

            encodings = X


    else:
        if fusion == 'Depth':
            _, channels = inputAccShape

            bnLayer = keras.layers.BatchNormalization()
            X = bnLayer(X)

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2,strides=2)
            conv1D = keras.layers.Conv1D(
                filters=channels * 16,
                kernel_size=21,
                strides=6,
                padding='valid',
                kernel_initializer = acc_kernel_initializer
            )

            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2,strides=2)
            conv1D = keras.layers.Conv1D(
                filters=channels * 32,
                kernel_size=11,
                strides=3,
                padding='valid',
                kernel_initializer = acc_kernel_initializer
            )

            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)

            activationLayer = keras.layers.ReLU()
            poolingLayer = keras.layers.MaxPooling1D(2,strides=2)
            conv1D = keras.layers.Conv1D(
                filters=channels * 64,
                kernel_size=5,
                strides=1,
                padding='valid',
                kernel_initializer = acc_kernel_initializer
            )

            X = conv1D(X)
            X = activationLayer(X)
            X = poolingLayer(X)



            flattenLayer = keras.layers.Flatten()
            X = flattenLayer(X)


            dnn = keras.layers.Dense(units=dimension,kernel_initializer = acc_kernel_initializer)
            bnLayer = keras.layers.BatchNormalization()
            activationLayer = keras.layers.ReLU()


            X = dnn(X)
            X = bnLayer(X)
            X = activationLayer(X)

            encodings = X

    return keras.models.Model(inputs = inputAcc,
                              outputs = encodings,
                              name = 'Encoder')


def get_projection_head(dimension = 512,
                        kernel_initializer = 'he_uniform',
                        regularizer = None,
                        dropout = 0.2):
    encodings = keras.Input(shape=(dimension,))

    dnn = keras.layers.Dense(units = dimension,
                             kernel_initializer = kernel_initializer,
                             kernel_regularizer = regularizer)
    bnLayer = keras.layers.BatchNormalization()
    activationLayer = keras.layers.ReLU()
    dropoutLayer = keras.layers.Dropout(rate=dropout)


    X = dnn(encodings)
    X = bnLayer(X)
    X = activationLayer(X)
    X = dropoutLayer(X)


    dnn = keras.layers.Dense(units = dimension,
                             kernel_initializer = kernel_initializer,
                             kernel_regularizer = regularizer)
    projections = dnn(X)

    return keras.models.Model(
        inputs = encodings,
        outputs = projections,
        name = 'ProjectionHead'
    )

def get_linear_probe(L, dimension = 128,
                     kernel_initializer = keras.initializers.he_uniform(),
                     class_kernel_initializer = keras.initializers.glorot_uniform()):

    projections = keras.Input(shape=(dimension,))

    dnn = keras.layers.Dense(units=L,
                             kernel_initializer=kernel_initializer,
                             name='accDense2')

    bnLayer = keras.layers.BatchNormalization(name='accBatch3')
    activationLayer = keras.layers.ReLU()

    X = dnn(projections)
    X = bnLayer(X)
    X = activationLayer(X)

    dnn = keras.layers.Dense(units=8,
                             activation = 'softmax',
                             kernel_initializer = class_kernel_initializer)

    outputs = dnn(X)

    return keras.models.Model(
        inputs = projections,
        outputs = outputs,
        name = 'linearProbe'
    )


class simCLR(keras.Model):
    def __init__(self,
                 input_shapes,
                 args,
                 L = None,
                 temperature = 0.1,
                 transfer = True):

        super(simCLR, self).__init__()
        self.args = args
        self.input_shapes = input_shapes
        self.dimension = args.train_args['dimension']


        self.transfer = transfer
        self.temperature = temperature
        self.encoder = get_encoder(input_shapes=input_shapes[0],
                                   args=args,
                                   dimension=self.dimension)

        self.proj_head = get_projection_head(dimension = self.dimension)

        self.contrastive_optimizer =  keras.optimizers.Adam(
            learning_rate=0.1
        )

        if not self.transfer:
            self.linear_probe = get_linear_probe(
                L = L,
                dimension = self.dimension
            )

        self.batch_size = args.train_args['trainBatchSize']
        self.negative_mask = np.ones(
            (self.batch_size, self.batch_size * 2), dtype=bool
        )

        for i in range(self.batch_size):
            self.negative_mask[i, i] = 0
            self.negative_mask[i, i + self.batch_size] = 0

        self.negative_mask = tf.constant(self.negative_mask)



        self.loss_function = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.SUM)

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary()
        self.proj_head.summary()

        if not self.transfer:
            self.linear_probe.summary()

    def compile(self,
                contrastive_optimizer = None,
                probe_optimizer = keras.optimizers.Adam(),
                loss_function = keras.losses.CategoricalCrossentropy(),
                **kwargs):

        super(simCLR, self).compile(**kwargs)


        if not self.transfer:
            self.probe_optimizer = probe_optimizer

            self.probe_loss = loss_function

            self.probe_loss_tracker = keras.metrics.Mean(
                name='p_loss'
            )

            self.probe_accuracy = keras.metrics.SparseCategoricalCrossentropy(
                name='p_acc'
            )

        if contrastive_optimizer:
            self.contrastive_optimizer = contrastive_optimizer

        self.contrastive_loss_tracker = keras.metrics.Mean(
            name="c_loss"
        )

        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )

    @property
    def metrics(self):
        if not self.transfer:

            return [
                self.contrastive_loss_tracker,
                self.contrastive_accuracy,
                self.probe_loss_tracker,
                self.probe_accuracy
            ]

        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy
        ]

    def contrastive_loss(self,
                         projections_1,
                         projections_2):

        projections_1 = tf.math.l2_normalize(
            projections_1, axis=1
        )

        projections_2 = tf.math.l2_normalize(
            projections_2, axis=1
        )

        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        contrastive_labels = tf.range(self.batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)

        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )

        return (loss_1_2 + loss_2_1) / 2

    def forward_pass(self, data1, data2):
        encodings_1 = self.encoder(data1)
        encodings_2 = self.encoder(data2)

        projections_1 = self.proj_head(encodings_1)
        projections_2 = self.proj_head(encodings_2)

        return projections_1, projections_2

    @tf.function
    def train_step(self, data):
        if not self.transfer:
            data, y_true = data
            labeled_data = tf.boolean_mask(data, y_true != unlabeled)
            labels = tf.boolean_mask(y_true, y_true != unlabeled)

            tmp_pnl = np.reshape(np.random.sample(self.pos_name_list,
                                       self.augmentations,
                                       replace=False),(-1,2))

            for positions in tmp_pnl:
                data_pos_1 = data[positions[0]]

                data_pos_2 = data[positions[1]]

                with tf.GradientTape() as tape:
                    encodings_1 = self.encoder(data_pos_1, training = True)
                    encodings_2 = self.encoder(data_pos_2, training = True)

                    projections_1 = self.proj_head(encodings_1, training = True)
                    projections_2 = self.proj_head(encodings_2, training = True)

                    contrastive_loss = self.contrastive_loss(
                        projections_1, projections_2
                    )

                gradients = tape.gradient(
                    contrastive_loss,
                    self.encoder.trainable_weights+\
                    self.proj_head.trainable_weights
                )

                self.contrastive_optimizer.apply_gradients(
                    zip(
                        gradients,
                        self.encoder.trainable_weights+\
                        self.proj_head.trainable_weights
                    )
                )



        if self.transfer:


            data_pos_1 = data[0]
            data_pos_2 = data[1]


            with tf.GradientTape() as tape:
                projections_1, projections_2 = self.forward_pass(data_pos_1, data_pos_2)

                contrastive_loss = self.contrastive_loss(
                    projections_1, projections_2
                )

            gradients = tape.gradient(
                contrastive_loss,
                self.encoder.trainable_weights + \
                self.proj_head.trainable_weights
            )

            self.contrastive_optimizer.apply_gradients(
                zip(
                    gradients,
                    self.encoder.trainable_weights + \
                    self.proj_head.trainable_weights
                )
            )

            self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def call(self, data):
        if self.transfer:


            data_pos_1 = data[0]
            data_pos_2 = data[1]

            projections_1, projections_2 = self.forward_pass(data_pos_1, data_pos_2)

            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2
            )

            self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        if self.transfer:


            data_pos_1 = data[0]
            data_pos_2 = data[1]

            projections_1, projections_2 = self.forward_pass(data_pos_1, data_pos_2)

            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2
            )

            self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}


def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:
        print(param)
        if param == parameter:
            original_value = data[args][param]
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)

    return original_value

def fit(evaluation = True,
        summary = True,
        verbose = 1):

    accBagSize = config_edit('train_args', 'accBagSize', 1)
    locBagSize = config_edit('train_args', 'locBagSize', 1)
    bagStride = config_edit('train_args', 'bagStride', 1)
    trainBatchSize = config_edit('train_args', 'trainBatchSize', 128)
    valBatchSize = config_edit('train_args', 'valBatchSize', 128)
    testBatchSize = config_edit('train_args', 'testBatchSize', 128)

    #print(accBagSize,locBagSize,bagStride)


    SD = SignalsDataset()

    train, val, test = SD(baseline=True,
                          simCLR=True)

    user = SD.shl_args.train_args['test_user']

    config_edit('train_args', 'accBagSize', accBagSize)
    config_edit('train_args', 'locBagSize', locBagSize)
    config_edit('train_args', 'bagStride', bagStride)
    config_edit('train_args', 'trainBatchSize', trainBatchSize)
    config_edit('train_args', 'valBatchSize', valBatchSize)
    config_edit('train_args', 'testBatchSize', testBatchSize)


    logdir = os.path.join('logs_user' + str(user),'simCLR_tensorboard')

    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    val_steps = SD.valSize // SD.valBatchSize

    test_steps = SD.testSize // SD.testBatchSize

    train_steps = SD.trainSize // SD.trainBatchSize

    Model = simCLR(SD.inputShape,
                   SD.shl_args)


    Model.compile(
        contrastive_optimizer = keras.optimizers.Adam(
            learning_rate=0.1
        )
    )

    save_dir = os.path.join('training','saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    model_type = 'simCLR'
    model_name = 'shl_%s_model.h5' %model_type
    filepath = os.path.join(save_dir, model_name)

    save_model = keras.callbacks.ModelCheckpoint(
                filepath = filepath,
                monitor = 'val_c_loss',
                verbose = 1,
                save_best_only = True,
                mode = 'min',
                save_weights_only=True
        )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor = 'val_c_loss',
        min_delta = 0,
        patience = 20,
        mode = 'min',
        verbose = 1
    )

    reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_c_loss',
        factor = 0.4,
        patience = 10,
        verbose = 1,
        mode = 'min'
    )



    if summary:
        print(Model.summary())



    Model.fit(
        train,
        epochs=SD.shl_args.train_args['simCLRepochs'],
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = [tensorboard_callback,
                     save_model,
                     early_stopping,
                     reduce_lr_plateau],
        use_multiprocessing = True,
        verbose=verbose
    )

    Model.built = True
    Model.load_weights(filepath)

    if evaluation:

        Model.evaluate(test,steps=test_steps)



    return filepath

# def contrastive_loss(
#                      projections_1,
#                      projections_2):
#
#     projections_1 = tf.math.l2_normalize(
#         projections_1, axis=1
#     )
#
#     projections_2 = tf.math.l2_normalize(
#         projections_2, axis=1
#     )
#
#     batch_size = tf.shape(projections_1)[0]
#
#     similarities = tf.matmul(tf.expand_dims(projections_1, 1), tf.expand_dims(projections_2, 2))
#     similarities = tf.reshape(similarities, (batch_size, 1))
#     similarities /= 1.0
#
#     negatives = tf.concat([projections_2, projections_1], axis=0)
#     negative_mask = np.ones(
#             (batch_size, batch_size * 2), dtype=bool
#         )
#
#     for i in range(batch_size):
#         negative_mask[i, i] = 0
#         negative_mask[i, i + batch_size] = 0
#
#     negative_mask = tf.constant(negative_mask)
#
#     loss = 0
#     loss_function = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.SUM)
#     for positives in [projections_1, projections_2]:
#         l_neg = tf.tensordot(tf.expand_dims(positives, 1), tf.expand_dims(tf.transpose(negatives), 0), axes=2)
#         labels = tf.zeros(batch_size, dtype=tf.int32)
#         l_neg = tf.boolean_mask(l_neg, negative_mask)
#         l_neg = tf.reshape(l_neg, (batch_size, -1))
#         l_neg /= 1.0
#
#         logits = tf.concat([similarities, l_neg], axis=1)
#
#         loss += loss_function(y_pred=logits, y_true=labels)
#
#     loss = loss / (2 * tf.cast(batch_size, dtype=tf.float32))
#
#     print(loss)
#
#
#     return loss
