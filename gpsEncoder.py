import shutil
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import *
from keras import Input
from keras import initializers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
import os
from dataset import Dataset
from myMetrics import gpsValMetrics, gpsTestMetrics, gpsValTables, gpsTestTables


def getGpsEncoder(input_shapes, args, L):
    mask = args.train_args['mask']
    kernelInitializer = initializers.he_uniform()
    gpsSeriesShape = input_shapes[0]
    gpsFeaturesShape = input_shapes[1]
    gpsSeries = Input(shape=gpsSeriesShape)
    gpsFeatures = Input(shape=gpsFeaturesShape)

    X = gpsSeries

    masking_layer = Masking(mask_value=mask, name='maskLayer1')
    X = masking_layer(X)

    bnLayer = BatchNormalization(name='locBatch', trainable=False)
    X = bnLayer(X)

    lstmLayer = LSTM(units=128,
                     name='locLSTM')
    X = lstmLayer(X)

    X = tf.concat([X, gpsFeatures], axis=1)

    denseLayer = Dense(
        units=128,
        kernel_initializer=kernelInitializer,
        name='locDense1'
    )

    bnLayer = BatchNormalization(name='locBatch1', trainable=False)
    activationLayer = ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = Dense(
        units=64,
        kernel_initializer=kernelInitializer,
        name='locDense2'
    )

    bnLayer = BatchNormalization(name='locBatch2', trainable=False)
    activationLayer = ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = Dense(
        units=L,
        kernel_initializer=kernelInitializer,
        name='locDense3'
    )

    bnLayer = BatchNormalization(name='locBatch3', trainable=False)
    activationLayer = ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    mask_w = K.switch(
        tf.reduce_all(tf.equal(gpsFeatures, mask), axis=1, keepdims=True),
        lambda: tf.zeros_like(X), lambda: tf.ones_like(X)
    )

    gpsEncodings = tf.multiply(X, mask_w)

    return Model(inputs=[gpsSeries, gpsFeatures],
                 outputs=gpsEncodings,
                 name='gpsEncoder')


def getClassifier(L, n_units=8):
    pooling = keras.Input(shape=L)
    kernel_initializer = initializers.glorot_uniform()

    X = pooling

    finalLayer = Dense(units=n_units,
                       activation='softmax',
                       kernel_initializer=kernel_initializer)

    yPred = finalLayer(X)

    return Model(inputs=pooling,
                 outputs=yPred,
                 name='Classifier')


def build(input_shapes, args, L=256):
    motorized = args.train_args['motorized']
    n_classes = 5 if motorized else 8

    shape = input_shapes

    gpsNetwork = getGpsEncoder(shape, args, L)

    classifier = getClassifier(L, n_units=n_classes)

    gpsSeries = Input(shape[0])
    gpsFeatures = Input(shape[1])
    gpsEncodings = gpsNetwork([gpsSeries, gpsFeatures])

    yPred = classifier(gpsEncodings)

    return Model([gpsSeries, gpsFeatures], yPred)


def fit(L=256, summary=True, verbose=0, mVerbose=False):

    data = Dataset()

    train, val, test = data(gpsTransfer=True)

    logdir = os.path.join('logs_user' + str(data.testUser), 'gpsEncoderTb')

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')

    save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_type = 'gpsEncoder'
    model_name = 'TMD_%s_model.h5' % model_type
    filepath = os.path.join(save_dir, model_name)

    val_steps = data.valSize // data.valBatchSize
    train_steps = data.trainSize // data.trainBatchSize
    test_steps = data.testSize // data.testBatchSize

    Model = build(input_shapes=data.inputShape, args=data.shl_args, L=L)

    optimizer = Adam(learning_rate=data.lr)

    loss_function = CategoricalCrossentropy()

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[categorical_accuracy]
    )

    val_metrics = gpsValMetrics(val, data.valBatchSize, val_steps, verbose)

    val_tables = gpsValTables(val, data.valBatchSize, val_steps, file_writer_val)

    save_model = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        mode='min',
        save_weights_only=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        mode='min',
        verbose=verbose
    )

    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=10,
        verbose=verbose,
        mode='min'
    )

    if summary and verbose:
        print(Model.summary())

    callbacks = [tensorboard_callback,
                 save_model,
                 early_stopping,
                 reduce_lr_plateau,
                 val_metrics,
                 val_tables]

    Model.fit(
        train,
        epochs=data.gpsEpochs,
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        verbose=verbose
    )

    test_metrics = gpsTestMetrics(test, data.testBatchSize, test_steps)

    test_tables = gpsTestTables(test,
                            data.testBatchSize,
                            test_steps,
                            file_writer_test)

    Model.load_weights(filepath)

    Model.evaluate(test, steps=test_steps, callbacks=[test_metrics, test_tables])

    del data.acceleration
    del data.location
    del data.labels
    del data

    return filepath
