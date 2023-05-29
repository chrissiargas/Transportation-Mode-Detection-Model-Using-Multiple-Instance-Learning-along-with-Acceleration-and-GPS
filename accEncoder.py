import shutil
import tensorflow as tf
import keras
import os
from keras.layers import *
from keras import Input
from keras import initializers
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from dataset import Dataset
from myMetrics import accValMetrics, accTestMetrics, accValTables, accTestTables


def getAccEncoder(input_shapes, args, L):
    kernelInitializer = initializers.he_uniform()

    useMIL = args.train_args['separate_MIL']
    dimension = 128
    shape = args.train_args['acc_shape']

    if useMIL:
        accShape = list(input_shapes)[1:]

    else:
        accShape = list(input_shapes)

    accSpectrograms = Input(shape=accShape)
    X = accSpectrograms

    if shape == '2D':
        _, _, channels = accShape
    if shape == '1D':
        _, channels = accShape

    bnLayer = BatchNormalization(name='accBatch1')
    X = bnLayer(X)

    if shape == '2D':
        paddingLayer = ZeroPadding2D(padding=(1, 1))  # same padding
        conv2D = Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv1'
        )
        bnLayer = BatchNormalization(name='accBatch2')
        activationLayer = ReLU()
        poolingLayer = MaxPooling2D((2, 2), strides=2)

        X = paddingLayer(X)
        X = conv2D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        paddingLayer = ZeroPadding2D(padding=(1, 1))  # same padding
        conv2D = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv2'
        )
        bnLayer = BatchNormalization(name='accBatch3')
        activationLayer = ReLU()
        poolingLayer = MaxPooling2D((2, 2), strides=2)

        X = paddingLayer(X)
        X = conv2D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        conv2D = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv3'
        )
        bnLayer = BatchNormalization(name='accBatch4')
        activationLayer = ReLU()
        poolingLayer = MaxPooling2D((2, 2), strides=2)

        X = conv2D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

    if shape == '1D':
        paddingLayer = ZeroPadding1D(padding=1)  # same padding
        conv1D = Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv1'
        )
        bnLayer = BatchNormalization(name='accBatch2')
        activationLayer = ReLU()
        poolingLayer = MaxPooling1D(2, strides=2)

        X = paddingLayer(X)
        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        paddingLayer = ZeroPadding1D(padding=1)  # same padding
        conv1D = Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv2'
        )
        bnLayer = BatchNormalization(name='accBatch3')
        activationLayer = ReLU()
        poolingLayer = MaxPooling1D(2, strides=2)

        X = paddingLayer(X)
        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        conv1D = Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv3'
        )
        bnLayer = BatchNormalization(name='accBatch4')
        activationLayer = ReLU()
        poolingLayer = MaxPooling1D(2, strides=2)

        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

    flattenLayer = Flatten()
    dropoutLayer = Dropout(rate=0.3)

    X = flattenLayer(X)
    X = dropoutLayer(X)

    dnn = Dense(units=dimension,
                kernel_initializer=kernelInitializer,
                name='accDense1')
    bnLayer = BatchNormalization(name='accBatch5')
    activationLayer = ReLU()
    dropoutLayer = Dropout(rate=0.25)

    X = dnn(X)
    X = bnLayer(X)
    X = activationLayer(X)
    X = dropoutLayer(X)

    dnn = Dense(units=L,
                kernel_initializer=kernelInitializer,
                name='accDense2')
    bnLayer = BatchNormalization(name='accBatch6')
    activationLayer = ReLU()
    dropoutLayer = Dropout(rate=0.25)

    X = dnn(X)
    X = bnLayer(X)
    X = activationLayer(X)
    X = dropoutLayer(X)

    accEncodings = X

    return Model(inputs=accSpectrograms,
                 outputs=accEncodings,
                 name='AccelerationEncoder')


def getMIL(args, L, D):
    kernelInitializer = initializers.glorot_uniform()
    kernelRegularizer = l2(0.01)

    accEncodings = Input(shape=L)

    D_layer = Dense(units=D,
                    activation='tanh',
                    kernel_initializer=kernelInitializer,
                    kernel_regularizer=kernelRegularizer,
                    name='D_layer')
    G_layer = Dense(units=D,
                    activation='sigmoid',
                    kernel_initializer=kernelInitializer,
                    kernel_regularizer=kernelRegularizer,
                    name='G_layer')
    K_layer = Dense(units=1,
                    kernel_initializer=kernelInitializer,
                    name='K_layer')

    attentionWs = D_layer(accEncodings)
    attentionWs = attentionWs * G_layer(accEncodings)
    attentionWs = K_layer(attentionWs)

    return Model(inputs=accEncodings,
                 outputs=attentionWs,
                 name='AccelerationMIL')


def getClassifier(L, n_units=8):
    pooling = keras.Input(shape=L)
    kernelInitializer = initializers.glorot_uniform()

    X = pooling

    finalLayer = Dense(units=n_units, activation='softmax', kernel_initializer=kernelInitializer)

    yPred = finalLayer(X)

    return Model(inputs=pooling, outputs=yPred, name='Classifier')


def build(input_shapes, args, L=32, D=12):
    batchSize = args.train_args['trainBatchSize']
    useMIL = args.train_args['separate_MIL']
    motorized = args.train_args['motorized']
    n_classes = 5 if motorized else 8
    accShape = input_shapes[0]
    posShape = input_shapes[1]
    accNetwork = getAccEncoder(accShape, args, L)
    classifier = getClassifier(L, n_units=n_classes)
    accBag = Input(accShape)
    accPos = Input(posShape, name='positional')
    accMIL = None
    accBagSize = None

    if useMIL:
        accMIL = getMIL(args, L, D)

    if useMIL:

        accBagSize = list(accShape)[0]
        accSize = list(accShape)[1:]
        concAccBag = tf.reshape(accBag, (batchSize * accBagSize, *accSize))

    else:

        concAccBag = accBag

    accEncodings = accNetwork(concAccBag)

    if useMIL:

        accAttentionWs = accMIL(accEncodings)
        accAttentionWs = tf.reshape(accAttentionWs, [batchSize, accBagSize])
        softmax = Softmax(name='weight_layer')
        accAttentionWs = tf.expand_dims(softmax(accAttentionWs), -2)
        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])

        if batchSize == 1:
            accPooling = tf.expand_dims(tf.squeeze(tf.matmul(accAttentionWs, accEncodings)), axis=0)
        else:
            accPooling = tf.squeeze(tf.matmul(accAttentionWs, accEncodings))

        yPred = classifier(accPooling)

    else:
        yPred = classifier(accEncodings)

    return Model([accBag, accPos], yPred)


def fit(data: Dataset,
        L=256, D=128,
        summary=True,
        verbose=0,
        mVerbose=False):

    train, val, test = data(accTransfer=True)

    logdir = os.path.join('logs_user' + str(data.testUser), 'accEncoderTb')

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
    w_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_val')
    w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')

    save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_type = 'accEncoder'
    model_name = 'TMD_%s_model.h5' % model_type
    filepath = os.path.join(save_dir, model_name)

    try:
        os.remove(filepath)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    val_steps = data.valSize // data.valBatchSize
    train_steps = data.trainSize // data.trainBatchSize
    test_steps = data.testSize // data.testBatchSize

    accNetwork = build(input_shapes=data.inputShape, args=data.shl_args, L=L, D=D)

    optimizer = Adam(learning_rate=float(data.lr))

    loss_function = CategoricalCrossentropy()

    accNetwork.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[categorical_accuracy]
    )

    val_metrics = accValMetrics(val, data.valBatchSize, val_steps, verbose)

    val_tables = accValTables(data.shl_args,
                              val,
                              data.valBatchSize,
                              val_steps,
                              file_writer_val,
                              w_file_writer_val)

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
        print(accNetwork.summary())

    if mVerbose:
        callbacks = [tensorboard_callback,
                     save_model,
                     early_stopping,
                     reduce_lr_plateau,
                     val_metrics,
                     val_tables]
    else:
        callbacks = [tensorboard_callback,
                     save_model,
                     early_stopping,
                     reduce_lr_plateau]

    accNetwork.fit(
        train,
        epochs=data.accEpochs,
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        verbose=verbose
    )

    test_metrics = accTestMetrics(test, data.testBatchSize, test_steps)

    test_tables = accTestTables(data.shl_args,
                                test,
                                data.testBatchSize,
                                test_steps,
                                file_writer_test,
                                w_file_writer_test)

    if mVerbose:
        callbacks = [test_metrics, test_tables]
    else:
        callbacks = [test_metrics]

    accNetwork.load_weights(filepath)
    accNetwork.evaluate(test, steps=test_steps, callbacks=callbacks)

    return filepath
