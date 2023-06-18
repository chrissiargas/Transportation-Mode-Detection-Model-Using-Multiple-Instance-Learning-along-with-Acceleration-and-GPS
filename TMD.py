import shutil
import tensorflow as tf
import keras
import os
import keras.backend as K
from keras.layers import *
from keras import Input
from keras import initializers
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
import accEncoder
import gpsEncoder
import postprocess
from dataset import Dataset
from myMetrics import valMetrics, valTables, testMetrics, testTables
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def getAccEncoder(input_shapes, args, L, transfer=False):
    kernelInitializer = initializers.he_uniform()
    dimension = 128
    shape = args.train_args['acc_shape']

    accShape = list(input_shapes[0])[1:]
    accBag = Input(shape=accShape)
    X = accBag

    if shape == '2D':
        _, _, channels = accShape
    if shape == '1D':
        _, channels = accShape

    bnLayer = BatchNormalization(name='accBatch1')
    X = bnLayer(X)

    if shape == '2D':
        paddingLayer = ZeroPadding2D(padding=(1, 1))
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

        paddingLayer = ZeroPadding2D(padding=(1, 1))
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
        paddingLayer = ZeroPadding1D(padding=1)
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

        paddingLayer = ZeroPadding1D(padding=1)
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
    if not transfer:
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
    if not transfer:
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

    return Model(inputs=accBag,
                 outputs=accEncodings,
                 name='AccelerationEncoder')


def getGpsEncoder(input_shapes, args, L):
    mask = args.train_args['mask']

    kernelIntializer = initializers.he_uniform()

    gpsSeriesShape = list(input_shapes[1])[1:]
    gpsFeaturesShape = list(input_shapes[2])[1:]
    gpsSeries = Input(shape=gpsSeriesShape)
    gpsFeatures = Input(shape=gpsFeaturesShape)

    X = gpsSeries
    # X = tf.reshape(X, (-1, gpsSeriesShape[0] * gpsSeriesShape[1]))

    masking_layer = Masking(mask_value=mask, name='maskLayer1')
    X = masking_layer(X)

    bnLayer = BatchNormalization(name='locBatch', trainable=False)
    X = bnLayer(X)

    lstmLayer = Bidirectional(LSTM(units=128, name='locLSTM'))
    X = lstmLayer(X)

    X = tf.concat([X, gpsFeatures], axis=1)

    denseLayer = Dense(
        units=128,
        kernel_initializer=kernelIntializer,
        name='locDense1'
    )
    bnLayer = BatchNormalization(name='locBatch1', trainable=False)
    activationLayer = ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = Dense(
        units=64,
        kernel_initializer=kernelIntializer,
        name='locDense2'
    )
    bnLayer = BatchNormalization(name='locBatch2', trainable=False)
    activationLayer = ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = Dense(
        units=L,
        kernel_initializer=kernelIntializer,
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

    locEncodings = tf.multiply(X, mask_w)

    return Model(inputs=[gpsSeries, gpsFeatures],
                 outputs=locEncodings,
                 name='gpsEncoder')


def getMIL(L, D, head=1):
    kernelInitializer = initializers.glorot_uniform()
    kernelRegularizer = l2(0.01)

    encodings = Input(shape=L)

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

    attentionWs = D_layer(encodings)
    attentionWs = attentionWs * G_layer(encodings)
    attentionWs = K_layer(attentionWs)

    return Model(inputs=encodings,
                 outputs=attentionWs,
                 name='MIL' + str(head))


def getHead(L, head=1):
    kernelInitializer = initializers.he_uniform()
    pooling = Input(shape=L)
    X = pooling

    denseLayer = Dense(
        units=L // 2,
        kernel_initializer=kernelInitializer
    )
    bnLayer = BatchNormalization(trainable=False)
    activationLayer = ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    output = activationLayer(X)

    return Model(inputs=pooling,
                 outputs=output,
                 name='head' + str(head))


def getClassifier(L, n_units=8, noHeads=False):
    kernel_initializer = initializers.glorot_uniform()
    if noHeads:
        head = Input(shape=(L))
    else:
        head = Input(shape=(L // 2))

    X = head

    finalLayer = Dense(units=n_units,
                       activation='softmax',
                       kernel_initializer=kernel_initializer)

    yPred = finalLayer(X)

    return Model(inputs=head,
                 outputs=yPred,
                 name='Classifier')


def build(input_shapes, args, accTransferNet=None, gpsTransferNet=None):
    n_heads = args.train_args['heads']
    useAccMIL = args.train_args['separate_MIL']
    motorized = args.train_args['motorized']
    fusion = args.train_args['fusion']
    finetuning = args.train_args['finetuning']
    useLayerNorm = args.train_args['layer_norm']
    noHeads = n_heads == 0

    L = args.train_args['L']
    D = args.train_args['D']
    n_classes = 5 if motorized else 8

    accShape = input_shapes[0]
    accBagSize = list(accShape)[0]
    accSize = list(accShape)[1:]

    gpsSeriesShape = input_shapes[1]
    gpsBagSize = list(gpsSeriesShape)[0]
    gpsSeriesSize = list(gpsSeriesShape)[1:]
    gpsFeaturesShape = input_shapes[2]
    gpsFeaturesSize = list(gpsFeaturesShape)[1:]

    posShape = input_shapes[3]

    accTransfer = True if accTransferNet else False
    accNetwork = getAccEncoder(input_shapes, args, L, accTransfer)

    if useAccMIL:
        accMIL = getMIL(L, D)

    if accTransfer:

        accNetwork.set_weights(accTransferNet.get_layer('AccelerationEncoder').get_weights())
        accNetwork.trainable = finetuning

        if useAccMIL:
            accMIL.set_weights(accTransferNet.get_layer('AccelerationMIL').get_weights())
            accMIL.trainable = finetuning

    gpsTransfer = True if gpsTransferNet else False
    gpsNetwork = getGpsEncoder(input_shapes, args, L)

    if gpsTransfer:
        gpsNetwork.set_weights(gpsTransferNet.get_layer('gpsEncoder').get_weights())
        gpsNetwork.trainable = finetuning

    MILs = []
    heads = []

    if fusion == 'concat':
        for head in range(n_heads):
            if useAccMIL:
                heads.append(getHead(L * (1 + gpsBagSize), head=head))
            else:
                heads.append(getHead(L * (accBagSize + gpsBagSize), head=head))

        if useAccMIL:
            classifier = getClassifier(L * (1 + gpsBagSize), n_units=n_classes, noHeads=noHeads)
        else:
            classifier = getClassifier(L * (accBagSize + gpsBagSize), n_units=n_classes, noHeads=noHeads)

    elif fusion == 'MIL':
        if n_heads == 0:
            MILs.append(getMIL(L, D, head=0))

        else:
            for head in range(n_heads):
                MILs.append(getMIL(L, D, head=head))
                heads.append(getHead(L, head=head))

        classifier = getClassifier(L, n_units=n_classes, noHeads=noHeads)

    accBags = Input(accShape)
    gpsSeriesBags = Input(gpsSeriesShape)
    gpsFeaturesBags = Input(gpsFeaturesShape)
    accPos = Input(posShape, name='positional')

    batchSize = tf.shape(accBags)[0]
    reshapedAccBags = tf.reshape(accBags, (batchSize * accBagSize, *accSize))
    reshapedGpsSeriesBags = tf.reshape(gpsSeriesBags, (batchSize * gpsBagSize, *gpsSeriesSize))
    reshapedGpsFeaturesBags = tf.reshape(gpsFeaturesBags, (batchSize * gpsBagSize, *gpsFeaturesSize))

    accEncodings = accNetwork(reshapedAccBags)
    gpsEncodings = gpsNetwork([reshapedGpsSeriesBags, reshapedGpsFeaturesBags])

    if not useAccMIL:
        if useLayerNorm:
            accLayerNorm = LayerNormalization(axis=1, center=True, scale=True)
            gpsLayerNorm = LayerNormalization(axis=1, center=True, scale=True)

            accEncodings = accLayerNorm(accEncodings)
            gpsEncodings = gpsLayerNorm(gpsEncodings)

        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])
        gpsEncodings = tf.reshape(gpsEncodings, [batchSize, gpsBagSize, L])
        Encodings = concatenate([accEncodings, gpsEncodings], axis=-2)

        poolings = []
        if fusion == 'MIL':
            Encodings = tf.reshape(Encodings, (batchSize * (accBagSize + gpsBagSize), L))

            for i, MIL in enumerate(MILs):
                attentionWs = MIL(Encodings)
                attentionWs = tf.reshape(attentionWs, [batchSize, accBagSize + gpsBagSize])
                softmax = Softmax(name='weight_layer_' + str(i))
                attentionWs = tf.expand_dims(softmax(attentionWs), -2)
                Encodings_ = tf.reshape(Encodings, [batchSize, accBagSize + gpsBagSize, L])
                flatten = Flatten()
                poolings.append(flatten(tf.matmul(attentionWs, Encodings_)))

        if fusion == 'concat':
            Encodings = tf.reshape(Encodings, (batchSize, L * (accBagSize + gpsBagSize)))
            for i in range(n_heads):
                poolings.append(Encodings)

    else:
        accAttentionWs = accMIL(accEncodings)
        accAttentionWs = tf.reshape(accAttentionWs, [batchSize, accBagSize])
        softmax = Softmax(name='acc_weight_layer')
        accAttentionWs = tf.expand_dims(softmax(accAttentionWs), -2)
        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])
        accPooling = tf.matmul(accAttentionWs, accEncodings)

        if useLayerNorm:
            accLayerNorm = LayerNormalization(axis=1, center=True, scale=True)
            gpsLayerNorm = LayerNormalization(axis=1, center=True, scale=True)

            accPooling = accLayerNorm(accPooling)
            gpsEncodings = gpsLayerNorm(gpsEncodings)

        gpsEncodings = tf.reshape(gpsEncodings, [batchSize, gpsBagSize, L])
        Encodings = concatenate([accPooling, gpsEncodings], axis=-2)

        poolings = []
        if fusion == 'MIL':
            Encodings = tf.reshape(Encodings, (batchSize * (1 + gpsBagSize), L))
            poolings = []

            for i, MIL in enumerate(MILs):
                attentionWs = MIL(Encodings)
                attentionWs = tf.reshape(attentionWs, [batchSize, 1 + gpsBagSize])
                softmax = Softmax(name='weight_layer_' + str(i))
                attentionWs = tf.expand_dims(softmax(attentionWs), -2)
                Encodings_ = tf.reshape(Encodings, [batchSize, 1 + gpsBagSize, L])
                flatten = Flatten()
                poolings.append(flatten(tf.matmul(attentionWs, Encodings_)))

        elif fusion == 'concat':
            Encodings = tf.reshape(Encodings, (batchSize, L * (1 + gpsBagSize)))
            for i in range(n_heads):
                poolings.append(Encodings)

    poolings = tf.stack(poolings)

    if noHeads:
        yPred = classifier(poolings[0])

    else:
        headOutputs = []
        for i in range(n_heads):
            headOutputs.append(heads[i](poolings[i]))

        headOutputs = tf.transpose(tf.stack(headOutputs), perm=(1, 0, 2))
        head_pooling = AveragePooling1D(pool_size=n_heads, strides=1)
        flatten = Flatten()
        head = flatten(head_pooling(headOutputs))
        yPred = classifier(head)

    return keras.models.Model([accBags, gpsSeriesBags, gpsFeaturesBags, accPos], yPred)


def TMD_MIL(data: Dataset,
            summary=False,
            verbose=0,
            mVerbose=False,
            postprocessing=True,
            accScores=False,
            gpsScores=False):

    data.initialize()

    accG = 1.
    f1G = 1.

    if data.gpsMode == 'load':

        data(gpsTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'gpsEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        gpsNetwork = gpsEncoder.build(data.inputShape, data.shl_args)

        optimizer = Adam(
            learning_rate=float(data.lr)
        )

        loss_function = CategoricalCrossentropy()

        gpsNetwork.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[categorical_accuracy]
        )

        gpsNetwork.load_weights(filepath)

    elif data.gpsMode == 'train':

        filepath, accG, f1G = gpsEncoder.fit(
            summary=summary,
            verbose=verbose,
            mVerbose=mVerbose,
            scores=gpsScores
        )

        if gpsScores:

            del data.acceleration
            del data.location
            del data.labels
            del data

            return 1., 1., 1., 1., 1., 1., 1., accG, f1G

        data(gpsTransfer=True)

        gpsNetwork = gpsEncoder.build(data.inputShape, data.shl_args)

        gpsNetwork.load_weights(filepath)

    else:

        gpsNetwork = None

    accA = 1.
    f1A = 1.
    cmA = 1.

    if data.accMode == 'load':

        data(accTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'accEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        accNetwork = accEncoder.build(data.inputShape, data.shl_args)

        optimizer = Adam(
            learning_rate=float(data.lr)
        )

        loss_function = CategoricalCrossentropy()

        accNetwork.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[categorical_accuracy]
        )

        accNetwork.load_weights(filepath)

    elif data.accMode == 'train':

        filepath, accA, f1A, cmA = accEncoder.fit(
            data=data,
            summary=summary,
            verbose=verbose,
            mVerbose=mVerbose,
            scores=accScores
        )

        if accScores:

            del data.acceleration
            del data.location
            del data.labels
            del data

            return 1., 1., 1., 1., accA, f1A, cmA, 1., 1.

        data(accTransfer=True)

        accNetwork = accEncoder.build(data.inputShape, data.shl_args)

        accNetwork.load_weights(filepath)

    else:

        accNetwork = None

    train, val, test = data()

    logdir = os.path.join('logs_user' + str(data.testUser), 'fullModelTb')

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
    w_pos_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_pos_val')
    w_pos_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_pos_test')

    save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_type = 'full'
    model_name = 'TMD_%s_model.h5' % model_type
    filepath = os.path.join(save_dir, model_name)

    try:
        os.remove(filepath)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    val_steps = data.valSize // data.valBatchSize
    train_steps = data.trainSize // data.trainBatchSize
    test_steps = data.testSize // data.testBatchSize

    optimizer = Adam(learning_rate=float(data.lr))

    loss_function = CategoricalCrossentropy()

    TMDMiller = build(data.inputShape, data.shl_args, accNetwork, gpsNetwork)

    TMDMiller.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=[categorical_accuracy])

    val_metrics = valMetrics(val, data.valBatchSize, val_steps, verbose=verbose)

    val_tables = valTables(data.shl_args,
                           val,
                           data.valBatchSize,
                           val_steps,
                           file_writer_val,
                           w_file_writer_val,
                           w_pos_file_writer_val)

    save_model = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        mode='min',
        save_weights_only=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        mode='min',
        verbose=verbose)

    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=10,
        verbose=verbose,
        mode='min')

    if summary and verbose:
        print(TMDMiller.summary())

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

    TMDMiller.fit(train,
                  epochs=data.epochs,
                  steps_per_epoch=train_steps,
                  validation_data=val,
                  validation_steps=val_steps,
                  callbacks=callbacks,
                  use_multiprocessing=True,
                  verbose=verbose)

    TMDMiller.load_weights(filepath)

    test_metrics = testMetrics(test, data.testBatchSize, test_steps)

    test_tables = testTables(data.shl_args,
                             test,
                             data.testBatchSize,
                             test_steps,
                             file_writer_test,
                             w_file_writer_test,
                             w_pos_file_writer_test)

    if mVerbose:
        callbacks = [test_metrics, test_tables]
    else:
        callbacks = [test_metrics]

    TMDMiller.evaluate(test, steps=test_steps, callbacks=callbacks)

    train_dataset, _, _ = postprocess.get_dataset(data=data, Model=TMDMiller, train=True)
    test_dataset, y, y_ = postprocess.get_dataset(data=data, Model=TMDMiller, train=False)

    accuracy = accuracy_score(y, y_)
    f1 = f1_score(y, y_, average='macro')

    modes = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    cm = confusion_matrix(y, y_)
    cm_df = pd.DataFrame(cm, index=['{:}'.format(x) for x in modes],
                         columns=['{:}'.format(x) for x in modes])
    cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Accuracy without post-processing: {}'.format(accuracy))
    print('F1-Score without post-processing: {}'.format(f1))

    postAccuracy = None
    postF1 = None

    if postprocessing:
        postY_ = postprocess.fit_predict(train_dataset, test_dataset)

        postAccuracy = accuracy_score(y, postY_)
        postF1 = f1_score(y, postY_, average='macro')

        print('Accuracy with post-processing: {}'.format(postAccuracy))
        print('F1-Score with post-processing: {}'.format(postF1))

    del data.acceleration
    del data.location
    del data.labels
    del data

    return accuracy, f1, postAccuracy, postF1, cm_df, accA, f1A, cmA, accG, f1G

