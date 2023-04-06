
import shutil
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Bidirectional, TimeDistributed
import tensorflow.keras.backend as K
from hmmlearn import hmm
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense
import accEncoder
import gpsEncoder
from dataset import Dataset
from hmm_params import hmmParams
from metrics import valMetrics, confusionMetric, testMetrics, testConfusionMetric


class MaskRelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)


def getAccEncoder(input_shapes, args, L, transfer = False):

    kernelInitializer = keras.initializers.he_uniform()
    dimension = 128
    shape = args.train_args['acc_shape']

    accShape = list(input_shapes[0])[1:]
    accBag = keras.Input(shape=accShape)
    X = accBag

    if shape == '2D':
        _, _, channels = accShape
    if shape == '1D':
        _, channels = accShape

    bnLayer = keras.layers.BatchNormalization(name='accBatch1')
    X = bnLayer(X)

    if shape == '2D':
        paddingLayer = keras.layers.ZeroPadding2D(padding=(1, 1))
        conv2D = keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv1'
        )

        bnLayer = keras.layers.BatchNormalization(name='accBatch2')
        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

        X = paddingLayer(X)
        X = conv2D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        paddingLayer = keras.layers.ZeroPadding2D(padding=(1, 1))
        conv2D = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv2'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch3')
        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

        X = paddingLayer(X)
        X = conv2D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        conv2D = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv3'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch4')
        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

        X = conv2D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

    if shape == '1D':
        paddingLayer = keras.layers.ZeroPadding1D(padding=1)
        conv1D = keras.layers.Conv1D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv1'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch2')
        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

        X = paddingLayer(X)
        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        paddingLayer = keras.layers.ZeroPadding1D(padding=1)
        conv1D = keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv2'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch3')
        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

        X = paddingLayer(X)
        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

        conv1D = keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernelInitializer,
            name='accConv3'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch4')
        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

    flattenLayer = keras.layers.Flatten()
    dropoutLayer = keras.layers.Dropout(rate = 0.3)

    X = flattenLayer(X)
    if not transfer:
        X = dropoutLayer(X)

    dnn = keras.layers.Dense(units=dimension,
                             kernel_initializer=kernelInitializer,
                             name='accDense1')
    bnLayer = keras.layers.BatchNormalization(name='accBatch5')
    activationLayer = keras.layers.ReLU()
    dropoutLayer = keras.layers.Dropout(rate=0.25)

    X = dnn(X)
    X = bnLayer(X)
    X = activationLayer(X)
    if not transfer:
        X = dropoutLayer(X)

    dnn = keras.layers.Dense(units=L,
                             kernel_initializer=kernelInitializer,
                             name='accDense2')
    bnLayer = keras.layers.BatchNormalization(name='accBatch6')
    activationLayer = keras.layers.ReLU()
    dropoutLayer = keras.layers.Dropout(rate=0.25)

    X = dnn(X)
    X = bnLayer(X)
    X = activationLayer(X)
    X = dropoutLayer(X)

    accEncodings = X

    return keras.models.Model(inputs=accBag,
                              outputs=accEncodings,
                              name='AccelerationEncoder')


def getGpsEncoder(input_shapes, args, L):
    mask = args.train_args['mask']
    kernelIntializer = keras.initializers.he_uniform()

    gpsSeriesShape = list(input_shapes[1])[1:]
    gpsFeaturesShape = list(input_shapes[2])[1:]
    gpsSeries = keras.Input(shape=gpsSeriesShape)
    gpsFeatures = keras.Input(shape=gpsFeaturesShape)

    X = gpsSeries

    masking_layer = keras.layers.Masking(mask_value=mask, name='maskLayer1')
    X = masking_layer(X)

    bnLayer = keras.layers.BatchNormalization(name='locBatch', trainable=False)
    X = bnLayer(X)

    lstmLayer = tf.keras.layers.LSTM(units=128,
                                     name='locLSTM')
    X = lstmLayer(X)

    X = tf.concat([X, gpsFeatures], axis=1)

    denseLayer = keras.layers.Dense(
        units=128,
        kernel_initializer=kernelIntializer,
        name='locDense1'
    )
    bnLayer = keras.layers.BatchNormalization(name='locBatch1', trainable=False)
    activationLayer = MaskRelu()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = keras.layers.Dense(
        units=64,
        kernel_initializer=kernelIntializer,
        name='locDense2'
    )
    bnLayer = keras.layers.BatchNormalization(name='locBatch2', trainable=False)
    activationLayer = MaskRelu()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = keras.layers.Dense(
        units=L,
        kernel_initializer=kernelIntializer,
        name='locDense3'
    )
    bnLayer = keras.layers.BatchNormalization(name='locBatch3', trainable=False)
    activationLayer = MaskRelu()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    mask_w = K.switch(
        tf.reduce_all(tf.equal(gpsFeatures, mask), axis=1, keepdims=True),
        lambda: tf.zeros_like(X), lambda: tf.ones_like(X)
    )

    locEncodings = tf.multiply(X, mask_w)

    return keras.models.Model(inputs=[gpsSeries, gpsFeatures],
                              outputs=locEncodings,
                              name='gpsEncoder')


def getMIL(args, L, D, head = 1):
    kernelInitializer = keras.initializers.glorot_uniform()
    kernelRegularizer = keras.regularizers.l2(0.01)

    encodings = keras.Input(shape = (L))

    D_layer = keras.layers.Dense(units=D,
                                 activation='tanh',
                                 kernel_initializer=kernelInitializer,
                                 kernel_regularizer=kernelRegularizer,
                                 name = 'D_layer')
    G_layer = keras.layers.Dense(units=D,
                                 activation='sigmoid',
                                 kernel_initializer=kernelInitializer,
                                 kernel_regularizer=kernelRegularizer,
                                 name = 'G_layer')
    K_layer = keras.layers.Dense(units=1,
                                 kernel_initializer = kernelInitializer,
                                 name = 'K_layer')

    attentionWs = D_layer(encodings)
    attentionWs = attentionWs * G_layer(encodings)
    attentionWs = K_layer(attentionWs)

    return keras.models.Model(inputs=encodings,
                              outputs=attentionWs,
                              name='MIL'+str(head))


def getHead(L, head=1):
    kernelInitializer = keras.initializers.he_uniform()
    pooling = keras.Input(shape=(L))
    X = pooling

    denseLayer = keras.layers.Dense(
        units = L // 2,
        kernel_initializer = kernelInitializer
    )
    bnLayer = keras.layers.BatchNormalization(trainable = False)
    activationLayer = keras.layers.ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    output = activationLayer(X)

    return keras.models.Model(inputs = pooling,
                              outputs = output,
                              name = 'head' + str(head))


def getClassifier(L):
    kernel_initializer = keras.initializers.glorot_uniform()
    head = keras.Input(shape=(L//2))
    X = head

    finalLayer = keras.layers.Dense(units=8,
                                    activation='softmax',
                                    kernel_initializer=kernel_initializer)

    yPred = finalLayer(X)

    return keras.models.Model(inputs = head,
                              outputs = yPred,
                              name = 'Classifier')


def build(input_shapes, args, L, D, accTransferNet = None, gpsTransferNet = None):

    n_heads = args.train_args['heads']
    useAccMIL = args.train_args['separate_MIL']
    accTransfer = True if accTransferNet else False
    accNetwork = getAccEncoder(input_shapes, args, L, accTransfer)

    if useAccMIL:
        accMIL = getMIL(args, L, D, True)

    if accTransfer:

        accNetwork.set_weights(accTransferNet.get_layer('AccelerationEncoder').get_weights())
        accNetwork.trainable = False

        if useAccMIL:

            accMIL.set_weights(accTransferNet.get_layer('AccelerationMIL').get_weights())
            accMIL.trainable = False

    gpsTransfer = True if gpsTransferNet else False
    gpsNetwork = getGpsEncoder(input_shapes, args, L)

    if gpsTransfer:

        gpsNetwork.set_weights(gpsTransferNet.get_layer('gpsEncoder').get_weights())
        gpsNetwork.trainable = False

    MILs = []
    heads = []

    for head in range(n_heads):
        MILs.append(getMIL(args, L, D, head=head))
        heads.append(getHead(L, head=head))

    classifier = getClassifier(L)

    accShape = input_shapes[0]
    accBagSize = list(accShape)[0]
    accSize = list(accShape)[1:]

    gpsSeriesShape = input_shapes[1]
    gpsBagSize = list(gpsSeriesShape)[0]
    gpsSeriesSize = list(gpsSeriesShape)[1:]
    gpsFeaturesShape = input_shapes[2]
    gpsFeaturesSize = list(gpsFeaturesShape)[1:]

    posShape = input_shapes[3]

    accBag = keras.layers.Input(accShape)
    gpsSeriesBag = keras.layers.Input(gpsSeriesShape)
    gpsFeaturesBag = keras.layers.Input(gpsFeaturesShape)
    accPos = keras.layers.Input(posShape, name='positional')
    batchSize = tf.shape(accBag)[0]

    concAccBags = tf.reshape(accBag, (batchSize * accBagSize, *accSize))
    concGpsSignalsBags = tf.reshape(gpsSeriesBag, (batchSize * gpsBagSize, *gpsSeriesSize))
    concGpsFeaturesBags = tf.reshape(gpsFeaturesBag, (batchSize * gpsBagSize, *gpsFeaturesSize))

    accEncodings = accNetwork(concAccBags)
    gpsEncodings = gpsNetwork([concGpsSignalsBags, concGpsFeaturesBags])

    if not useAccMIL:
        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])
        gpsEncodings = tf.reshape(gpsEncodings, [batchSize, gpsBagSize, L])
        Encodings = tf.keras.layers.concatenate([accEncodings, gpsEncodings], axis=-2)
        Encodings = tf.reshape(Encodings, (batchSize * (accBagSize + gpsBagSize), L))
        poolings = []

        for i, MIL in enumerate(MILs):
            attentionWs = MIL(Encodings)
            attentionWs = tf.reshape(attentionWs, [batchSize, accBagSize + gpsBagSize])
            softmax = keras.layers.Softmax(name='weight_layer_' + str(i))
            attentionWs = tf.expand_dims(softmax(attentionWs), -2)
            Encodings_ = tf.reshape(Encodings, [batchSize, accBagSize + gpsBagSize, L])
            Flatten = tf.keras.layers.Flatten()
            poolings.append(Flatten(tf.matmul(attentionWs, Encodings_)))

    else:
        accAttentionWs = accMIL(accEncodings)
        accAttentionWs = tf.reshape(accAttentionWs, [batchSize, accBagSize])
        softmax = keras.layers.Softmax(name='acc_weight_layer')
        accAttentionWs = tf.expand_dims(softmax(accAttentionWs), -2)
        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])
        accPooling = tf.matmul(accAttentionWs, accEncodings)
        gpsEncodings = tf.reshape(gpsEncodings, [batchSize, gpsBagSize, L])
        Encodings = tf.keras.layers.concatenate([accPooling, gpsEncodings], axis=-2)
        Encodings = tf.reshape(Encodings, (batchSize * (1 + gpsBagSize), L))
        poolings = []

        for i, MIL in enumerate(MILs):
            attentionWs = MIL(Encodings)
            attentionWs = tf.reshape(attentionWs, [batchSize, 1 + gpsBagSize])
            softmax = keras.layers.Softmax(name='weight_layer_'+str(i))
            attentionWs = tf.expand_dims(softmax(attentionWs), -2)
            Encodings_ = tf.reshape(Encodings, [batchSize, 1 + gpsBagSize, L])
            Flatten = tf.keras.layers.Flatten()
            poolings.append(Flatten(tf.matmul(attentionWs, Encodings_)))

    poolings =  tf.stack(poolings)
    heads = []
    for i in range(n_heads):
        heads.append(heads[i](poolings[i]))

    heads = tf.transpose(tf.stack(heads), perm=(1,0,2))
    head_pooling = keras.layers.AveragePooling1D(pool_size=(n_heads),strides=1)
    Flatten = tf.keras.layers.Flatten()
    head = Flatten(head_pooling(heads))
    yPred = classifier(head)

    return keras.models.Model([accBag, gpsSeriesBag, gpsFeaturesBag, accPos], yPred)


def TMD_MIL(data: Dataset,
            summary = False,
            verbose = 0):

    L = 256
    D = 128

    if data.gpsMode == 'load':

        data(gpsTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'gpsEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        gpsNetwork = gpsEncoder.build(data.inputShape, data.shl_args, L)

        optimizer = keras.optimizers.Adam(
            learning_rate=data.lr
        )

        loss_function = keras.losses.CategoricalCrossentropy()

        gpsNetwork.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[keras.metrics.categorical_accuracy]
        )

        gpsNetwork.load_weights(filepath)

    elif data.gpsMode == 'train':

        filepath = gpsEncoder.fit(
            L=L,
            summary=summary,
            verbose=verbose
        )

        data(gpsTransfer=True)

        gpsNetwork = gpsEncoder.build(data.inputShape, data.shl_args, L)

        gpsNetwork.load_weights(filepath)

    else:

        gpsNetwork = None

    if data.accMode == 'load':

        data(accTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'accEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        accNetwork = accEncoder.build(data.inputShape,
                                      data.shl_args,
                                      L, D)

        optimizer = keras.optimizers.Adam(
            learning_rate=data.lr
        )

        loss_function = keras.losses.CategoricalCrossentropy()

        accNetwork.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[keras.metrics.categorical_accuracy]
        )

        accNetwork.load_weights(filepath)

    elif data.accMode == 'train':

        filepath = accEncoder.fit(
            data = data,
            L=L, D=D,
            summary=summary,
            verbose=verbose
        )

        data(accTransfer=True)

        accNetwork = accEncoder.build(data.inputShape, data.shl_args, L, D)

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

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

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

    val_steps = data.valSize // data.valBatchSize
    train_steps = data.trainSize // data.trainBatchSize
    test_steps = data.testSize // data.testBatchSize

    optimizer = keras.optimizers.Adam(
        learning_rate = data.lr
    )

    loss_function = keras.losses.CategoricalCrossentropy()

    Model = build(data.inputShape,
                     data.shl_args,
                     L, D, accNetwork,
                     gpsNetwork)

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[keras.metrics.categorical_accuracy]
    )

    val_metrics = valMetrics(val, data.valBatchSize, val_steps, verbose = verbose)

    val_cm = confusionMetric(data.shl_args,
                              val,
                              data.valBatchSize,
                              val_steps,
                              file_writer_val,
                              w_file_writer_val,
                              w_pos_file_writer_val)

    save_model = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        mode='min',
        save_weights_only=True
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        mode='min',
        verbose=verbose
    )

    reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
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
                 reduce_lr_plateau]

    Model.fit(
        train,
        epochs=data.epochs,
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        verbose=verbose
    )

    Model.load_weights(filepath)

    test_metrics = testMetrics(test, data.testBatchSize, test_steps)

    test_cm = testConfusionMetric(data.shl_args,
                                  test,
                                  data.testBatchSize,
                                  test_steps,
                                  file_writer_test,
                                  w_file_writer_test,
                                  w_pos_file_writer_test)

    callbacks = [test_metrics]

    Model.evaluate(test, steps=test_steps, callbacks=callbacks)

    if data.postprocessing:
        params = hmmParams()

        if data.testUser == 1:
            conf = params.conf1
            trans = params.trans1

        elif data.testUser == 2:
            conf = params.conf2
            trans = params.trans2

        elif data.testUser == 3:
            conf = params.conf3
            trans = params.trans3

        startprob = [1. / 8 for _ in range(8)]

        postprocessing_model = hmm.MultinomialHMM(n_components=8,
                                                  algorithm='viterbi',
                                                  random_state=93,
                                                  n_iter=100
                                                  )

        postprocessing_model.startprob_ = startprob
        postprocessing_model.transmat_ = trans
        postprocessing_model.emissionprob_ = conf

        x, y, lengths = data.postprocess(Model=Model, fit=False)

        y_ = postprocessing_model.predict(x, lengths)
        score = sklearn.metrics.accuracy_score(y, y_)
        f1_score = sklearn.metrics.f1_score(y, y_, average='macro')

        print()
        print(score)
        print(f1_score)
        print()

        score = sklearn.metrics.accuracy_score(y, x)
        f1_score = sklearn.metrics.f1_score(y, x, average='macro')

        print(score)
        print(f1_score)
        print()

        postprocessing_model = hmm.MultinomialHMM(n_components=8,
                                                  algorithm='viterbi',
                                                  random_state=93,
                                                  n_iter=100
                                                  )

        postprocessing_model.startprob_ = startprob
        postprocessing_model.transmat_ = params.totalTrans
        postprocessing_model.emissionprob_ = params.totalConf

        y_ = postprocessing_model.predict(x, lengths)
        score = sklearn.metrics.accuracy_score(y, y_)
        f1_score = sklearn.metrics.f1_score(y, y_, average='macro')

        print()
        print(score)
        print(f1_score)
        print()

        score = sklearn.metrics.accuracy_score(y, x)
        f1_score = sklearn.metrics.f1_score(y, x, average='macro')

        print(score)
        print(f1_score)

    del data

    return