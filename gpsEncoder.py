import shutil
import tensorflow as tf
import tensorflow.keras as keras
import os
from dataset import Dataset
import tensorflow.keras.backend as K
from metrics import valMetricsGPS, testMetricsGPS, confusionMetricGPS, testConfusionMetricGPS


class MaskRelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)


def getGpsEncoder(input_shapes, args, L):
    mask = args.train_args['mask']
    kernelInitializer = keras.initializers.he_uniform()

    gpsSeriesShape = input_shapes[0]
    gpsFeaturesShape = input_shapes[1]

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
        kernel_initializer=kernelInitializer,
        name='locDense1'
    )

    bnLayer = keras.layers.BatchNormalization(name='locBatch1', trainable=False)
    activationLayer = MaskRelu()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = keras.layers.Dense(
        units=64,
        kernel_initializer=kernelInitializer,
        name='locDense2'
    )

    bnLayer = keras.layers.BatchNormalization(name='locBatch2', trainable=False)
    activationLayer = MaskRelu()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = keras.layers.Dense(
        units=L,
        kernel_initializer=kernelInitializer,
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

    gpsEncodings = tf.multiply(X, mask_w)

    return keras.models.Model(inputs=[gpsSeries, gpsFeatures],
                              outputs=gpsEncodings,
                              name='gpsEncoder')


def getClassifier(L):

    pooling = keras.Input(shape=(L))
    kernel_initializer = keras.initializers.glorot_uniform()

    X = pooling

    finalLayer = keras.layers.Dense(units=8,
                                    activation='softmax',
                                    kernel_initializer=kernel_initializer)

    yPred = finalLayer(X)

    return keras.models.Model(inputs = pooling,
                              outputs = yPred,
                              name = 'Classifier')


def build(input_shapes, args, L = 256):

    shape = input_shapes

    gpsNetwork = getGpsEncoder(shape, args, L)

    classifier = getClassifier(L)

    gpsSeries = keras.layers.Input(shape[0])
    gpsFeatures = keras.layers.Input(shape[1])
    gpsEncodings = gpsNetwork([gpsSeries, gpsFeatures])

    yPred = classifier(gpsEncodings)

    return keras.models.Model([gpsSeries, gpsFeatures], yPred)


def fit(L = 256, summary = True, verbose = 0):

    data = Dataset()

    train, val, test = data(gpsTransfer = True)

    logdir = os.path.join('logs_user' + str(data.testUser), 'gpsEncoderTb')

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')

    save_dir = os.path.join('training','saved_models','user'+str(data.testUser))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_type = 'gpsEncoder'
    model_name = 'TMD_%s_model.h5' %model_type
    filepath = os.path.join(save_dir, model_name)

    val_steps = data.valSize // data.valBatchSize
    train_steps = data.trainSize // data.trainBatchSize
    test_steps = data.testSize // data.testBatchSize

    Model = build(input_shapes=data.inputShape, args=data.shl_args, L=L)

    optimizer = keras.optimizers.Adam(learning_rate=data.lr)

    loss_function = keras.losses.CategoricalCrossentropy()

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[keras.metrics.categorical_accuracy]
    )

    val_metrics = valMetricsGPS(val, data.valBatchSize, val_steps, verbose)

    val_cm = confusionMetricGPS(val, data.valBatchSize, val_steps, file_writer_val)

    save_model = keras.callbacks.ModelCheckpoint(
        filepath = filepath,
        monitor = 'val_loss',
        verbose = verbose,
        save_best_only = True,
        mode = 'min',
        save_weights_only=True
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 30,
        mode = 'min',
        verbose = verbose
    )

    reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.4,
        patience = 10,
        verbose = verbose,
        mode = 'min'
    )

    if summary and verbose:
        print(Model.summary())

    callbacks = [tensorboard_callback,
                 save_model,
                 early_stopping,
                 reduce_lr_plateau]

    Model.fit(
        train,
        epochs=data.gpsEpochs,
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = callbacks,
        use_multiprocessing = True,
        verbose = verbose
    )

    test_metrics = testMetricsGPS(test,data.testBatchSize,test_steps)

    test_cm = testConfusionMetricGPS(test,
                                  data.testBatchSize,
                                  test_steps,
                                  file_writer_test)

    Model.load_weights(filepath)

    Model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    del data

    return filepath
