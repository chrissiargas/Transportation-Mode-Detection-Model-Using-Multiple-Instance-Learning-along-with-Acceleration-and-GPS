import io
import itertools
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from dataset import SignalsDataset
import tensorflow.keras.backend as K



class valMetrics(keras.callbacks.Callback):
    def __init__(self, val, batchSize, steps, verbose = 1):
        super(valMetrics, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps
        self.score = 'macro'
        self.verbose = verbose


    def on_epoch_end(self, epoch, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))


        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            val_predict[step * self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(val_data)),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)
            step += 1

        f1 = f1_score(val_true, val_predict, average=self.score)
        recall = recall_score(val_true, val_predict, average=self.score)
        precision =precision_score(val_true, val_predict, average=self.score)

        del val_predict
        del val_true

        if self.verbose:
            print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(f1,precision,recall))

        return

class testMetrics(keras.callbacks.Callback):
    def __init__(self, test, batchSize, steps, verbose = 1):
        super(testMetrics, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps
        self.score = 'macro'
        self.verbose = verbose

    def on_test_end(self, logs=None):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))

        for batch in self.test.take(self.steps):

            test_data = batch[0]
            test_target = batch[1]


            test_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(test_data)),axis=1)
            test_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(test_target,axis=1)
            step += 1

        test_f1 = f1_score(test_true, test_predict, average=self.score)
        test_recall = recall_score(test_true, test_predict, average=self.score)
        test_precision = precision_score(test_true, test_predict, average=self.score)

        del test_predict
        del test_true

        if self.verbose:
            print(" - test_f1: %f - test_precision: %f - test_recall %f" %(test_f1,test_precision,test_recall))

        return

class confusion_metric(keras.callbacks.Callback):
    def __init__(self,
                 val,
                 batchSize,
                 steps,
                 file_writer):

        super(confusion_metric, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps

        self.class_names = [
            'Still',
            'Walking',
            'Run',
            'Bike',
            'Car',
            'Bus',
            'Train',
            'Subway'
        ]


        self.file_writer = file_writer



    def on_train_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))

        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            pred = self.model.call(val_data)

            val_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(pred),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)

            step += 1

        val_f1 = f1_score(val_true, val_predict, average="macro")
        val_recall = recall_score(val_true, val_predict, average="macro")
        val_precision =precision_score(val_true, val_predict, average="macro")

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(val_f1,val_precision,val_recall))


        cm = confusion_matrix(val_true,val_predict)
        global CM
        CM = cm/ cm.sum(axis=1)[:, np.newaxis]

        cm_df = pd.DataFrame(cm,
                             index = self.class_names,
                             columns = self.class_names)
        cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        figure = plt.figure(figsize=(10,10))
        sns.heatmap(cm_df, annot = True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        cm_image = plot_to_image(figure)

        with self.file_writer.as_default():
            tf.summary.image('Confusion Matrix', cm_image, step=1)



        return


class testConfusionMetric(keras.callbacks.Callback):
    def __init__(self,
                 test,
                 batchSize,
                 steps,
                 file_writer):

        super(testConfusionMetric, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps

        self.class_names = [
            'Still',
            'Walking',
            'Run',
            'Bike',
            'Car',
            'Bus',
            'Train',
            'Subway'
        ]

        self.file_writer = file_writer

    def on_test_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))


        for batch in self.test.take(self.steps):

            test_data = batch[0]
            test_target = batch[1]

            pred = self.model.call(test_data)

            test_predict[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(np.asarray(pred), axis=1)
            test_true[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(test_target, axis=1)

            step += 1

        test_f1 = f1_score(test_true, test_predict, average="macro")
        test_recall = recall_score(test_true, test_predict, average="macro")
        test_precision = precision_score(test_true, test_predict, average="macro")

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" % (test_f1, test_precision, test_recall))

        cm = confusion_matrix(test_true, test_predict)
        global CM
        CM = cm / cm.sum(axis=1)[:, np.newaxis]

        cm_df = pd.DataFrame(cm,
                             index=self.class_names,
                             columns=self.class_names)
        cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        cm_image = plot_to_image(figure)

        with self.file_writer.as_default():
            tf.summary.image('Confusion Matrix', cm_image, step=1)

        return



class MaskRelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)



def get_loc_encoder(input_shapes, args, L):
    kernel_initializer = keras.initializers.he_uniform()

    inputLocSignalsShape = input_shapes[0]
    inputLocFeaturesShape = input_shapes[1]


    inputLocSignals = keras.Input(shape=inputLocSignalsShape)
    inputLocFeatures = keras.Input(shape=inputLocFeaturesShape)

    mask = args.train_args['mask']

    X = inputLocSignals

    masking_layer = keras.layers.Masking(mask_value=mask, name='maskLayer1')
    X = masking_layer(X)

    bnLayer = keras.layers.BatchNormalization(name='locBatch', trainable=False)
    X = bnLayer(X)

    if args.train_args['GPSNet'] == 'LSTM':
        lstmLayer = tf.keras.layers.LSTM(units=128,
                                         name='locLSTM')
        X = lstmLayer(X)

    elif args.train_args['GPSNet'] == 'FCLSTM':
        TDDenseLayer = TimeDistributed(
            keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer,
                name='TDlocDense1'
            )
        )

        TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch1')
        TDactivationLayer = MaskRelu()

        X = TDDenseLayer(X)
        X = TDbnLayer(X)
        X = TDactivationLayer(X)

        TDDenseLayer = TimeDistributed(
            keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer,
                name='TDlocDense2'
            )
        )

        TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch2')
        TDactivationLayer = MaskRelu()

        X = TDDenseLayer(X)
        X = TDbnLayer(X)
        X = TDactivationLayer(X)

        lstmLayer1 = tf.keras.layers.LSTM(units=128,
                                          return_sequences=True,
                                          name='locLSTM1')
        lstmLayer2 = tf.keras.layers.LSTM(units=128,
                                          name='locLSTM2')

        X = lstmLayer1(X)
        X = lstmLayer2(X)



    X = tf.concat([X, inputLocFeatures], axis=1)

    denseLayer = keras.layers.Dense(
        units=128,
        kernel_initializer=kernel_initializer,
        name='locDense1'
    )

    bnLayer = keras.layers.BatchNormalization(name='locBatch1', trainable=False)
    activationLayer = MaskRelu()

    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)

    denseLayer = keras.layers.Dense(
        units=64,
        kernel_initializer=kernel_initializer,
        name='locDense2'
    )

    bnLayer = keras.layers.BatchNormalization(name='locBatch2', trainable=False)
    activationLayer = MaskRelu()


    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)


    denseLayer = keras.layers.Dense(
        units=L,
        kernel_initializer=kernel_initializer,
        name='locDense3'
    )

    bnLayer = keras.layers.BatchNormalization(name='locBatch3', trainable=False)
    activationLayer = MaskRelu()


    X = denseLayer(X)
    X = bnLayer(X)
    X = activationLayer(X)


    mask_w = K.switch(
        tf.reduce_all(tf.equal(inputLocFeatures, mask), axis=1, keepdims=True),
        lambda: tf.zeros_like(X), lambda: tf.ones_like(X)
    )

    loc_encodings = tf.multiply(X, mask_w)

    return keras.models.Model(inputs=[inputLocSignals, inputLocFeatures],
                              outputs=loc_encodings,
                              name='LocationEncoder')

def get_classifier(L):

    pooling = keras.Input(shape=(L))
    kernel_initializer = keras.initializers.glorot_uniform()

    X = pooling



    finalLayer = keras.layers.Dense(units=8,
                                    activation='softmax',
                                    kernel_initializer=kernel_initializer)


    y_pred = finalLayer(X)


    return keras.models.Model(inputs = pooling,
                              outputs = y_pred,
                              name = 'Classifier')

def create_model(input_shapes,
                 args,
                 L = 256):

    locShape = input_shapes

    loc_encoder = get_loc_encoder(locShape, args, L)

    classifier = get_classifier(L)

    locSignals = keras.layers.Input(locShape[0])
    locFeatures = keras.layers.Input(locShape[1])



    locEncodings = loc_encoder([locSignals, locFeatures])

    y_pred = classifier(locEncodings)

    return keras.models.Model([locSignals, locFeatures], y_pred)


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image,0)

    return image

import ruamel.yaml

def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            original_value = data[args][param]
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)

    return original_value


def fit(L = 256,
        summary = True,
        verbose = 0,
        user_seperated = False,
        round = False):

    SD = SignalsDataset()

    train, val, test = SD(locTransfer = True, user_seperated = user_seperated, round = round)

    user = SD.shl_args.train_args['test_user']
    logdir = os.path.join('logs_user' + str(user),'loc_transfer_tensorboard')

    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    val_steps = SD.valSize // SD.valBatchSize

    train_steps = SD.trainSize // SD.trainBatchSize

    test_steps = SD.testSize // SD.testBatchSize

    lr = SD.shl_args.train_args['learning_rate']

    Model = create_model(input_shapes=SD.inputShape, args=SD.shl_args, L=L)

    optimizer = keras.optimizers.Adam(
        learning_rate=lr
    )

    loss_function = keras.losses.CategoricalCrossentropy()

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[keras.metrics.categorical_accuracy]
    )

    val_metrics = valMetrics(val, SD.valBatchSize, val_steps, verbose)

    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')

    val_cm = confusion_metric(val,
                              SD.valBatchSize,
                              val_steps,
                              file_writer_val)

    save_dir = os.path.join('training','saved_models','user'+str(user))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    model_type = 'location_classifier'
    model_name = 'shl_%s_model.h5' %model_type
    filepath = os.path.join(save_dir, model_name)

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
        epochs=SD.shl_args.train_args['locEpochs'],
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = callbacks,
        use_multiprocessing = True,
        verbose = verbose
    )

    test_metrics = testMetrics(test,SD.testBatchSize,test_steps)

    test_cm = testConfusionMetric(test,
                                  SD.testBatchSize,
                                  test_steps,
                                  file_writer_test)

    Model.load_weights(filepath)

    Model.evaluate(test,steps=test_steps,callbacks=[test_metrics])

    del SD.acceleration
    del SD.location
    del SD.labels

    return filepath
