import io
import itertools
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
import simCLR


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

            val_predict[step * self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(val_data, verbose=0)),axis=1)
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


            test_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(self.model.predict(test_data, verbose=0)),axis=1)
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
                 args,
                 val,
                 batchSize,
                 steps,
                 file_writer,
                 weights_file_writer):

        super(confusion_metric, self).__init__()
        self.val = val
        self.batchSize = batchSize
        self.steps = steps
        self.accBagSize = args.train_args['accBagSize']
        self.MIL = args.train_args['seperate_MIL']
        self.random_position = args.train_args['random_position']

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
        self.weights_file_writer = weights_file_writer

        if args.train_args['positions'] == None:

            if args.data_args['dataset'] == 'CompleteUser1':
                self.pnl = ['Hips']

            else:
                self.pnl = ['Torso','Hips','Bag','Hand']

        else:
            self.pnl = args.train_args['positions']


    def on_train_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        val_predict = np.zeros((total))
        val_true = np.zeros((total))

        if self.MIL and self.random_position:

            self.weighting = keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("weight_layer").output
            )

            self.positional = keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("positional").output
            )

            weights = np.zeros((total,self.accBagSize))
            positions = np.zeros((total, self.accBagSize))

        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            pred = self.model.call(val_data)

            val_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(pred),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)

            if self.MIL and self.random_position:
                weights[step * self.batchSize: (step + 1) * self.batchSize] = self.weighting(val_data)
                positions[step * self.batchSize: (step + 1) * self.batchSize] = self.positional(val_data)

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


        if self.MIL and self.random_position:

            fig, axs = plt.subplots(ncols=1, figsize=(12, 16))
            fig.suptitle('Weight Matrix')

            for i in range(self.accBagSize):
                wm_pred = np.concatenate([val_predict[:, np.newaxis], positions[:, [i]], weights[:, [i]]], axis=1)
                wm_pred_df = pd.DataFrame(
                    wm_pred,
                    columns=['class', 'position', 'weight']
                )

                if i==0:
                    wm_pos_sum = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                    wm_pos_sum = pd.pivot_table(wm_pos_sum, values="weight", index=["class"], columns=["position"],
                                                fill_value=0)




                    wm_pos_count = wm_pred_df.groupby(['class', 'position']).size().to_frame(name='size').reset_index()
                    wm_pos_count = pd.pivot_table(wm_pos_count, values="size", index=["class"], columns=["position"],
                                                    fill_value=0)

                else:
                    wm_pos_sum_ = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                    wm_pos_sum = wm_pos_sum.add(pd.pivot_table(wm_pos_sum_, values="weight", index=["class"], columns=["position"],
                                                fill_value=0).values)

                    wm_pos_count_ = wm_pred_df.groupby(['class', 'position']).size().to_frame(name='size').reset_index()
                    wm_pos_count = wm_pos_count.add(pd.pivot_table(wm_pos_count_, values="size", index=["class"],
                                                  columns=["position"],
                                                  fill_value=0).values)


                if i == self.accBagSize-1:
                    wm_pos = wm_pos_sum.div(wm_pos_count.values)
                    wm_pos = wm_pos.div(wm_pos.sum(axis=1), axis=0)

                    sns.heatmap(wm_pos, ax=axs, cbar=False, annot=True)
                    fig.colorbar(axs.collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)
                    axs.set_yticklabels(labels=self.class_names, rotation=45)
                    axs.set_xticklabels(labels=self.pnl)

                    print(wm_pos)

            wm_image = plot_to_image(fig)

            with self.weights_file_writer.as_default():
                tf.summary.image('Weight Matrix', wm_image, step=1)

        return


class testConfusionMetric(keras.callbacks.Callback):
    def __init__(self,
                 args,
                 test,
                 batchSize,
                 steps,
                 file_writer,
                 weights_file_writer):

        super(testConfusionMetric, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps
        self.accBagSize = args.train_args['accBagSize']
        self.MIL = args.train_args['seperate_MIL']
        self.random_position = args.train_args['random_position']

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
        self.weights_file_writer = weights_file_writer

        if args.train_args['positions'] == None:

            if args.data_args['dataset'] == 'CompleteUser1':
                self.pnl = ['Hips']

            else:
                self.pnl = ['Torso', 'Hips', 'Bag', 'Hand']

        else:
            self.pnl = args.train_args['positions']

    def on_test_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))

        if self.MIL and self.random_position:
            self.weighting = keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("weight_layer").output
            )

            self.positional = keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("positional").output
            )

            weights = np.zeros((total, self.accBagSize))
            positions = np.zeros((total, self.accBagSize))

        for batch in self.test.take(self.steps):

            test_data = batch[0]
            test_target = batch[1]

            pred = self.model.call(test_data)

            test_predict[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(np.asarray(pred), axis=1)
            test_true[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(test_target, axis=1)

            if self.MIL and self.random_position:
                weights[step * self.batchSize: (step + 1) * self.batchSize] = self.weighting(test_data)
                positions[step * self.batchSize: (step + 1) * self.batchSize] = self.positional(test_data)

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

        if self.MIL and self.random_position:

            fig, axs = plt.subplots(ncols=1, figsize=(12, 16))
            fig.suptitle('Weight Matrix')

            for i in range(self.accBagSize):
                wm_pred = np.concatenate([test_predict[:, np.newaxis], positions[:, [i]], weights[:, [i]]], axis=1)
                wm_pred_df = pd.DataFrame(
                    wm_pred,
                    columns=['class', 'position', 'weight']
                )

                if i==0:
                    wm_pos_sum = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                    wm_pos_sum = pd.pivot_table(wm_pos_sum, values="weight", index=["class"], columns=["position"],
                                                fill_value=0)




                    wm_pos_count = wm_pred_df.groupby(['class', 'position']).size().to_frame(name='size').reset_index()
                    wm_pos_count = pd.pivot_table(wm_pos_count, values="size", index=["class"], columns=["position"],
                                                    fill_value=0)

                else:
                    wm_pos_sum_ = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                    wm_pos_sum = wm_pos_sum.add(pd.pivot_table(wm_pos_sum_, values="weight", index=["class"], columns=["position"],
                                                fill_value=0).values)

                    wm_pos_count_ = wm_pred_df.groupby(['class', 'position']).size().to_frame(name='size').reset_index()
                    wm_pos_count = wm_pos_count.add(pd.pivot_table(wm_pos_count_, values="size", index=["class"],
                                                  columns=["position"],
                                                  fill_value=0).values)


                if i == self.accBagSize-1:
                    wm_pos = wm_pos_sum.div(wm_pos_count.values)
                    wm_pos = wm_pos.div(wm_pos.sum(axis=1), axis=0)

                    sns.heatmap(wm_pos, ax=axs, cbar=False, annot=True)
                    fig.colorbar(axs.collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)
                    axs.set_yticklabels(labels=self.class_names, rotation=45)
                    axs.set_xticklabels(labels=self.pnl)

                    print(wm_pos)

            wm_image = plot_to_image(fig)

            with self.weights_file_writer.as_default():
                tf.summary.image('Weight Matrix', wm_image, step=1)

        return





def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image,0)

    return image

def get_acc_encoder(input_shapes, args, L):

    kernel_initializer = keras.initializers.he_uniform()


    seperateMIL = args.train_args['seperate_MIL']
    dimension=args.train_args['dimension']
    shape=args.train_args['acc_shape']

    if seperateMIL:
        inputAccShape = list(input_shapes)[1:]

    else:
        inputAccShape = list(input_shapes)

    inputAcc = keras.Input( shape = inputAccShape )
    X = inputAcc

    if shape=='2D':
        _, _, channels = inputAccShape
    if shape=='1D':
        _, channels = inputAccShape


    bnLayer = keras.layers.BatchNormalization(name='accBatch1')
    X = bnLayer(X)

    if shape == '2D':
        paddingLayer = keras.layers.ZeroPadding2D(padding=(1, 1))
        conv2D = keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernel_initializer,
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
            kernel_initializer=kernel_initializer,
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

        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling2D((2, 2), strides=2)

        conv2D = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernel_initializer,
            name='accConv3'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch4')

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
            kernel_initializer=kernel_initializer,
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
            kernel_initializer=kernel_initializer,
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

        activationLayer = keras.layers.ReLU()
        poolingLayer = keras.layers.MaxPooling1D(2, strides=2)

        conv1D = keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            kernel_initializer=kernel_initializer,
            name='accConv3'
        )
        bnLayer = keras.layers.BatchNormalization(name='accBatch4')

        X = conv1D(X)
        X = bnLayer(X)
        X = activationLayer(X)
        X = poolingLayer(X)

    flattenLayer = keras.layers.Flatten()
    dropoutLayer = keras.layers.Dropout(rate=args.train_args['input_dropout'])

    X = flattenLayer(X)
    X = dropoutLayer(X)

    dnn = keras.layers.Dense(units=dimension,
                             kernel_initializer=kernel_initializer,
                             name='accDense1')

    bnLayer = keras.layers.BatchNormalization(name='accBatch5')

    activationLayer = keras.layers.ReLU()
    dropoutLayer = keras.layers.Dropout(rate=0.25)

    X = dnn(X)
    X = bnLayer(X)
    X = activationLayer(X)
    X = dropoutLayer(X)

    dnn = keras.layers.Dense(units=L,
                             kernel_initializer=kernel_initializer,
                             name='accDense2')

    bnLayer = keras.layers.BatchNormalization(name='accBatch6')
    activationLayer = keras.layers.ReLU()
    dropoutLayer = keras.layers.Dropout(rate=0.25)

    X = dnn(X)
    X = bnLayer(X)
    X = activationLayer(X)
    X = dropoutLayer(X)

    acc_encoding = X

    return keras.models.Model(inputs=inputAcc,
                              outputs=acc_encoding,
                              name='AccelerationEncoder')

def get_attention_layer(args, L, D):

    kernel_initializer = keras.initializers.glorot_uniform()
    kernel_regularizer = keras.regularizers.l2(0.01)
    gated = args.train_args['use_gated']

    encodings = keras.Input(shape = (L))
    D_layer = keras.layers.Dense(units=D,
                                 activation='tanh',
                                 kernel_initializer = kernel_initializer,
                                 kernel_regularizer = kernel_regularizer,
                                 name = 'D_layer')

    if gated:
        G_layer = keras.layers.Dense(units=D,
                                     activation='sigmoid',
                                     kernel_initializer = kernel_initializer,
                                     kernel_regularizer = kernel_regularizer,
                                     name = 'G_layer')

    K_layer = keras.layers.Dense(units=1,
                                 kernel_initializer = kernel_initializer,
                                 name = 'K_layer')

    attention_weights = D_layer(encodings)

    if gated:
        attention_weights = attention_weights * G_layer(encodings)

    attention_weights = K_layer(attention_weights)


    return keras.models.Model(inputs=encodings,
                              outputs=attention_weights,
                              name='AccelerationAttentionLayer')


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
                 L = 32, D = 12):

    batchSize = args.train_args['trainBatchSize']
    accBagSize = args.train_args['accBagSize']
    seperateMIL = args.train_args['seperate_MIL']



    accShape = input_shapes[0]
    pos_shape = input_shapes[1]

    acc_encoder = get_acc_encoder(accShape, args, L)

    if seperateMIL:
        acc_attention = get_attention_layer(args, L, D)

    classifier = get_classifier(L)

    acc_bags = keras.layers.Input(accShape)
    acc_pos = keras.layers.Input(pos_shape, name='positional')

    if seperateMIL:

        accBagSize = list(accShape)[0]
        accSize = list(accShape)[1:]
        concAccBags = tf.reshape(acc_bags, (batchSize * accBagSize, *accSize))

    else:

        concAccBags = acc_bags

    accEncodings = acc_encoder(concAccBags)

    if seperateMIL:

        acc_attention_weights = acc_attention(accEncodings)
        acc_attention_weights = tf.reshape(acc_attention_weights,
                                           [batchSize, accBagSize])

        softmax = keras.layers.Softmax(name='weight_layer')

        acc_attention_weights = tf.expand_dims(softmax(acc_attention_weights), -2)

        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])

        if batchSize == 1:
            accPooling = tf.expand_dims(tf.squeeze(tf.matmul(acc_attention_weights, accEncodings)), axis=0)
        else:
            accPooling = tf.squeeze(tf.matmul(acc_attention_weights, accEncodings))

        y_pred = classifier(accPooling)

    else:
        y_pred = classifier(accEncodings)

    return keras.models.Model([acc_bags, acc_pos], y_pred)


import tensorflow.keras.backend as K


def fit(SD,
        L = 256, D = 128,
        summary = True,
        verbose = 0,
        user_seperated = False,
        round = False):

    train, val, test = SD(accTransfer = True, user_seperated = user_seperated, round = round)

    user = SD.shl_args.train_args['test_user']
    logdir = os.path.join('logs_user' + str(user),'transfer_tensorboard')

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

    Model = create_model(input_shapes=SD.inputShape, args=SD.shl_args, L=L, D=D)

    optimizer = keras.optimizers.Adam(
        learning_rate=lr
    )

    loss_function = keras.losses.CategoricalCrossentropy()

    seperateMIL = SD.shl_args.train_args['seperate_MIL']

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[keras.metrics.categorical_accuracy]
    )

    val_metrics = valMetrics(val, SD.valBatchSize, val_steps, verbose)

    file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
    w_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_val')
    w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')

    val_cm = confusion_metric(SD.shl_args,
                              val,
                              SD.valBatchSize,
                              val_steps,
                              file_writer_val,
                              w_file_writer_val)

    save_dir = os.path.join('training','saved_models','user'+str(user))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)




    model_type = 'acceleration_classifier'
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
        epochs=SD.shl_args.train_args['accEpochs'],
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = callbacks,
        use_multiprocessing = True,
        verbose = verbose
    )

    test_metrics = testMetrics(test,SD.testBatchSize,test_steps)

    test_cm = testConfusionMetric(SD.shl_args,
                                  test,
                                  SD.testBatchSize,
                                  test_steps,
                                  file_writer_test,
                                  w_file_writer_test)

    Model.load_weights(filepath)
    Model.evaluate(test,steps=test_steps,callbacks=[test_metrics])



    return filepath