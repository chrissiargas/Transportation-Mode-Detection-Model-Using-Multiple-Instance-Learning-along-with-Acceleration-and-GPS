import io
import itertools
import shutil
import matplotlib
import sklearn.metrics

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import tree
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Bidirectional, TimeDistributed
import accEncoder

import tensorflow.keras.backend as K

from hmmlearn import hmm
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense

import accEncoder
import gpsEncoder
from dataset import Dataset



class MaskRelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)


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
                 weights_file_writer,
                 weights_pos_file_writer):
        super(confusion_metric, self).__init__()

        self.val = val
        self.batchSize = batchSize
        self.steps = steps
        self.accBagSize = args.train_args['accBagSize']
        self.locBagSize = args.train_args['locBagSize']
        self.seperateMIL = args.train_args['seperate_MIL']
        self.n_heads = args.train_args['heads']

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
        self.weights_pos_file_writer = weights_pos_file_writer
        self.random_position = args.train_args['random_position']


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

        weights = []
        weighting = []
        acc_weights = []

        if self.seperateMIL:
            for head in range(self.n_heads):
                weighting.append(keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("weight_layer_" + str(head)).output
                ))

                weights.append(np.zeros((total, 2)))

            if self.random_position:
                acc_weighting = keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer('acc_weight_layer').output
                )

                positional = keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("positional").output
                )

                acc_weights = np.zeros((total, self.accBagSize))
                positions = np.zeros((total, self.accBagSize))

        else:
            for head in range(self.n_heads):
                weighting.append(keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("weight_layer_" + str(head)).output
                ))

                weights.append(np.zeros((total, self.locBagSize + self.accBagSize)))

            if self.random_position:

                positional = keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("positional").output
                )

                positions = np.zeros((total, self.accBagSize))

                for head in range(self.n_heads):
                    acc_weights.append(np.zeros((total, self.accBagSize)))

        for batch in self.val.take(self.steps):

            val_data = batch[0]
            val_target = batch[1]

            pred = self.model.call(val_data)
            val_predict[step*self.batchSize : (step+1)*self.batchSize] = np.argmax(np.asarray(pred),axis=1)
            val_true[step * self.batchSize : (step + 1) * self.batchSize] = np.argmax(val_target,axis=1)

            for head in range(self.n_heads):
                weights[head][step * self.batchSize: (step + 1) * self.batchSize] = weighting[head](val_data)

            if self.random_position:
                if self.seperateMIL:
                    acc_weights[step * self.batchSize: (step + 1) * self.batchSize] = acc_weighting(val_data)
                    positions[step * self.batchSize: (step + 1) * self.batchSize] = positional(val_data)

                else:
                    for head in range(self.n_heads):
                        acc_weights[head][step * self.batchSize: (step + 1) * self.batchSize] = weighting[head](val_data)[:, :self.accBagSize]
                    positions[step * self.batchSize: (step + 1) * self.batchSize] = positional(val_data)

            step += 1

        val_f1 = f1_score(val_true, val_predict, average="macro")
        val_recall = recall_score(val_true, val_predict, average="macro")
        val_precision =precision_score(val_true, val_predict, average="macro")

        print(" - val_f1: %f - val_precision: %f - val_recall: %f" %(val_f1,val_precision,val_recall))


        cm = confusion_matrix(val_true,val_predict)
        global CM
        CM = cm / cm.sum(axis=1)[:, np.newaxis]

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


        fig, axs = plt.subplots(ncols=self.n_heads, figsize=(12, 16))
        fig.subplots_adjust(wspace=0.01)
        fig.suptitle('Weight Matrix')

        for head in range(self.n_heads):
            instances = ["Acceleration " + str(i + 1) for i in range(self.accBagSize)]
            wm_pred = np.concatenate([val_predict[:, np.newaxis], weights[head]], axis=1)

            if self.seperateMIL:
                wm_pred_df = pd.DataFrame(
                    wm_pred,
                    columns=['class', 'Acceleration', 'Location']
                )

            else:
                wm_pred_df = pd.DataFrame(
                    wm_pred,
                    columns=['class', *instances, 'GPS' ]
                )

                wm_pred_df['Accelerotometer'] = wm_pred_df.loc[:, instances].sum(axis=1)
                wm_pred_df.drop(instances, inplace=True, axis=1)

            wm = wm_pred_df.groupby(['class'], as_index=False).mean()
            del wm['class']

            if self.n_heads == 1:

                sns.heatmap(wm, ax=axs, cbar=False, annot=True)
                if self.seperateMIL:
                    axs.set_xticklabels(labels=['Accelerometer', 'GPS'])
                else:
                    axs.set_xticklabels(labels=['GPS', 'Accelerometer'])
                axs.set_yticklabels(labels=self.class_names,
                                    rotation=45)

                fig.colorbar(axs.collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)

            else:
                sns.heatmap(wm, ax=axs[head], cbar=False, annot=True)

                if self.seperateMIL:
                    axs[head].set_xticklabels(labels=['Accelerometer', 'GPS'])
                else:
                    axs[head].set_xticklabels(labels=['GPS', 'Accelerometer'])

                if head == 0:
                    axs[head].set_yticklabels(labels=self.class_names,
                                              rotation=45)

                if head == self.n_heads - 1:
                    fig.colorbar(axs[head].collections[0], ax=axs[head], location="right", use_gridspec=False,
                                 pad=0.2)

        wm_image = plot_to_image(fig)

        with self.weights_file_writer.as_default():
            tf.summary.image('Weight Matrix', wm_image, step=1)




        if self.random_position:

            if not self.seperateMIL:
                fig, axs = plt.subplots(ncols=self.n_heads, figsize=(12, 16))
                fig.subplots_adjust(wspace=0.01)
                fig.suptitle('Weight Matrix')

                for head in range(self.n_heads):
                    if self.n_heads == 1:

                        for i in range(self.accBagSize):
                            wm_pred = np.concatenate(
                                [val_predict[:, np.newaxis], positions[:, [i]], acc_weights[head][:, [i]]], axis=1)
                            wm_pred_df = pd.DataFrame(
                                wm_pred,
                                columns=['class', 'position', 'weight']
                            )

                            if i == 0:
                                wm_pos_sum = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = pd.pivot_table(wm_pos_sum, values="weight", index=["class"],
                                                            columns=["position"],
                                                            fill_value=0)

                                wm_pos_count = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = pd.pivot_table(wm_pos_count, values="size", index=["class"],
                                                              columns=["position"],
                                                              fill_value=0)

                            else:
                                wm_pos_sum_ = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = wm_pos_sum.add(
                                    pd.pivot_table(wm_pos_sum_, values="weight", index=["class"], columns=["position"],
                                                   fill_value=0).values)

                                wm_pos_count_ = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = wm_pos_count.add(
                                    pd.pivot_table(wm_pos_count_, values="size", index=["class"],
                                                   columns=["position"],
                                                   fill_value=0).values)

                            if i == self.accBagSize - 1:
                                wm_pos = wm_pos_sum.div(wm_pos_count.values)
                                wm_pos = wm_pos.div(wm_pos.sum(axis=1), axis=0)


                                sns.heatmap(wm_pos, ax=axs, cbar=False, annot=True)
                                fig.colorbar(axs.collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)
                                axs.set_yticklabels(labels=self.class_names, rotation=45)
                                axs.set_xticklabels(labels=self.pnl)


                    else:

                        for i in range(self.accBagSize):
                            wm_pred = np.concatenate(
                                [val_predict[:, np.newaxis], positions[:, [i]], acc_weights[head][:, [i]]], axis=1)
                            wm_pred_df = pd.DataFrame(
                                wm_pred,
                                columns=['class', 'position', 'weight']
                            )

                            if i == 0:
                                wm_pos_sum = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = pd.pivot_table(wm_pos_sum, values="weight", index=["class"],
                                                            columns=["position"],
                                                            fill_value=0)

                                wm_pos_count = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = pd.pivot_table(wm_pos_count, values="size", index=["class"],
                                                              columns=["position"],
                                                              fill_value=0)

                            else:
                                wm_pos_sum_ = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = wm_pos_sum.add(
                                    pd.pivot_table(wm_pos_sum_, values="weight", index=["class"], columns=["position"],
                                                   fill_value=0).values)

                                wm_pos_count_ = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = wm_pos_count.add(
                                    pd.pivot_table(wm_pos_count_, values="size", index=["class"],
                                                   columns=["position"],
                                                   fill_value=0).values)

                            if i == self.accBagSize - 1:
                                wm_pos = wm_pos_sum.div(wm_pos_count.values)
                                wm_pos = wm_pos.div(wm_pos.sum(axis=1), axis=0)

                                sns.heatmap(wm_pos, ax=axs[head], cbar=False, annot=True)


                                axs[head].set_xticklabels(labels=self.pnl)

                                if head == 0:
                                    axs[head].set_yticklabels(labels=self.class_names, rotation=45)

                                if head == self.n_heads - 1:
                                    fig.colorbar(axs[head].collections[0], ax=axs[head], location="right", use_gridspec=False,
                                                 pad=0.2)



            else:
                fig, axs = plt.subplots(ncols=1, figsize=(12, 16))
                fig.suptitle('Weight Matrix')

                for i in range(self.accBagSize):
                    wm_pred = np.concatenate([val_predict[:, np.newaxis], positions[:, [i]], acc_weights[:, [i]]], axis=1)
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



            wm_image = plot_to_image(fig)

            with self.weights_pos_file_writer.as_default():
                tf.summary.image('Position-Weight Matrix', wm_image, step=1)



        return



class testConfusionMetric(keras.callbacks.Callback):
    def __init__(self,                  args,
                 test,
                 batchSize,
                 steps,
                 file_writer,
                 weights_file_writer,
                 weights_pos_file_writer):
        super(testConfusionMetric, self).__init__()
        self.test = test
        self.batchSize = batchSize
        self.steps = steps
        self.accBagSize = args.train_args['accBagSize']
        self.locBagSize = args.train_args['locBagSize']
        self.seperateMIL = args.train_args['seperate_MIL']
        self.n_heads = args.train_args['heads']

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
        self.weights_pos_file_writer = weights_pos_file_writer
        self.random_position = args.train_args['random_position']


        if args.train_args['positions'] == None:

            if args.data_args['dataset'] == 'CompleteUser1':
                self.pnl = ['Hips']

            else:
                self.pnl = ['Torso','Hips','Bag','Hand']

        else:
            self.pnl = args.train_args['positions']

    def on_test_end(self, logs={}):
        total = self.batchSize * self.steps
        step = 0
        test_predict = np.zeros((total))
        test_true = np.zeros((total))

        weights = []
        weighting = []
        acc_weights = []

        if self.seperateMIL:
            for head in range(self.n_heads):
                weighting.append(keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("weight_layer_" + str(head)).output
                ))

                weights.append(np.zeros((total, 2)))

            if self.random_position:
                acc_weighting = keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer('acc_weight_layer').output
                )

                positional = keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("positional").output
                )

                acc_weights = np.zeros((total, self.accBagSize))
                positions = np.zeros((total, self.accBagSize))

        else:
            for head in range(self.n_heads):
                weighting.append(keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("weight_layer_" + str(head)).output
                ))

                weights.append(np.zeros((total, self.locBagSize + self.accBagSize)))

            if self.random_position:

                positional = keras.models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("positional").output
                )

                positions = np.zeros((total, self.accBagSize))

                for head in range(self.n_heads):
                    acc_weights.append(np.zeros((total, self.accBagSize)))


        for batch in self.test.take(self.steps):
            test_data = batch[0]
            test_target = batch[1]
            pred = self.model.call(test_data)

            test_predict[step * self.batchSize: (step + 1) * self.batchSize] = \
                np.argmax(np.asarray(pred), axis=1)
            test_true[step * self.batchSize: (step + 1) * self.batchSize] = np.argmax(test_target, axis=1)

            for head in range(self.n_heads):
                weights[head][step * self.batchSize: (step + 1) * self.batchSize] = weighting[head](test_data)

            if self.random_position:
                if self.seperateMIL:
                    acc_weights[step * self.batchSize: (step + 1) * self.batchSize] = acc_weighting(test_data)
                    positions[step * self.batchSize: (step + 1) * self.batchSize] = positional(test_data)

                else:
                    for head in range(self.n_heads):
                        acc_weights[head][step * self.batchSize: (step + 1) * self.batchSize] =  weighting[head](test_data)[:, :self.accBagSize]
                    positions[step * self.batchSize: (step + 1) * self.batchSize] = positional(test_data)

            step += 1

        test_f1 = f1_score(test_true, test_predict, average="macro")
        test_recall = recall_score(test_true, test_predict, average="macro")
        test_precision = precision_score(test_true, test_predict, average="macro")


        print(" - test_f1: %f - test_precision: %f - test_recall %f" %(test_f1,test_precision,test_recall))

        cm = confusion_matrix(test_true, test_predict)
        cm_df = pd.DataFrame(cm,
                             index=self.class_names,
                             columns=self.class_names)
        cm_df = cm_df.astype('float') / cm.sum(axis=1)[:,np.newaxis]



        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        cm_image = plot_to_image(figure)

        with self.file_writer.as_default():
            tf.summary.image('Confusion Matrix', cm_image, step=1)


        fig, axs = plt.subplots(ncols=self.n_heads, figsize=(12, 16))
        fig.subplots_adjust(wspace=0.01)
        fig.suptitle('Weight Matrix')

        for head in range(self.n_heads):
            instances = ["Acceleration " + str(i + 1) for i in range(self.accBagSize)]
            wm_pred = np.concatenate([test_predict[:, np.newaxis], weights[head]], axis=1)

            if self.seperateMIL:
                wm_pred_df = pd.DataFrame(
                    wm_pred,
                    columns=['class', 'Acceleration', 'Location']
                )

            else:
                wm_pred_df = pd.DataFrame(
                    wm_pred,
                    columns=['class', *instances, 'GPS' ]
                )

                wm_pred_df['Accelerometer'] = wm_pred_df.loc[:, instances].sum(axis=1)
                wm_pred_df.drop(instances, inplace=True, axis=1)


            wm = wm_pred_df.groupby(['class'], as_index=False).mean()
            del wm['class']

            if self.n_heads == 1:

                sns.heatmap(wm, ax=axs, cbar=False, annot=True)

                if self.seperateMIL:
                    axs.set_xticklabels(labels=['Accelerometer', 'GPS'])
                else:
                    axs.set_xticklabels(labels=['GPS', 'Accelerometer'])

                axs.set_yticklabels(labels=self.class_names,
                                    rotation=45)

                fig.colorbar(axs.collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)

            else:

                sns.heatmap(wm, ax=axs[head], cbar=False, annot=True)

                if self.seperateMIL:
                    axs[head].set_xticklabels(labels=['Accelerometer', 'GPS'])
                else:
                    axs[head].set_xticklabels(labels=['GPS', 'Accelerometer'])

                if head == 0:
                    axs[head].set_yticklabels(labels=self.class_names,
                                              rotation=45)

                if head == self.n_heads - 1:
                    fig.colorbar(axs[head].collections[0], ax=axs[head], location="right", use_gridspec=False,
                                 pad=0.2)

        wm_image = plot_to_image(fig)

        with self.weights_file_writer.as_default():
            tf.summary.image('Weight Matrix', wm_image, step=1)



        if self.random_position:

            if not self.seperateMIL:
                fig, axs = plt.subplots(ncols=self.n_heads, figsize=(12, 16))
                fig.subplots_adjust(wspace=0.01)
                fig.suptitle('Weight Matrix')

                for head in range(self.n_heads):
                    if self.n_heads == 1:

                        for i in range(self.accBagSize):
                            wm_pred = np.concatenate(
                                [test_predict[:, np.newaxis], positions[:, [i]], acc_weights[head][:, [i]]], axis=1)
                            wm_pred_df = pd.DataFrame(
                                wm_pred,
                                columns=['class', 'position', 'weight']
                            )

                            if i == 0:
                                wm_pos_sum = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = pd.pivot_table(wm_pos_sum, values="weight", index=["class"],
                                                            columns=["position"],
                                                            fill_value=0)

                                wm_pos_count = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = pd.pivot_table(wm_pos_count, values="size", index=["class"],
                                                              columns=["position"],
                                                              fill_value=0)

                            else:
                                wm_pos_sum_ = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = wm_pos_sum.add(
                                    pd.pivot_table(wm_pos_sum_, values="weight", index=["class"], columns=["position"],
                                                   fill_value=0).values)

                                wm_pos_count_ = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = wm_pos_count.add(
                                    pd.pivot_table(wm_pos_count_, values="size", index=["class"],
                                                   columns=["position"],
                                                   fill_value=0).values)

                            if i == self.accBagSize - 1:
                                wm_pos = wm_pos_sum.div(wm_pos_count.values)
                                wm_pos = wm_pos.div(wm_pos.sum(axis=1), axis=0)

                                sns.heatmap(wm_pos, ax=axs, cbar=False, annot=True)
                                fig.colorbar(axs.collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)
                                axs.set_yticklabels(labels=self.class_names, rotation=45)
                                axs.set_xticklabels(labels=self.pnl)


                    else:

                        for i in range(self.accBagSize):
                            wm_pred = np.concatenate(
                                [test_predict[:, np.newaxis], positions[:, [i]], acc_weights[head][:, [i]]], axis=1)
                            wm_pred_df = pd.DataFrame(
                                wm_pred,
                                columns=['class', 'position', 'weight']
                            )

                            if i == 0:
                                wm_pos_sum = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = pd.pivot_table(wm_pos_sum, values="weight", index=["class"],
                                                            columns=["position"],
                                                            fill_value=0)

                                wm_pos_count = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = pd.pivot_table(wm_pos_count, values="size", index=["class"],
                                                              columns=["position"],
                                                              fill_value=0)

                            else:
                                wm_pos_sum_ = wm_pred_df.groupby(['class', 'position'], as_index=False).sum()
                                wm_pos_sum = wm_pos_sum.add(
                                    pd.pivot_table(wm_pos_sum_, values="weight", index=["class"], columns=["position"],
                                                   fill_value=0).values)

                                wm_pos_count_ = wm_pred_df.groupby(['class', 'position']).size().to_frame(
                                    name='size').reset_index()
                                wm_pos_count = wm_pos_count.add(
                                    pd.pivot_table(wm_pos_count_, values="size", index=["class"],
                                                   columns=["position"],
                                                   fill_value=0).values)

                            if i == self.accBagSize - 1:
                                wm_pos = wm_pos_sum.div(wm_pos_count.values)
                                wm_pos = wm_pos.div(wm_pos.sum(axis=1), axis=0)

                                sns.heatmap(wm_pos, ax=axs[head], cbar=False, annot=True)


                                axs[head].set_xticklabels(labels=self.pnl)

                                if head == 0:
                                    axs[head].set_yticklabels(labels=self.class_names, rotation=45)

                                if head == self.n_heads - 1:
                                    fig.colorbar(axs[head].collections[0], ax=axs[head], location="right", use_gridspec=False,
                                                 pad=0.2)



            else:
                fig, axs = plt.subplots(ncols=1, figsize=(12, 16))
                fig.suptitle('Weight Matrix')

                for i in range(self.accBagSize):
                    wm_pred = np.concatenate([test_predict[:, np.newaxis], positions[:, [i]], acc_weights[:, [i]]], axis=1)
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



            wm_image = plot_to_image(fig)

            with self.weights_pos_file_writer.as_default():
                tf.summary.image('Position-Weight Matrix', wm_image, step=1)



        return




def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image,0)

    return image


def get_acc_encoder(input_shapes, args, L, transfer = False):

    kernel_initializer = keras.initializers.he_uniform()

    dimension=args.train_args['dimension']
    shape = args.train_args['acc_shape']

    inputAccShape = list(input_shapes[0])[1:]
    inputAcc = keras.Input(shape=inputAccShape)
    X = inputAcc

    if shape == '2D':
        _, _, channels = inputAccShape
    if shape == '1D':
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
    if not transfer:
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
    if not transfer:
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

def get_loc_encoder(input_shapes, args, L):
    kernel_initializer = keras.initializers.he_uniform()

    inputLocSignalsShape = list(input_shapes[1])[1:]
    inputLocFeaturesShape = list(input_shapes[2])[1:]
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


    if args.train_args['GPSNet'] == 'FCLSTM':
        TDDenseLayer = TimeDistributed(
            keras.layers.Dense(
                units=16,
                kernel_initializer=kernel_initializer,
                name='TDlocDense1'
            )
        )
        TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch1', trainable = False)
        TDactivationLayer = MaskRelu()

        X = TDDenseLayer(X)
        X = TDbnLayer(X)
        X = TDactivationLayer(X)

        TDDenseLayer = TimeDistributed(
            keras.layers.Dense(
                units=16,
                kernel_initializer=kernel_initializer,
                name='TDlocDense2'
            )
        )
        TDbnLayer = keras.layers.BatchNormalization(name='TDlocBatch2', trainable = False)
        TDactivationLayer = MaskRelu()

        X = TDDenseLayer(X)
        X = TDbnLayer(X)
        X = TDactivationLayer(X)

        lstmLayer1 = tf.keras.layers.LSTM(units=64,
                                          return_sequences=True,
                                          name='locLSTM1')
        lstmLayer2 = tf.keras.layers.LSTM(units=64,
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

def get_attention_layer(args, L, D, acc = False, head = 1):
    kernel_initializer = keras.initializers.glorot_uniform()
    kernel_regularizer = keras.regularizers.l2(0.01)
    gated = args.train_args['use_gated']

    encodings = keras.Input(shape = (L))

    D_layer = keras.layers.Dense(units=D,
                                 activation='tanh',
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 name = 'D_layer')

    if gated:
        G_layer = keras.layers.Dense(units=D,
                                     activation='sigmoid',
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     name = 'G_layer')

    K_layer = keras.layers.Dense(units=1,
                                 kernel_initializer = kernel_initializer,
                                 name = 'K_layer')

    attention_weights = D_layer(encodings)

    if gated:
        attention_weights = attention_weights * G_layer(encodings)

    attention_weights = K_layer(attention_weights)

    if not acc:
        return keras.models.Model(inputs=encodings,
                                  outputs=attention_weights,
                                  name='AttentionLayer_'+str(head))

    else:
        return keras.models.Model(inputs=encodings,
                                  outputs=attention_weights,
                                  name='AccelerationAttentionLayer')

def get_head_linear(L, head=1):
    kernel_initializer = keras.initializers.he_uniform()

    pooling = keras.Input(shape=(L))

    X = pooling

    denseLayer = keras.layers.Dense(
        units=L // 2,
        kernel_initializer=kernel_initializer
    )

    bnLayer = keras.layers.BatchNormalization(trainable=False)
    activationLayer = keras.layers.ReLU()

    X = denseLayer(X)
    X = bnLayer(X)
    output = activationLayer(X)

    return keras.models.Model(inputs=pooling,
                              outputs=output,
                              name='head_linear_' + str(head))


def get_classifier(L):
    kernel_initializer = keras.initializers.glorot_uniform()

    head = keras.Input(shape=(L//2))
    X = head

    finalLayer = keras.layers.Dense(units=8,
                                    activation='softmax',
                                    kernel_initializer=kernel_initializer)

    y_pred = finalLayer(X)

    return keras.models.Model(inputs = head,
                              outputs = y_pred,
                              name = 'Classifier')

def create_model(input_shapes, args, L, D, transferModel = None, locTransferModel = None):
    batchSize = args.train_args['trainBatchSize']
    n_heads = args.train_args['heads']
    seperateMIL = args.train_args['seperate_MIL']
    transfer = True if transferModel else False
    acc_encoder = get_acc_encoder(input_shapes, args, L, transfer)


    if seperateMIL:
        acc_attention = get_attention_layer(args, L, D, True)

    if transfer:

        acc_encoder.set_weights(transferModel.get_layer('AccelerationEncoder').get_weights())
        acc_encoder.trainable = False

        if seperateMIL:

            acc_attention.set_weights(transferModel.get_layer('AccelerationAttentionLayer').get_weights())
            acc_attention.trainable = False



    locTransfer = True if locTransferModel else False
    loc_encoder = get_loc_encoder(input_shapes, args, L)

    if locTransfer:

        loc_encoder.set_weights(locTransferModel.get_layer('LocationEncoder').get_weights())
        loc_encoder.trainable = False

    attention_layers = []
    head_linears = []
    for head in range(n_heads):
        attention_layers.append(get_attention_layer(args, L, D, head=head))
        head_linears.append(get_head_linear(L, head=head))

    classifier = get_classifier(L)


    accShape = input_shapes[0]
    accBagSize = list(accShape)[0]
    accSize = list(accShape)[1:]

    locSignalsShape = input_shapes[1]
    locSignalsBagSize = list(locSignalsShape)[0]
    locSignalsSize = list(locSignalsShape)[1:]

    locFeaturesShape = input_shapes[2]
    locFeaturesBagSize = list(locFeaturesShape)[0]
    locFeaturesSize = list(locFeaturesShape)[1:]

    posShape = input_shapes[3]

    acc_bags = keras.layers.Input(accShape)
    loc_signals_bags = keras.layers.Input(locSignalsShape)
    loc_features_bags = keras.layers.Input(locFeaturesShape)
    acc_pos = keras.layers.Input(posShape, name='positional')
    batchSize = tf.shape(acc_bags)[0]

    concAccBags = tf.reshape(acc_bags, (batchSize * accBagSize, *accSize))
    concLocSignalsBags = tf.reshape(loc_signals_bags, (batchSize * locSignalsBagSize, *locSignalsSize))
    concLocFeaturesBags = tf.reshape(loc_features_bags, (batchSize * locFeaturesBagSize, *locFeaturesSize))


    accEncodings = acc_encoder(concAccBags)
    locEncodings = loc_encoder([concLocSignalsBags, concLocFeaturesBags])

    if not seperateMIL:
        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])
        locEncodings = tf.reshape(locEncodings, [batchSize, locSignalsBagSize, L])
        Encodings = tf.keras.layers.concatenate([accEncodings, locEncodings], axis=-2)
        Encodings = tf.reshape(Encodings, (batchSize * (accBagSize + locSignalsBagSize), L))
        poolings = []

        for i, attention_layer in enumerate(attention_layers):
            attention_weights = attention_layer(Encodings)
            attention_weights = tf.reshape(attention_weights, [batchSize, accBagSize + locSignalsBagSize])
            softmax = keras.layers.Softmax(name='weight_layer_' + str(i))
            attention_weights = tf.expand_dims(softmax(attention_weights), -2)


            Encodings_ = tf.reshape(Encodings, [batchSize, accBagSize + locSignalsBagSize, L])

            Flatten = tf.keras.layers.Flatten()
            poolings.append(Flatten(tf.matmul(attention_weights, Encodings_)))




    else:
        acc_attention_weights = acc_attention(accEncodings)
        acc_attention_weights = tf.reshape(acc_attention_weights,
                                           [batchSize, accBagSize])

        softmax = keras.layers.Softmax( name='acc_weight_layer' )

        acc_attention_weights = tf.expand_dims(softmax(acc_attention_weights), -2)

        accEncodings = tf.reshape(accEncodings, [batchSize, accBagSize, L])

        accPooling = tf.matmul(acc_attention_weights, accEncodings)
        locEncodings = tf.reshape(locEncodings, [batchSize, locSignalsBagSize, L])

        Encodings = tf.keras.layers.concatenate([accPooling, locEncodings], axis=-2)

        Encodings = tf.reshape(Encodings, (batchSize * (1 + locSignalsBagSize), L))

        poolings = []

        for i, attention_layer in enumerate(attention_layers):
            attention_weights = attention_layer(Encodings)
            attention_weights = tf.reshape(attention_weights, [batchSize, 1 + locSignalsBagSize])
            softmax = keras.layers.Softmax(name='weight_layer_'+str(i))
            attention_weights = tf.expand_dims(softmax(attention_weights), -2)

            Encodings_ = tf.reshape(Encodings, [batchSize, 1 + locSignalsBagSize, L])

            Flatten = tf.keras.layers.Flatten()
            poolings.append(Flatten(tf.matmul(attention_weights, Encodings_)))



    poolings =  tf.stack(poolings)
    heads = []
    for i in range(n_heads):
        heads.append(head_linears[i](poolings[i]))

    heads = tf.transpose(tf.stack(heads), perm=(1,0,2))
    head_pooling = keras.layers.AveragePooling1D(pool_size=(n_heads),strides=1)
    Flatten = tf.keras.layers.Flatten()
    head = Flatten(head_pooling(heads))
    y_pred = classifier(head)

    return keras.models.Model([acc_bags, loc_signals_bags, loc_features_bags, acc_pos], y_pred)


def evaluate(data: Dataset,
             verbose = 0):

    L = 256
    D = 128

    if data.gpsMode in ['load','train']:

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

    else:

        gpsNetwork = None

    if data.accMode in ['load','train']:

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

    else:

        accNetwork = None

    train, val, test = data()

    save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
    if not os.path.isdir(save_dir):
        return

    model_type = 'full'
    model_name = 'TMD_%s_model.h5' % model_type
    filepath = os.path.join(save_dir, model_name)

    test_steps = data.testSize // data.testBatchSize

    Model = create_model(data.inputShape,
                         data.shl_args,
                         L, D, accNetwork, gpsNetwork)

    optimizer = keras.optimizers.Adam(
        learning_rate=data.lr
    )

    loss_function = keras.losses.CategoricalCrossentropy()

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[keras.metrics.categorical_accuracy]
    )

    Model.load_weights(filepath)

    test_metrics = testMetrics(test, data.testBatchSize, test_steps, verbose)

    logdir = os.path.join('logs_user' + str(data.testUser), 'fullModelTb')

    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
    w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')
    w_pos_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_pos_test')

    test_cm = testConfusionMetric(data.shl_args,
                                  test,
                                  data.testBatchSize,
                                  test_steps,
                                  file_writer_test,
                                  w_file_writer_test,
                                  w_pos_file_writer_test)

    callbacks = [test_metrics, test_cm]

    Model.evaluate(test, steps=test_steps, callbacks=callbacks)

    del data

    return