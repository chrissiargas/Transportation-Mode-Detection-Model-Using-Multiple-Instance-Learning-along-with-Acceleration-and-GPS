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
import transfer

import tensorflow.keras.backend as K

from hmmlearn import hmm
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense
import transfer
import locTransfer

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


def MIL_fit(SignalsDataset,
            evaluation = False,
            summary = False,
            verbose = 0,
            load = False):

    decisionTree = SignalsDataset.shl_args.train_args['decision_tree']

    if decisionTree:

        SignalsDataset(randomTree=True,DHMM=False)
    #
    #     depths = []
    #     for i in range(3, 20):
    #         clf = tree.DecisionTreeClassifier(max_depth=i)
    #         clf = clf.fit(trainX, trainY)
    #         depths.append((i, clf.score(valX, valY)))
    #         # print(clf.score(testX, testY))
    #
    #     max_score = 0
    #     for depth in depths:
    #         if depth[1]>max_score:
    #             best_depth = depth[0]
    #             max_score = depth[1]
    #
    #     # train_X = pd.concat([trainX,valX])
    #     # train_Y = pd.concat([trainY,valY])
    #
    #     clf = tree.DecisionTreeClassifier(max_depth=best_depth)
    #     clf.fit(trainX,trainY)
    #     print(clf.score(testX,testY))
    #
    #
    #     trainX, trainY, testX, testY, trans_mx = SignalsDataset(randomTree=True, DHMM=True)
    #
    #     valY_ = clf.predict(valX)
    #
    #     conf_mx = sklearn.metrics.confusion_matrix(valY_,valY)
    #
    #     discrete_model = hmm.MultinomialHMM(n_components=5,
    #                        algorithm='viterbi',  # decoder algorithm.
    #                        random_state=93,
    #                        n_iter=10
    #                        )
    #
    #     print(trans_mx)
    #     print(conf_mx)
    #
    #     discrete_model.startprob_ = [1./5. for _ in range(5)]
    #     discrete_model.transmat_ = trans_mx
    #     discrete_model.emissionprob_ = conf_mx
    #
    #
    #
    #     X = []
    #     Y = []
    #     lengths = []
    #     for seq,seq_y in zip(testX,testY):
    #         X.extend(clf.predict_proba(seq))
    #         lengths.append(len(seq))
    #         Y.extend(seq_y['label'].to_list())
    #
    #
    #     Y_ = discrete_model.predict(X,lengths = lengths)
    #     score = sklearn.metrics.accuracy_score(Y,Y_)
    #
    #     print(score)

    else:
        L = 256
        D = 128



        if SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'load' or \
            (load and SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'train'):



            train, val, test = SignalsDataset(locTransfer=True, user_seperated=False, round=True)
            user = SignalsDataset.shl_args.train_args['test_user']

            save_dir = os.path.join('training', 'saved_models', 'user' + str(user))
            if not os.path.isdir(save_dir):
                return

            model_type = 'location_classifier'
            model_name = 'shl_%s_model.h5' % model_type
            filepath = os.path.join(save_dir, model_name)

            locTransferModel = locTransfer.create_model(SignalsDataset.inputShape,
                                                  SignalsDataset.shl_args,
                                                  L)

            lr = SignalsDataset.shl_args.train_args['learning_rate']

            optimizer = keras.optimizers.Adam(
                learning_rate=lr
            )

            loss_function = keras.losses.CategoricalCrossentropy()


            locTransferModel.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[keras.metrics.categorical_accuracy]
            )

            locTransferModel.load_weights(filepath)

            if evaluation:
                user = SignalsDataset.shl_args.train_args['test_user']
                logdir = os.path.join('logs_user' + str(user), 'loc_transfer_tensorboard')

                try:
                    shutil.rmtree(logdir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                test_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize

                test_metrics = locTransfer.testMetrics(test, SignalsDataset.testBatchSize, test_steps, verbose)

                file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')

                test_cm = locTransfer.testConfusionMetric(test,
                                                       SignalsDataset.testBatchSize,
                                                       test_steps,
                                                       file_writer_test)

                locTransferModel.evaluate(test, steps=test_steps, callbacks=[test_metrics, test_cm])

        elif SignalsDataset.shl_args.train_args['transfer_learning_loc'] == 'train':

            filepath = locTransfer.fit(
                L=L,
                summary=summary,
                verbose=verbose,
                user_seperated=False,
                round=True
            )

            SignalsDataset(locTransfer=True, user_seperated=False, round=True)

            locTransferModel = locTransfer.create_model(SignalsDataset.inputShape,
                                                  SignalsDataset.shl_args,
                                                  L)

            locTransferModel.load_weights(filepath)



        else:

            locTransferModel = None



        if SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'load' or \
                (load and SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'train'):



            train, val, test = SignalsDataset(accTransfer=True, user_seperated=False, round=True)
            user = SignalsDataset.shl_args.train_args['test_user']

            save_dir = os.path.join('training', 'saved_models', 'user' + str(user))
            if not os.path.isdir(save_dir):
                return

            model_type = 'acceleration_classifier'
            model_name = 'shl_%s_model.h5' % model_type
            filepath = os.path.join(save_dir, model_name)

            transferModel = transfer.create_model(SignalsDataset.inputShape,
                                                  SignalsDataset.shl_args,
                                                  L, D)

            lr = SignalsDataset.shl_args.train_args['learning_rate']

            optimizer = keras.optimizers.Adam(
                learning_rate=lr
            )

            loss_function = keras.losses.CategoricalCrossentropy()


            transferModel.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[keras.metrics.categorical_accuracy]
            )

            transferModel.load_weights(filepath)

            if evaluation:
                user = SignalsDataset.shl_args.train_args['test_user']
                logdir = os.path.join('logs_user' + str(user), 'transfer_tensorboard')

                try:
                    shutil.rmtree(logdir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                test_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize

                test_metrics = transfer.testMetrics(test, SignalsDataset.testBatchSize, test_steps, verbose)

                file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
                w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')

                test_cm = transfer.testConfusionMetric(SignalsDataset.shl_args, test,
                                                       SignalsDataset.testBatchSize,
                                                       test_steps,
                                                       file_writer_test,
                                                       w_file_writer_test)

                transferModel.evaluate(test, steps=test_steps, callbacks=[test_metrics, test_cm])


        elif SignalsDataset.shl_args.train_args['transfer_learning_acc'] == 'train':

            filepath = transfer.fit(
                SD = SignalsDataset,
                L=L, D=D,
                summary=summary,
                verbose=verbose,
                user_seperated=False,
                round=True
            )

            SignalsDataset(accTransfer=True, user_seperated=False, round=True)

            transferModel = transfer.create_model(SignalsDataset.inputShape,
                                                  SignalsDataset.shl_args,
                                                  L, D)

            transferModel.load_weights(filepath)



        else:

            transferModel = None



        if load:


            train, val, test = SignalsDataset(user_seperated=False, round=True)
            user = SignalsDataset.shl_args.train_args['test_user']

            save_dir = os.path.join('training', 'saved_models', 'user' + str(user))
            if not os.path.isdir(save_dir):
                return

            model_type = 'MILattention'
            model_name = 'shl_%s_model.h5' % model_type
            filepath = os.path.join(save_dir, model_name)

            test_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize

            Model = create_model(SignalsDataset.inputShape,
                                 SignalsDataset.shl_args,
                                 L, D, transferModel, locTransferModel)


            lr = SignalsDataset.shl_args.train_args['learning_rate']

            optimizer = keras.optimizers.Adam(
                learning_rate=lr
            )

            loss_function = keras.losses.CategoricalCrossentropy()

            Model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[keras.metrics.categorical_accuracy]
            )

            Model.load_weights(filepath)

            test_metrics = testMetrics(test, SignalsDataset.testBatchSize, test_steps, verbose)

            user = SignalsDataset.shl_args.train_args['test_user']
            logdir = os.path.join('logs_user' + str(user), 'MIL_tensorboard')

            file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
            w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')
            w_pos_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_pos_test')

            test_cm = testConfusionMetric(SignalsDataset.shl_args,
                                          test,
                                          SignalsDataset.testBatchSize,
                                          test_steps,
                                          file_writer_test,
                                          w_file_writer_test,
                                          w_pos_file_writer_test)

            callbacks = [test_metrics, test_cm]




        else:

            train, val, test = SignalsDataset(user_seperated=False, round=True)

            user = SignalsDataset.shl_args.train_args['test_user']
            logdir = os.path.join('logs_user' + str(user), 'MIL_tensorboard')

            if not os.path.isdir(logdir):
                os.makedirs(logdir)

            try:
                shutil.rmtree(logdir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

            val_steps = SignalsDataset.valSize // SignalsDataset.valBatchSize

            train_steps = SignalsDataset.trainSize // SignalsDataset.trainBatchSize

            test_steps = SignalsDataset.testSize // SignalsDataset.testBatchSize

            lr = SignalsDataset.shl_args.train_args['learning_rate']

            optimizer = keras.optimizers.Adam(
                learning_rate = lr
            )

            loss_function = keras.losses.CategoricalCrossentropy()

            Model = create_model(SignalsDataset.inputShape,
                                 SignalsDataset.shl_args,
                                 L, D, transferModel,
                                 locTransferModel)


            Model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[keras.metrics.categorical_accuracy]
            )

            val_metrics = valMetrics(val,
                                     SignalsDataset.valBatchSize,
                                     val_steps, verbose = verbose)

            file_writer_val = tf.summary.create_file_writer(logdir + '/cm_val')
            file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
            w_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_val')
            w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')
            w_pos_file_writer_val = tf.summary.create_file_writer(logdir + '/wm_pos_val')
            w_pos_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_pos_test')

            val_cm = confusion_metric(SignalsDataset.shl_args,
                                      val,
                                      SignalsDataset.valBatchSize,
                                      val_steps,
                                      file_writer_val,
                                      w_file_writer_val,
                                      w_pos_file_writer_val)

            save_dir = os.path.join('training', 'saved_models', 'user' + str(user))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            model_type = 'MILattention'
            model_name = 'shl_%s_model.h5' % model_type
            filepath = os.path.join(save_dir, model_name)

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
                epochs=SignalsDataset.shl_args.train_args['epochs'],
                steps_per_epoch=train_steps,
                validation_data=val,
                validation_steps=val_steps,
                callbacks=callbacks,
                use_multiprocessing=True,
                verbose=verbose
            )

            Model.load_weights(filepath)

            test_metrics = testMetrics(test, SignalsDataset.testBatchSize, test_steps)


            test_cm = testConfusionMetric(SignalsDataset.shl_args,
                                          test,
                                          SignalsDataset.testBatchSize,
                                          test_steps,
                                          file_writer_test,
                                          w_file_writer_test,
                                          w_pos_file_writer_test)

            callbacks = [test_metrics]

            Model.evaluate(test, steps=test_steps, callbacks=callbacks)

        if  SignalsDataset.shl_args.train_args['post']:

            conf_mx_tot = [[8.48631047e-01,1.75435610e-02,0.00000000e+00,3.22697445e-03,
                          2.56860581e-02,1.80804887e-02,1.81497017e-02,6.86821689e-02],
                        [1.41643127e-02,8.71545134e-01,2.86630531e-02,6.28925532e-02,
                          3.86783078e-03,8.70140785e-03,1.78889959e-03,8.37680879e-03],
                        [1.57232704e-03,1.44608991e-01,8.43522672e-01,1.02960103e-02,
                          0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                        [2.89856920e-02,6.03752665e-02,1.23448290e-02,7.66177898e-01,
                          5.90714709e-02,5.89323796e-02,1.52207002e-03,1.25903938e-02],
                        [1.08139069e-02,3.53355725e-03,0.00000000e+00,1.64567900e-02,
                          7.90479160e-01,1.12533573e-01,2.91910299e-02,3.69919829e-02],
                        [2.99581081e-02,2.47951302e-02,7.85052599e-03,1.29915467e-01,
                          1.33447974e-01,6.10014364e-01,2.66983723e-02,3.73200590e-02],
                        [1.65589794e-02,3.29666042e-04,0.00000000e+00,0.00000000e+00,
                          6.90901480e-02,1.84047285e-02,8.13753912e-01,8.18625662e-02],
                        [6.46488147e-02,1.12098650e-02,0.00000000e+00,5.65690850e-04,
                          3.90896361e-02,1.22412406e-02,1.09662977e-01,7.62581776e-01]]


            trans_mx_tot =  [[0.9   	,0.08318829,0.    	,0.01092641,0.    	,0.    	,
                              0.    	,0.0058853 ],
                            [0.02234441,0.9   	,0.01632224,0.00610274,0.    	,0.02975425,
                              0.01574926,0.00972709],
                            [0.    	,0.07505669,0.9   	,0.    	,0.    	,0.02494331,
                              0.    	,0.    	],
                            [0.02913679,0.07086321,0.    	,0.9   	,0.    	,0.    	,
                              0.    	,0.    	],
                            [0.05  	,0.05  	,0.    	,0. 	   ,0.9   	,0.    	,
                              0.    	,0.    	],
                            [0.05446786,0.04553214,0.    	,0.    	,0.    	,0.9   	,
                              0.    	,0.    	],
                            [0.1   	,0.    	,0.    	,0.    	,0.    	,0.    	,
                              0.9   	,0.    	],
                            [0.06012513,0.03987487,0.    	,0.    	,0.    	,0.    	,
                              0.    	,0.9   	]]


            if user == 3:
                conf_mx = [[8.43283582e-01, 1.20824449e-02, 0.00000000e+00, 2.48756219e-03,
                            3.55366027e-02, 1.67022033e-02, 1.45700071e-02, 7.53375977e-02],
                           [2.80612245e-02, 8.58737245e-01, 1.59438776e-02, 7.23852041e-02,
                            3.18877551e-04, 7.33418367e-03, 1.27551020e-03, 1.59438776e-02],
                           [0.00000000e+00, 2.04081633e-02, 9.79591837e-01, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [2.48447205e-02, 9.13930790e-02, 5.32386868e-03, 8.54480923e-01,
                            8.87311446e-04, 2.04081633e-02, 0.00000000e+00, 2.66193434e-03],
                           [2.78298100e-02, 8.29542414e-03, 0.00000000e+00, 2.54214611e-02,
                            7.73615199e-01, 5.96735349e-02, 3.74632058e-02, 6.77013647e-02],
                           [3.33625988e-02, 2.67778753e-02, 0.00000000e+00, 1.98419666e-01,
                            5.75065847e-02, 6.60667252e-01, 3.95083406e-03, 1.93151888e-02],
                           [1.20481928e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            6.62650602e-02, 0.00000000e+00, 7.55792400e-01, 1.65894347e-01],
                           [2.41832838e-02, 2.41832838e-02, 0.00000000e+00, 1.69707255e-03,
                            4.45481544e-02, 8.06109461e-03, 1.77768350e-01, 7.19558761e-01]]

                trans_mx = [[0.97633136, 0.01923077, 0.        , 0.00295858, 0.        ,
                            0.        , 0.        , 0.00147929],
                           [0.0069541 , 0.96244784, 0.00556328, 0.00278164, 0.        ,
                            0.01251739, 0.00556328, 0.00417246],
                           [0.        , 0.02777778, 0.95833333, 0.        , 0.        ,
                            0.01388889, 0.        , 0.        ],
                           [0.00232558, 0.00465116, 0.        , 0.99302326, 0.        ,
                            0.        , 0.        , 0.        ],
                           [0.00120337, 0.00120337, 0.        , 0.        , 0.99759326,
                            0.        , 0.        , 0.        ],
                           [0.00888889, 0.00444444, 0.        , 0.        , 0.        ,
                            0.98666667, 0.        , 0.        ],
                           [0.0018018 , 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.9981982 , 0.        ],
                           [0.00462963, 0.00308642, 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.99228395]]

            elif user == 2:
                conf_mx = [[8.43706294e-01, 4.19580420e-03, 0.00000000e+00, 3.49650350e-03,
                            9.79020979e-03, 2.55244755e-02, 2.44755245e-02, 8.88111888e-02],
                           [5.21512386e-03, 8.70491091e-01, 3.56366797e-02, 6.34506736e-02,
                            3.91134289e-03, 1.47761843e-02, 3.47674924e-03, 3.04215558e-03],
                           [4.71698113e-03, 3.20754717e-01, 6.74528302e-01, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [4.56621005e-03, 3.65296804e-02, 4.56621005e-03, 6.43835616e-01,
                            1.64383562e-01, 1.11872146e-01, 4.56621005e-03, 2.96803653e-02],
                           [3.72538584e-03, 5.32197978e-04, 0.00000000e+00, 2.39489090e-02,
                            7.39222991e-01, 1.69771155e-01, 3.99148483e-02, 2.28845130e-02],
                           [1.74281677e-02, 2.40226095e-02, 2.35515780e-02, 8.14884597e-02,
                            1.56853509e-01, 6.41073952e-01, 4.85162506e-02, 7.06547339e-03],
                           [2.20037453e-02, 4.68164794e-04, 0.00000000e+00, 0.00000000e+00,
                            7.06928839e-02, 3.69850187e-02, 8.32865169e-01, 3.69850187e-02],
                           [8.96070510e-02, 7.34484025e-03, 0.00000000e+00, 0.00000000e+00,
                            3.67242012e-03, 1.06500184e-02, 5.54535439e-02, 8.33272126e-01]]

                trans_mx = [[0.99116348, 0.00736377, 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.00147275],
                           [0.00365631, 0.98171846, 0.00365631, 0.        , 0.        ,
                            0.00731261, 0.00182815, 0.00182815],
                           [0.        , 0.00961538, 0.98076923, 0.        , 0.        ,
                            0.00961538, 0.        , 0.        ],
                           [0.        , 0.00628931, 0.        , 0.99371069, 0.        ,
                            0.        , 0.        , 0.        ],
                           [0.        , 0.        , 0.        , 0.        , 1.        ,
                            0.        , 0.        , 0.        ],
                           [0.00420168, 0.00210084, 0.        , 0.        , 0.        ,
                            0.99369748, 0.        , 0.        ],
                           [0.00368324, 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.99631676, 0.        ],
                           [0.00302572, 0.00151286, 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.99546142]]

            elif user == 1:
                conf_mx = [[8.58903266e-01, 3.63524338e-02, 0.00000000e+00, 3.69685767e-03,
                            3.17313617e-02, 1.20147874e-02, 1.54035736e-02, 4.18977203e-02],
                           [9.21658986e-03, 8.85407066e-01, 3.44086022e-02, 5.28417819e-02,
                            7.37327189e-03, 3.99385561e-03, 6.14439324e-04, 6.14439324e-03],
                           [0.00000000e+00, 9.26640927e-02, 8.76447876e-01, 3.08880309e-02,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [5.75461455e-02, 5.32030402e-02, 2.71444083e-02, 8.00217155e-01,
                            1.19435396e-02, 4.45168295e-02, 0.00000000e+00, 5.42888165e-03],
                           [8.86524823e-04, 1.77304965e-03, 0.00000000e+00, 0.00000000e+00,
                            8.58599291e-01, 1.08156028e-01, 1.01950355e-02, 2.03900709e-02],
                           [3.90835580e-02, 2.35849057e-02, 0.00000000e+00, 1.09838275e-01,
                            1.85983827e-01, 5.28301887e-01, 2.76280323e-02, 8.55795148e-02],
                           [1.56250000e-02, 5.20833333e-04, 0.00000000e+00, 0.00000000e+00,
                            7.03125000e-02, 1.82291667e-02, 8.52604167e-01, 4.27083333e-02],
                           [8.01561093e-02, 2.10147103e-03, 0.00000000e+00, 0.00000000e+00,
                            6.90483338e-02, 1.80126088e-02, 9.57670369e-02, 7.34914440e-01]]

                trans_mx = [[0.98234552, 0.01513241, 0.        , 0.00252207, 0.        ,
                            0.        , 0.        , 0.        ],
                           [0.00923483, 0.96701847, 0.00527704, 0.00263852, 0.        ,
                            0.00659631, 0.00659631, 0.00263852],
                           [0.        , 0.03333333, 0.96666667, 0.        , 0.        ,
                            0.        , 0.        , 0.        ],
                           [0.00369004, 0.00369004, 0.        , 0.99261993, 0.        ,
                            0.        , 0.        , 0.        ],
                           [0.00148368, 0.00148368, 0.        , 0.        , 0.99703264,
                            0.        , 0.        , 0.        ],
                           [0.00662252, 0.00993377, 0.        , 0.        , 0.        ,
                            0.98344371, 0.        , 0.        ],
                           [0.00583658, 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.99416342, 0.        ],
                           [0.00141844, 0.00141844, 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.99716312]]

            startprob = [1. / 8 for _ in range(8)]

            postprocessing_model = hmm.MultinomialHMM(n_components=8,
                                                      algorithm='viterbi',
                                                      random_state=93,
                                                      n_iter=100
                                                      )

            postprocessing_model.startprob_ = startprob
            postprocessing_model.transmat_ = trans_mx
            postprocessing_model.emissionprob_ = conf_mx

            # print(trans_mx)
            # print(conf_mx)

            x, y, lengths = SignalsDataset.postprocess(Model=Model, fit=False)

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
            postprocessing_model.transmat_ = trans_mx_tot
            postprocessing_model.emissionprob_ = conf_mx_tot

            # print(trans_mx_tot)
            # print(conf_mx_tot)

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

        del SignalsDataset.acceleration
        del SignalsDataset.location
        del SignalsDataset.labels

        return