import tensorflow as tf
import os
import accEncoder
import gpsEncoder
from dataset import Dataset
import TMD
from myMetrics import testMetrics, testTables
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import postprocess


def evaluate(data: Dataset, verbose=0, postprocessing=True):
    data.initialize()

    accG = 1.
    f1G = 1.
    if data.gpsMode in ['load', 'train']:

        data(gpsTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'gpsEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        gpsNetwork = gpsEncoder.build(data.inputShape, data.shl_args)

        optimizer = Adam(learning_rate=data.lr)

        loss_function = CategoricalCrossentropy()

        gpsNetwork.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[categorical_accuracy]
        )

        gpsNetwork.load_weights(filepath)

    else:

        gpsNetwork = None

    accA = 1.
    f1A = 1.
    cmA = 1.

    if data.accMode in ['load', 'train']:

        data(accTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'accEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        accNetwork = accEncoder.build(data.inputShape, data.shl_args)

        optimizer = Adam(learning_rate=data.lr)

        loss_function = CategoricalCrossentropy()

        accNetwork.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[categorical_accuracy]
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

    Model = TMD.build(data.inputShape, data.shl_args, accNetwork, gpsNetwork)

    optimizer = Adam(learning_rate=data.lr)

    loss_function = CategoricalCrossentropy()

    Model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[categorical_accuracy]
    )

    Model.load_weights(filepath)

    test_metrics = testMetrics(test, data.testBatchSize, test_steps, verbose)

    logdir = os.path.join('logs_user' + str(data.testUser), 'fullModelTb')

    file_writer_test = tf.summary.create_file_writer(logdir + '/cm_test')
    w_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_test')
    w_pos_file_writer_test = tf.summary.create_file_writer(logdir + '/wm_pos_test')

    test_cm = testTables(data.shl_args,
                         test,
                         data.testBatchSize,
                         test_steps,
                         file_writer_test,
                         w_file_writer_test,
                         w_pos_file_writer_test)

    callbacks = [test_metrics, test_cm]

    # Model.evaluate(test, steps=test_steps, callbacks=callbacks)

    train_dataset, _, _ = postprocess.get_dataset(data=data, Model=Model, train=True)
    test_dataset, y, y_ = postprocess.get_dataset(data=data, Model=Model, train=False)

    accuracy = accuracy_score(y, y_)
    f1 = f1_score(y, y_, average='macro')

    modes = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    cm = confusion_matrix(y, y_)
    cm_df = pd.DataFrame(cm, index=['{:}'.format(x) for x in modes],
                         columns=['{:}'.format(x) for x in modes])
    cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Accuracy without post-processing: {}'.format(accuracy))
    print('F1-Score without post-processing: {}'.format(f1))

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
