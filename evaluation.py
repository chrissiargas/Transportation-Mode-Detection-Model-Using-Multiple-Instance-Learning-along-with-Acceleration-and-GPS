import tensorflow as tf
import os
import accEncoder
import gpsEncoder
from dataset import Dataset
import TMD
from myMetrics import testMetrics, testTables
from hmmParams import hmmParams
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from hmmlearn.hmm import MultinomialHMM
from sklearn.metrics import f1_score, accuracy_score


def evaluate(data: Dataset, verbose=0):
    L = 256
    D = 128

    if data.gpsMode in ['load', 'train']:

        data(gpsTransfer=True)

        save_dir = os.path.join('training', 'saved_models', 'user' + str(data.testUser))
        if not os.path.isdir(save_dir):
            return

        model_type = 'gpsEncoder'
        model_name = 'TMD_%s_model.h5' % model_type
        filepath = os.path.join(save_dir, model_name)

        gpsNetwork = gpsEncoder.build(data.inputShape, data.shl_args, L)

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

    if data.accMode in ['load', 'train']:

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

    Model = TMD.build(data.inputShape, data.shl_args, L, D, accNetwork, gpsNetwork)

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

    Model.evaluate(test, steps=test_steps, callbacks=callbacks)

    if data.postprocessing:
        params = hmmParams()
        conf, trans = params(data.complete, data.testUser)

        startprob = [1. / 8 for _ in range(8)]

        HMM = MultinomialHMM(n_components=8,
                             algorithm='viterbi',
                             random_state=93,
                             n_iter=100
                             )

        HMM.startprob_ = startprob
        HMM.transmat_ = trans
        HMM.emissionprob_ = conf

        x, y, lengths = data.postprocess(Model=Model)

        y_ = HMM.predict(x, lengths)
        accuracy = accuracy_score(y, y_)
        f1 = f1_score(y, y_, average='macro')

        print()
        print('Accuracy with post-processing: {}'.format(accuracy))
        print('F1-Score with post-processing: {}'.format(f1))

        accuracy = accuracy_score(y, x)
        f1 = f1_score(y, x, average='macro')

        print('Accuracy without post-processing: {}'.format(accuracy))
        print('F1-Score without post-processing: {}'.format(f1))

    del data.acceleration
    del data.location
    del data.labels
    del data

    return
