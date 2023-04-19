import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras
import os
from hmmlearn import hmm
import accEncoder
import gpsEncoder
from dataset import Dataset
from TMD import build, testMetrics
from myMetrics import testMetrics, testTables
from hmmParams import hmmParams


def evaluate(data: Dataset, verbose = 0):

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

    Model = build(data.inputShape,
                    data.shl_args,
                    L, D,
                    accNetwork, gpsNetwork)

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

        x, y, lengths = data.postprocess(Model=Model)

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