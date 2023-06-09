import copy

import numpy as np
import pandas as pd
from sklearn import tree
import sklearn
from hmmlearn import hmm
from dataset import Dataset
from main import config_edit
import os

regenerateData = False
regenerateCSV = True
motorized = False
dT_threshold = 3600
useVal = True

train_position = None
for test_position in ['Hand', 'Torso', 'Bag', 'Hips']:
    scores = pd.DataFrame()
    for user in [1, 2, 3]:

        train_position = train_position if train_position == 'all' else test_position
        config_edit('train_args', 'train_position', train_position)
        config_edit('train_args', 'test_position', test_position)
        config_edit('train_args', 'test_user', user)
        print('USER: ' + str(user))

        if regenerateCSV:
            filepath = 'not-motorized/' if not motorized else 'motorized/'
            filepath += 'train_position-' + train_position + '-test-position-' + test_position

            data = Dataset(regenerate=regenerateData)
            data.initialize()
            train, val, test, train_val = data.toCSVs(filepath=filepath, motorized=motorized)

        else:
            filepath = 'not-motorized/' if not motorized else 'motorized/'
            filepath += 'train_position-' + train_position + '-test-position-' + test_position

            train = pd.read_csv(os.path.join(filepath, 'train' + str(user) + '.csv'))
            val = pd.read_csv(os.path.join(filepath, 'val' + str(user) + '.csv'))
            test = pd.read_csv(os.path.join(filepath, 'test' + str(user) + '.csv'))
            train_val = pd.read_csv(os.path.join(filepath, 'train_val' + str(user) + '.csv'))

        n_classes = 5 if motorized else 8
        classes = [i for i in range(n_classes)]

        if useVal:
            trainX = train[['vel', 'var', '1Hz', '2Hz', '3Hz']]
            trainY = train[['Label']]
            valX = val[['vel', 'var', '1Hz', '2Hz', '3Hz']]
            valY = val[['Label']]

        else:
            trainX = train_val[['vel', 'var', '1Hz', '2Hz', '3Hz']]
            trainY = train_val[['Label']]

        testX = test[['vel', 'var', '1Hz', '2Hz', '3Hz']]
        testY = test[['Label']]

        clf = tree.DecisionTreeClassifier(max_depth=11)
        clf.fit(trainX, trainY)

        testY_ = clf.predict(testX)

        acc = sklearn.metrics.accuracy_score(testY, testY_)
        f1 = sklearn.metrics.f1_score(testY, testY_, average='macro')

        if useVal:
            valY_ = clf.predict(valX)
            conf_mx = sklearn.metrics.confusion_matrix(valY_, valY, normalize='true')

        else:
            conf_mx = sklearn.metrics.confusion_matrix(testY_, testY, normalize='true')

        train_val['dT'] = train_val['Time'].diff().abs()
        split = train_val.index[train_val['dT'] > dT_threshold * 1000].tolist()
        split.append(len(train_val))

        begin = 0
        ySeqs = []
        for end in split:
            if train_position == 'all':
                ySeqs.append(train_val.iloc[begin:end:4][['Label']])
            else:
                ySeqs.append(train_val.iloc[begin:end][['Label']])

            begin = end

        transition_mx = None
        for i, seq in enumerate(ySeqs):
            seq_ = copy.copy(seq)
            seq_['label_'] = seq_.shift(-1)

            groups = seq_.groupby(['Label', 'label_'])

            counts = {g[0]: len(g[1]) for g in groups}

            matrix = pd.DataFrame()

            for x in classes:
                matrix[x] = pd.Series([counts.get((x, y), 0) for y in classes], index=classes)

            if i != 0:
                transition_mx = transition_mx.add(matrix)

            else:
                transition_mx = matrix

        transition_mx["sum"] = transition_mx.sum(axis=1)
        transition_mx = transition_mx.div(transition_mx["sum"], axis=0)
        transition_mx = transition_mx.drop(columns=['sum'])
        transition_mx = transition_mx.values.tolist()

        test['dT'] = test['Time'].diff().abs()
        split = test.index[test['dT'] > dT_threshold * 1000].tolist()
        split.append(len(test))

        begin = 0
        y_Seqs = []
        ySeqs = []

        for end in split:
            if test_position == 'all':
                y_Seqs.extend([test.iloc[begin + j:end:4][['vel', 'var', '1Hz', '2Hz', '3Hz']] for j in range(4)])
                ySeqs.extend([test.iloc[begin + j:end:4][['Label']] for j in range(4)])
            else:
                y_Seqs.append(test.iloc[begin:end][['vel', 'var', '1Hz', '2Hz', '3Hz']])
                ySeqs.append(test.iloc[begin:end][['Label']])

            begin = end

        X = []
        Y = []
        lens = []
        for seqY_, seqY in zip(y_Seqs, ySeqs):
            X.extend(clf.predict(seqY_))
            lens.append(len(seqY_))
            Y.extend(seqY['Label'].to_list())

        X = np.reshape(X, (-1, 1))

        discrete_model = hmm.MultinomialHMM(n_components=n_classes,
                                            algorithm='viterbi',
                                            n_iter=300,
                                            init_params='')

        discrete_model.n_features = n_classes
        discrete_model.startprob_ = [1. / n_classes for _ in range(n_classes)]
        discrete_model.transmat_ = transition_mx
        discrete_model.emissionprob_ = conf_mx

        discrete_model.fit(X, lens)
        Y_ = discrete_model.predict(X, lens)

        postAcc = sklearn.metrics.accuracy_score(Y, Y_)
        postF1 = sklearn.metrics.f1_score(Y, Y_, average='macro')

        print('score without post-processing: {}'.format(acc))
        print('F1 score without post-processing: {}'.format(f1))
        print('score with post-processing: {}'.format(postAcc))
        print('F1 score with post-processing: {}'.format(postF1))
        print()

        theseScores = {'Test User': str(user),
                       'Accuracy': acc,
                       'F1-Score': f1,
                       'post-Accuracy': postAcc,
                       'post-F1-Score': postF1}

        scores = scores.append(theseScores, ignore_index=True)

    meanPerUser = scores.groupby(['Test User']).mean()
    meanPerUser.columns = [str(col) + '_mean' for col in meanPerUser.columns]
    stats = meanPerUser
    stats.loc['All'] = stats.mean()
    stats['Test User'] = stats.index
    print(stats)

    scoresFile = os.path.join(filepath, "scores.csv")
    statsFile = os.path.join(filepath, "stats.csv")
    scores.to_csv(scoresFile, index=False)
    stats.to_csv(statsFile, index=False)
