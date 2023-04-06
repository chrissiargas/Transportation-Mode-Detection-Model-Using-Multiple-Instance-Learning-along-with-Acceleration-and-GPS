import numpy as np
import pandas as pd
from sklearn import tree
import sklearn
from hmmlearn import hmm



motorized_class = True
dT_threshold = 60000

for user in ['1','2','3']:
    print('USER: ' + user)

    fileplace = 'not-motorized/' if not motorized_class else 'motorized/'
    train = pd.read_csv(fileplace + 'train' + user + '.csv')
    val = pd.read_csv(fileplace + 'val' + user + '.csv')
    test = pd.read_csv(fileplace + 'test' + user + '.csv')


    train['set'] = 'train'
    val['set'] = 'val'

    train_val = pd.concat([train, val], ignore_index=False)
    train_val['in'] = train_val.index
    train_val.sort_values(['User','Day','Label Time','in'], inplace=True, ignore_index=True)




    train_val['dT'] = train_val['Label Time'].diff().abs()
    split = train_val.index[train_val['dT'] > dT_threshold].tolist()

    train_val.to_csv('train_val' + user + '.csv', index=False)

    train_val = train_val[['Label Time','Position','Label','set','dT']]

    last_check = 0
    split_lbs = []
    for index in split:
        split_lbs.extend([train_val.iloc[last_check+i:index - 1:4][['Label']] for i in range(4)])
        last_check = index

    n_classes = 5 if motorized_class else 8
    classes = [i for i in range(n_classes)]

    transition_mx = None
    for i, seq in enumerate(split_lbs):
        seq_ = seq
        seq_['label_'] = seq_.shift(-1)

        groups = seq_.groupby(['Label', 'label_'])

        counts = {i[0]: len(i[1]) for i in groups}


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

    trainX = train[['vel','var','1Hz','2Hz','3Hz']]
    trainY = train[['Label']]
    valX = val[['vel','var','1Hz','2Hz','3Hz']]
    valY = val[['Label']]
    testX = test[['vel','var','1Hz','2Hz','3Hz']]
    testY = test[['Label']]

    depths = []
    for i in range(3, 20):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(trainX, trainY)
        depths.append((i, clf.score(valX, valY)))

    max_score = 0
    for depth in depths:
        if depth[1] > max_score:
            best_depth = depth[0]
            max_score = depth[1]

    clf = tree.DecisionTreeClassifier(max_depth=best_depth)
    clf.fit(trainX, trainY)
    testY_ = clf.predict(testX)
    print('score without post-processing: ' + str(sklearn.metrics.accuracy_score(testY, testY_)))
    print('F1 score without post-processing: ' + str(sklearn.metrics.f1_score(testY, testY_, average='macro')))



    valY_ = clf.predict(valX)
    conf_mx = sklearn.metrics.confusion_matrix(valY_,valY, normalize='true')


    test['dT'] = test['Label Time'].diff().abs()
    split = test.index[test['dT'] > dT_threshold].tolist()

    last_check = 0
    split_data = []
    split_lbs = []

    for index in split:

        split_data.extend([test.iloc[last_check+i:index - 1:4][['vel','var','1Hz','2Hz','3Hz']] for i in range(4)])
        split_lbs.extend([test.iloc[last_check+i:index - 1:4][['Label']] for i in range(4)])
        last_check = index


    X = []
    Y = []
    lens = []
    for seqX, seqY in zip(split_data, split_lbs):

        # X.extend(clf.predict_proba(seqX))
        X.extend(clf.predict(seqX))
        lens.append(len(seqX))
        Y.extend(seqY['Label'].to_list())

    # print(X[:10])
    # X = np.array(X)
    # X_ = np.zeros_like(X)
    # X_[np.arange(len(X)), X.argmax(1)] = 1
    # X = X_.astype(int)
    # print(X[:10])


    X = np.reshape(X, (-1,1))

    discrete_model = hmm.MultinomialHMM(n_components=n_classes,
                            algorithm='viterbi',  # decoder algorithm.
                            random_state=93,
                            n_iter=100,
                            init_params=''
                           )

    # print(len(X))
    # print(len(Y))
    discrete_model.n_features = n_classes
    discrete_model.startprob_ = [1./n_classes for _ in range(n_classes)]
    discrete_model.transmat_ = transition_mx
    discrete_model.emissionprob_ = conf_mx

    discrete_model.fit(X, lens)
    Y_ = discrete_model.predict(X, lens)
    # Y_ = np.reshape(Y_, (-1,n_classes))
    # print(Y_)
    # print(len(Y_))

    score = sklearn.metrics.accuracy_score(Y,Y_)
    f1_score = sklearn.metrics.f1_score(Y, Y_, average='macro')

    print('score with post-processing: ' + str(score))
    print('F1 score with post-processing: ' + str(f1_score))
    print()











