import numpy as np
import pandas as pd
from hmm_filter.hmm_filter import HMMFilter


def get_dataset(data, Model, train=False):
    y_, y, lengths, time = data.yToSequence(Model=Model, prob=True, train=train)

    dataset = pd.DataFrame()
    begin = 0
    session_id = 0
    n_classes = 5 if data.motorized else 8
    modes = [i for i in range(n_classes)]

    for length in lengths:
        session_Id = [session_id for _ in range(length)]
        timestamp = time[begin: begin + length]
        true = y[begin: begin + length]

        if not train:
            probabs = [{k: v / sum(probs) for k, v in zip(modes, probs) if v > 0} for probs in y_[begin: begin + length]]
            pred = np.argmax(y_[begin: begin + length], axis=1)

            seq = pd.DataFrame({'session_id': session_Id,
                                'timestamp': timestamp,
                                'true': true,
                                'pred': pred,
                                'prob': probabs})

        else:
            seq = pd.DataFrame({'session_id': session_Id,
                                'timestamp': timestamp,
                                'true': true})

        dataset = pd.concat([dataset, seq], ignore_index=True)

        begin += length
        session_id += 1

    y_pred = None if train else np.argmax(y_, axis=1)
    return dataset, y, y_pred


def fit_predict(train_dataset, test_dataset):
    hmmfilter = HMMFilter()
    hmmfilter.fit(train_dataset, session_column="session_id", prediction_column="true")
    post = hmmfilter.predict(test_dataset, session_column='session_id',
                             probabs_column="prob", prediction_column='pred')
    postY_ = post['pred'].tolist()
    return postY_
