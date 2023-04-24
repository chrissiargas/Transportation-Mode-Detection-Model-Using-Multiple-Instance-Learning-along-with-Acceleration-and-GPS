import pprint
import time
from dataset import Dataset
from TMD import TMD_MIL
import sys
import ruamel.yaml
import warnings
import os
from evaluation import evaluate
import pandas as pd


savePath = os.path.join("saves", "save-" + time.strftime("%Y%m%d-%H%M%S"))

terminalFile = os.path.join(savePath, "terminal.txt")
scoresFile = os.path.join(savePath, "scores.csv")
statsFile = os.path.join(savePath, "stats.csv")
paramsFile = os.path.join(savePath, "parameters.yaml")

scores = pd.DataFrame()

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(terminalFile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:

            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)


def config_save():
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(paramsFile, 'w') as fb:
        yaml.dump(parameters, fb)


def scores_save():
    scores.to_csv(scoresFile, index=False)
    stats = pd.DataFrame()

    if 'Test User' in scores:

        meanPerUser = scores.groupby(['Test User']).mean()
        stdPerUser = scores.groupby(['Test User']).std()
        meanPerUser.columns = [str(col) + '_mean' for col in meanPerUser.columns]
        stdPerUser.columns = [str(col) + '_std' for col in stdPerUser.columns]
        stats = pd.concat([meanPerUser, stdPerUser], axis=1)
        stats.loc['All'] = stats.mean()
        stats['Test User'] = stats.index

    else:

        mean = dict(scores.mean())
        std = dict(scores.std())
        statsDict = {}
        mean = {k + '_mean': v for k, v in mean.items()}
        std = {k + '_std': v for k, v in std.items()}
        statsDict.update(mean)
        statsDict.update(std)
        stats = stats.append(statsDict, ignore_index=True)

    stats.to_csv(statsFile, index=False)


def execute(repeat = 10,
            evaluation = False,
            regenerate = False,
            all_users = True,
            logger = False,
            postprocessing = True,
            mVerbose = False):

    global scores

    if logger:
        sys.stdout = Logger()

    regenerated = False
    if all_users:
        for _ in range(repeat):
            for test_user in [1, 2, 3]:

                print('USER: ' + str(test_user))

                config_edit('train_args', 'test_user', test_user)

                regenerate = regenerate if not regenerated else False
                regenerated = True

                data = Dataset(regenerate=regenerate)

                if logger:
                    pprint.pprint(data.shl_args.data_args)
                    print()
                    pprint.pprint(data.shl_args.train_args)
                    print()

                if evaluation:
                    acc, f1, postAcc, postF1 = evaluate(data=data,
                                                        verbose=data.verbose,
                                                        postprocessing=postprocessing)

                else:
                    acc, f1, postAcc, postF1 = TMD_MIL(data=data,
                                                       summary=True,
                                                       verbose=data.verbose,
                                                       postprocessing=postprocessing,
                                                       mVerbose=mVerbose)

                if postprocessing:
                    theseScores = {'Test User': str(test_user),
                                   'Accuracy': acc,
                                   'F1-Score': f1,
                                   'post-Accuracy': postAcc,
                                   'post-F1-Score': postF1}
                else:
                    theseScores = {'Test User': str(test_user),
                                   'Accuracy': acc,
                                   'F1-Score': f1}

                scores = scores.append(theseScores, ignore_index=True)

            save()

    else:
        for _ in range(repeat):
            regenerate = regenerate if not regenerated else False
            regenerated = False

            data = Dataset(regenerate=regenerate)

            if logger:
                pprint.pprint(data.shl_args.data_args)
                print()
                pprint.pprint(data.shl_args.train_args)
                print()

            if evaluation:
                acc, f1, postAcc, postF1 = evaluate(data=data,
                                                    verbose=data.verbose,
                                                    postprocessing=postprocessing)

            else:
                acc, f1, postAcc, postF1 = TMD_MIL(data=data,
                                                   summary=True,
                                                   verbose=data.verbose,
                                                   postprocessing=postprocessing,
                                                   mVerbose=mVerbose)

            if postprocessing:
                theseScores = {'Accuracy': acc,
                               'F1-Score': f1,
                               'post-Accuracy': postAcc,
                               'post-F1-Score': postF1}
            else:
                theseScores = {'Accuracy': acc,
                               'F1-Score': f1}

            scores = scores.append(theseScores, ignore_index=True)


def save():
    try:
        os.makedirs(savePath)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    scores_save()
    config_save()


def main():
    repeat = 10
    evaluation = False
    regenerate = False
    all_users = True
    logger = False
    postprocessing = True
    mVerbose = False

    try:
        execute(repeat = repeat,
                evaluation = evaluation,
                regenerate = regenerate,
                all_users = all_users,
                logger = logger,
                postprocessing = postprocessing,
                mVerbose = mVerbose)
    finally:
        save()


if __name__ == "__main__":
    main()





