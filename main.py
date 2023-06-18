import copy
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

scores = pd.DataFrame()
cm = [[] for _ in range(3)]

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


def config_save(paramsFile):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(paramsFile, 'w') as fb:
        yaml.dump(parameters, fb)


def scores_save(scoresFile, statsFile, cmFolder):
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

    sumCm = pd.DataFrame()
    nUsers = 0
    for i, cmUser in enumerate(cm):
        if len(cmUser):
            nUsers += 1

        sumCmUser = pd.DataFrame()
        for j, df in enumerate(cmUser):
            cmFile = os.path.join(cmFolder, 'confusion_{}_{}.csv'.format(i, j))
            df.to_csv(cmFile, index=False)

            if j == 0:
                sumCmUser = copy.deepcopy(df)
            else:
                sumCmUser += df

        # print(sumCmUser)
        # print(len(cmUser))
        # print(sumCmUser / len(cmUser))
        meanCmUser = sumCmUser / len(cmUser)
        # print(meanCmUser)
        cmFile = os.path.join(cmFolder, 'confusion_{}.csv'.format(i))
        meanCmUser.to_csv(cmFile, index=False)

        if i == 0:
            sumCm = copy.deepcopy(meanCmUser)
        else:
            sumCm += meanCmUser

    meanCm = sumCm / nUsers
    cmFile = os.path.join(cmFolder, 'confusion.csv')
    meanCm.to_csv(cmFile, index=False)


def execute(repeat=10,
            evaluation=False,
            regenerate=False,
            all_users=True,
            logger=False,
            postprocessing=False,
            mVerbose=False,
            hparams=None,
            accScores=False,
            gpsScores=False):
    global scores
    global cm
    scores = pd.DataFrame()
    cm = [[] for _ in range(3)]

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
                    acc, f1, postAcc, postF1, cmT, accA, f1A, cmA, accG, f1G = evaluate(data=data,
                                                                                        verbose=data.verbose,
                                                                                        postprocessing=postprocessing)

                else:
                    acc, f1, postAcc, postF1, cmT, accA, f1A, cmA, accG, f1G = TMD_MIL(data=data,
                                                                                       summary=True,
                                                                                       verbose=data.verbose,
                                                                                       postprocessing=postprocessing,
                                                                                       mVerbose=mVerbose,
                                                                                       accScores=accScores,
                                                                                       gpsScores=gpsScores)

                if postprocessing:
                    if accScores:
                        theseScores = {'Test User': str(test_user),
                                       'AccuracyAcc': accA,
                                       'f1Acc': f1A}

                        cm[test_user - 1].append(cmA)

                    elif gpsScores:
                        theseScores = {'Test User': str(test_user),
                                       'AccuracyGPS': accG,
                                       'f1GPS': f1G}

                    else:
                        theseScores = {'Test User': str(test_user),
                                       'Accuracy': acc,
                                       'F1-Score': f1,
                                       'post-Accuracy': postAcc,
                                       'post-F1-Score': postF1}

                        cm[test_user - 1].append(cmT)

                else:
                    if accScores:
                        theseScores = {'Test User': str(test_user),
                                       'AccuracyAcc': accA,
                                       'f1Acc': f1A}

                        cm[test_user - 1].append(cmA)

                    elif gpsScores:
                        theseScores = {'Test User': str(test_user),
                                       'AccuracyGPS': accA,
                                       'f1GPS': f1A}

                    else:
                        theseScores = {'Test User': str(test_user),
                                       'Accuracy': acc,
                                       'F1-Score': f1}

                        cm[test_user - 1].append(cmT)

                scores = scores.append(theseScores, ignore_index=True)

                save(hparams)

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

            save(hparams)


def save(hparams=None):
    if not hparams:
        try:
            path = savePath
            os.makedirs(savePath)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            path = savePath
            cmFolder = os.path.join(path, "confusion")
            os.makedirs(cmFolder)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    else:
        try:
            path = os.path.join(savePath, hparams)
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            cmFolder = os.path.join(path, "confusion")
            os.makedirs(cmFolder)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    scoresFile = os.path.join(path, "scores.csv")
    statsFile = os.path.join(path, "stats.csv")
    paramsFile = os.path.join(path, "parameters.yaml")
    cmFolder = os.path.join(path, "confusion")

    scores_save(scoresFile, statsFile, cmFolder)
    config_save(paramsFile)


def AccelerationExperiments(repeat=3,
                            all_users=True,
                            postprocessing=False,
                            regenerate=True,
                            mVerbose=False,
                            accScores=True):
    global scores
    global cm

    samplingRates = [0.025]
    totalD = 60
    for samplingRate in samplingRates:
        scores = pd.DataFrame()
        duration = int(totalD / samplingRate)
        config_edit('data_args', 'accSamplingRate', samplingRate)
        config_edit('data_args', 'accDuration', duration)
        config_edit('train_args', 'accDuration', duration)
        config_edit('data_args', 'accBagStride', duration)
        config_edit('train_args', 'accBagStride', duration)
        config_edit('data_args', 'accStride', duration)
        hparams = 'samplingRate-' + str(samplingRate)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose,
                accScores=accScores)

    samplingRate = 0.1
    totalD = 60
    duration = int(totalD / samplingRate)
    config_edit('data_args', 'accSamplingRate', samplingRate)
    config_edit('data_args', 'accDuration', duration)
    config_edit('train_args', 'accDuration', duration)
    config_edit('data_args', 'accBagStride', duration)
    config_edit('train_args', 'accBagStride', duration)
    config_edit('data_args', 'accStride', duration)

    acc_signalss = [['Acc_norm', 'Acc_y', 'Acc_x', 'Acc_z', 'Jerk']]

    for acc_signals in acc_signalss:
        scores = pd.DataFrame()
        config_edit('train_args', 'acc_signals', acc_signals)
        hparams = 'acc_signals-' + str(acc_signals)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose,
                accScores=accScores)
        regenerate = False

    acc_signals = ['Acc_norm', 'Jerk']
    config_edit('train_args', 'acc_signals', acc_signals)
    #
    # freq_interpolations = ['linear', 'log']
    # for freq_interpolation in freq_interpolations:
    #     scores = pd.DataFrame()
    #     config_edit('train_args', 'freq_interpolation', freq_interpolation)
    #     hparams = 'freq_interpolation-' + str(freq_interpolation)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #
    # freq_interpolation = 'log'
    # config_edit('train_args', 'freq_interpolation', freq_interpolation)
    #
    # log_powers = [False, True]
    # for log_power in log_powers:
    #     scores = pd.DataFrame()
    #     config_edit('train_args', 'log_power', log_power)
    #     hparams = 'log_power-' + str(log_power)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #
    # log_power = True
    # config_edit('train_args', 'log_power', log_power)
    #
    # specto_augments = [[], ['frequencyMask', 'timeMask']]
    # for specto_augment in specto_augments:
    #     scores = pd.DataFrame()
    #     config_edit('train_args', 'specto_augment', specto_augment)
    #     hparams = 'specto_augment-' + str(specto_augment)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #
    # specto_augment = ['frequencyMask', 'timeMask']
    # config_edit('train_args', 'specto_augment', specto_augment)
    #
    # specto_overlaps = [8.5, 9, 9.5, 9.9]
    # for specto_overlap in specto_overlaps:
    #     scores = pd.DataFrame()
    #     config_edit('train_args', 'specto_overlap', specto_overlap)
    #     hparams = 'specto_overlap-' + str(specto_overlap)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)


def AccMILExperiments(repeat=3,
                      all_users=True,
                      postprocessing=False,
                      regenerate=True,
                      mVerbose=False,
                      accScores=True):
    global scores
    global cm

    # N = [1, 2, 3, 4, 5, 6, 7]
    # for n in N:
    #     duration = n * 600
    #     scores = pd.DataFrame()
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     hparams = 'duration-' + str(duration)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)

    # config_edit('train_args', 'separate_MIL', True)
    # accBagSizes = [7]
    # specto_overlaps = [9]
    # bagDuration = 1800
    # for accBagSize, specto_overlap in zip(accBagSizes, specto_overlaps):
    #     scores = pd.DataFrame()
    #     duration = int(bagDuration / accBagSize)
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'specto_overlap', specto_overlap)
    #
    #     hparams = 'BagDuration-' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)

    # duration = 600
    # config_edit('data_args', 'accDuration', duration)
    # config_edit('train_args', 'accDuration', duration)
    # config_edit('data_args', 'accBagStride', duration)
    # config_edit('train_args', 'accBagStride', duration)
    # config_edit('train_args', 'separate_MIL', True)
    #
    # accBagSizes = [1]
    # for accBagSize in accBagSizes:
    #     scores = pd.DataFrame()
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #     hparams = 'instanceDuration-' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)

    config_edit('train_args', 'separate_MIL', True)
    config_edit('train_args', 'train_bag_positions', 'random')
    config_edit('train_args', 'test_bag_positions', 'random')
    # config_edit('train_args', 'fusion', 'MIL')
    #
    # config_edit('train_args', 'epochs', 160)
    # config_edit('train_args', 'oversampling', False)

    config_edit('train_args', 'transfer_learning_acc', 'train')
    config_edit('train_args', 'transfer_learning_loc', 'none')

    regenerate = False
    accBagSizes = [3]
    bagDuration = 1800
    for accBagSize in accBagSizes:
        scores = pd.DataFrame()
        duration = int(bagDuration / accBagSize)
        config_edit('data_args', 'accDuration', duration)
        config_edit('train_args', 'accDuration', duration)
        config_edit('data_args', 'accBagStride', duration)
        config_edit('train_args', 'accBagStride', duration)
        config_edit('data_args', 'accBagSize', accBagSize)
        config_edit('train_args', 'accBagSize', accBagSize)

        hparams = 'ACC-MIL' + 'accBagSize-' + str(accBagSize)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose,
                accScores=accScores)


def GPSExperiments(repeat=3,
                   all_users=True,
                   postprocessing=False,
                   regenerate=True,
                   mVerbose=False,
                   accScores=False,
                   gpsScores=True):
    global scores

    # gpsSamplingRates = [10, 20, 30, 40, 50, 60]
    # samplingThresholds = [2, 4, 6, 8, 10, 12]
    # interpolateThresholds = [18, 9, 6, 5, 4, 3]

    # gpsSamplingRates = [20, 30, 40, 50, 60]
    # samplingThresholds = [4, 6, 8, 10, 12]
    # interpolateThresholds = [9, 6, 5, 4, 3]
    #
    # for gpsSamplingRate, samplingThreshold, interpolateThreshold in zip(gpsSamplingRates, samplingThresholds,
    #                                                                     interpolateThresholds):
    #
    #     duration = 720
    #     locDuration = int(duration / gpsSamplingRate)
    #     scores = pd.DataFrame()
    #     config_edit('data_args', 'gpsSamplingRate', gpsSamplingRate)
    #     config_edit('data_args', 'samplingThreshold', samplingThreshold)
    #     config_edit('data_args', 'interpolateThreshold', interpolateThreshold)
    #     config_edit('data_args', 'locDuration', locDuration)
    #
    #     hparams = 'gpsSamplingRate-' + str(gpsSamplingRate)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores,
    #             gpsScores=gpsScores)

    # gpsSamplingRate = 60
    # samplingThreshold = 10
    # interpolateThreshold = 3
    # locDuration = 12
    #
    # config_edit('data_args', 'gpsSamplingRate', gpsSamplingRate)
    # config_edit('data_args', 'samplingThreshold', samplingThreshold)
    # config_edit('data_args', 'interpolateThreshold', interpolateThreshold)
    # config_edit('data_args', 'locDuration', locDuration)
    #
    # time_featuress = [['Velocity'], ['Velocity', 'Acceleration'], ['Velocity', 'Acceleration', 'BearingRate']]
    # for time_features in time_featuress:
    #     scores = pd.DataFrame()
    #     config_edit('train_args', 'time_features', time_features)
    #     hparams = 'time_features-' + str(time_features)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores,
    #             gpsScores=gpsScores)
    #
    #     regenerate = False

    time_features = ['Velocity', 'Acceleration']
    config_edit('train_args', 'time_features', time_features)

    statistical_featuress = [['TotalMovability'], ['Mean', 'Var'], ['TotalMovability', 'Mean', 'Var']]
    for statistical_features in statistical_featuress:
        scores = pd.DataFrame()
        config_edit('train_args', 'statistical_features', statistical_features)
        hparams = 'statistical_features-' + str(statistical_features)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose,
                accScores=accScores,
                gpsScores=gpsScores)

    statistical_features = ['TotalMovability', 'Mean', 'Var']
    config_edit('train_args', 'statistical_features', statistical_features)

    gps_augmentations = [False, True]
    for gps_augmentation in gps_augmentations:
        scores = pd.DataFrame()
        config_edit('train_args', 'gps_augmentation', gps_augmentation)
        hparams = 'gps_augmentation-' + str(gps_augmentation)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose,
                accScores=accScores,
                gpsScores=gpsScores)


def HpExecute2(repeat=10,
               all_users=True,
               postprocessing=False,
               regenerate=True,
               mVerbose=False):
    global scores

    headss = [1, 2, 3, 4]
    for heads in headss:
        config_edit('train_args', 'heads', heads)
        hparams = 'heads-' + str(heads)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose)


def MM_MILExperiments(repeat=3,
                      all_users=True,
                      postprocessing=False,
                      regenerate=False,
                      mVerbose=False,
                      accScores=False):
    global scores
    global cm

    # config_edit('train_args', 'separate_MIL', False)
    # config_edit('train_args', 'train_bag_positions', 'same')
    # config_edit('train_args', 'test_bag_positions', 'same')
    # config_edit('train_args', 'transfer_learning_acc', 'none')
    # config_edit('train_args', 'transfer_learning_loc', 'none')
    # config_edit('train_args', 'epochs', 160)
    # config_edit('train_args', 'oversampling', True)
    # config_edit('train_args', 'fusion', 'MIL')

    # regenerate = True
    # accBagSizes = [3]
    # bagDuration = 1800
    # for accBagSize in accBagSizes:
    #     scores = pd.DataFrame()
    #     duration = int(bagDuration / accBagSize)
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #
    #     hparams = 'MM-MIL-' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #
    # regenerate = False
    # config_edit('train_args', 'fusion', 'concat')
    # config_edit('train_args', 'separate_MIL', True)
    # accBagSizes = [3]
    # bagDuration = 1800
    # for accBagSize in accBagSizes:
    #     scores = pd.DataFrame()
    #     duration = int(bagDuration / accBagSize)
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #
    #     hparams = 'ACC-MIL-MM-CONCAT-' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)

    config_edit('train_args', 'separate_MIL', False)
    config_edit('train_args', 'train_bag_positions', 'same')
    config_edit('train_args', 'test_bag_positions', 'same')
    config_edit('train_args', 'fusion', 'MIL')

    config_edit('train_args', 'epochs', 160)
    config_edit('train_args', 'oversampling', False)

    config_edit('train_args', 'transfer_learning_acc', 'train')
    config_edit('train_args', 'transfer_learning_loc', 'none')

    regenerate = False
    accBagSizes = [3]
    bagDuration = 1800
    for accBagSize in accBagSizes:
        scores = pd.DataFrame()
        duration = int(bagDuration / accBagSize)
        config_edit('data_args', 'accDuration', duration)
        config_edit('train_args', 'accDuration', duration)
        config_edit('data_args', 'accBagStride', duration)
        config_edit('train_args', 'accBagStride', duration)
        config_edit('data_args', 'accBagSize', accBagSize)
        config_edit('train_args', 'accBagSize', accBagSize)

        hparams = 'TRANSFER-ACC-MM-MIL' + 'accBagSize-' + str(accBagSize)
        execute(repeat=repeat,
                all_users=all_users,
                postprocessing=postprocessing,
                regenerate=regenerate,
                hparams=hparams,
                mVerbose=mVerbose,
                accScores=accScores)
    #
    # regenerate = False
    # config_edit('train_args', 'fusion', 'concat')
    # config_edit('train_args', 'separate_MIL', True)
    # accBagSizes = [3]
    # bagDuration = 1800
    # for accBagSize in accBagSizes:
    #     scores = pd.DataFrame()
    #     duration = int(bagDuration / accBagSize)
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #
    #     hparams = 'TRANFER-ACC-MIL-MM-CONCAT-' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #
    # config_edit('train_args', 'separate_MIL', False)
    # config_edit('train_args', 'train_bag_positions', 'same')
    # config_edit('train_args', 'test_bag_positions', 'same')
    # config_edit('train_args', 'transfer_learning_acc', 'none')
    # config_edit('train_args', 'transfer_learning_loc', 'none')
    # config_edit('train_args', 'epochs', 160)
    # config_edit('train_args', 'oversampling', True)
    # config_edit('train_args', 'fusion', 'MIL')
    #
    # regenerate = True
    # accBagSizes = [1]
    # bagDuration = 1800
    # config_edit('train_args', 'fusion', 'concat')
    #
    # for accBagSize in accBagSizes:
    #     scores = pd.DataFrame()
    #     duration = int(bagDuration / accBagSize)
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #
    #     hparams = 'MM-CONCAT' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #
    # config_edit('train_args', 'separate_MIL', False)
    # config_edit('train_args', 'train_bag_positions', 'same')
    # config_edit('train_args', 'test_bag_positions', 'same')
    # config_edit('train_args', 'transfer_learning_loc', 'none')
    # config_edit('train_args', 'fusion', 'MIL')
    #
    # config_edit('train_args', 'epochs', 160)
    # config_edit('train_args', 'oversampling', False)
    # config_edit('train_args', 'transfer_learning_acc', 'train')
    #
    # regenerate = False
    # accBagSizes = [1]
    # bagDuration = 1800
    # config_edit('train_args', 'fusion', 'concat')
    #
    # for accBagSize in accBagSizes:
    #     scores = pd.DataFrame()
    #     duration = int(bagDuration / accBagSize)
    #     config_edit('data_args', 'accDuration', duration)
    #     config_edit('train_args', 'accDuration', duration)
    #     config_edit('data_args', 'accBagStride', duration)
    #     config_edit('train_args', 'accBagStride', duration)
    #     config_edit('data_args', 'accBagSize', accBagSize)
    #     config_edit('train_args', 'accBagSize', accBagSize)
    #
    #     hparams = 'TRANFER-MM-CONCAT' + 'accBagSize-' + str(accBagSize)
    #     execute(repeat=repeat,
    #             all_users=all_users,
    #             postprocessing=postprocessing,
    #             regenerate=regenerate,
    #             hparams=hparams,
    #             mVerbose=mVerbose,
    #             accScores=accScores)
    #


def main():
    try:
        execute(repeat=4,
                all_users=True,
                postprocessing=True,
                regenerate=False,
                mVerbose=False,
                accScores=False,
                evaluation=False)

        # MM_MILExperiments(repeat=4,
        #                   all_users=True,
        #                   postprocessing=True,
        #                   regenerate=False,
        #                   mVerbose=False,
        #                   accScores=False)

        # AccMILExperiments(repeat=4,
        #                   all_users=True,
        #                   postprocessing=False,
        #                   regenerate=False,
        #                   mVerbose=True,
        #                   accScores=True)
    finally:
        save()


if __name__ == "__main__":
    main()
