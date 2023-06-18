import contextlib
import random
import pandas as pd
from configParser import Parser
from extractData import extractData
from buildData import buildData
from transformers import *
from features import *
import os

np.set_printoptions(precision=14)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class Dataset:

    def __init__(self, regenerate=False):

        parser = Parser()
        self.shl_args = parser.get_args()
        self.verbose = self.shl_args.train_args['verbose']

        if not regenerate:
            xData = extractData(self.shl_args)

            if not xData.found:

                bData = buildData(args=self.shl_args, verbose=self.verbose)

                bData()
                del bData

                self.acceleration, self.labels, self.location = xData()

            else:
                self.acceleration, self.labels, self.location = xData()

            del xData

        else:

            bData = buildData(args=self.shl_args,
                              verbose=self.verbose,
                              delete_dst=True,
                              delete_tmp=True,
                              delete_final=True,
                              delete_filter=True)

            bData()
            del bData

            xData = extractData(self.shl_args)

            self.acceleration, \
                self.labels, \
                self.location = xData()

            del xData

    def initialize(self):

        self.complete = (self.shl_args.data_args['dataset'] == 'CompleteUser1')

        if self.complete:

            self.positions = ['Hips']
            self.positionsDict = {'Hips': 0}
            self.files = {
                '1': ['010317', '010617', '020317', '020517', '020617', '030317',
                      '030517', '030617', '030717', '040517', '040717', '050517',
                      '050617', '050717', '060317', '060617', '070317', '070617',
                      '080317', '080517', '080617', '090317', '090517', '090617',
                      '100317', '100517', '110517', '120517', '120617', '130317',
                      '130617', '140317', '140617', '150317', '150517', '150617',
                      '160317', '170317', '170517', '190417',
                      '190517', '200317', '200417', '200517', '200617', '210317',
                      '220317', '220517', '220617', '230317', '230517', '230617',
                      '240317', '240417', '240517', '250317', '250417', '250517',
                      '260417', '260517', '260617', '270317', '270417', '270617',
                      '280317', '280417', '280617', '290317', '290517', '290617',
                      '300317', '300517', '300617', '310517']
            }

        else:
            self.positions = ['Torso', 'Hips', 'Bag', 'Hand']
            self.positionsDict = {
                'Torso': 0,
                'Hips': 1,
                'Bag': 2,
                'Hand': 3
            }

        if self.complete:
            self.whichGPS = 'Hips'

        else:
            self.whichGPS = 'Hand'

        self.testUser = self.shl_args.train_args['test_user']
        self.bags = self.labels.shape[0]
        self.accBagSize = self.shl_args.train_args['accBagSize']
        self.trainBagPositions = self.shl_args.train_args['train_bag_positions']
        self.testBagPositions = self.shl_args.train_args['test_bag_positions']
        self.trainBatchSize = self.shl_args.train_args['trainBatchSize']
        self.valBatchSize = self.shl_args.train_args['valBatchSize']
        self.testBatchSize = self.shl_args.train_args['testBatchSize']
        self.gpsMode = self.shl_args.train_args['transfer_learning_loc']
        self.gpsEpochs = self.shl_args.train_args['locEpochs']
        self.accMode = self.shl_args.train_args['transfer_learning_acc']
        self.accEpochs = self.shl_args.train_args['accEpochs']
        self.accMIL = self.shl_args.train_args['separate_MIL']
        self.lr = self.shl_args.train_args['learning_rate']
        self.epochs = self.shl_args.train_args['epochs']
        self.syncing = self.shl_args.data_args['sync']
        self.gpsDuration = self.shl_args.data_args['locDuration']
        self.syncThreshold = self.shl_args.train_args['pair_threshold'] * 1000
        self.transThreshold = self.shl_args.train_args['transition_threshold'] * 1000
        self.paddingThreshold = self.gpsDuration
        self.useSpectro = self.shl_args.train_args['use_spectrograms']
        self.trainPosition = self.shl_args.train_args['train_position']
        self.testPosition = self.shl_args.train_args['test_position']
        self.trainPerInstance = 4 if self.trainBagPositions == 'all' else 1
        self.testPerInstance = 4 if self.testBagPositions == 'all' else 1
        self.multipleTrain = self.shl_args.train_args['multiple_train']
        self.multipleVal = self.shl_args.train_args['multiple_val']
        self.multipleTest = self.shl_args.train_args['multiple_test']
        self.accShape = None
        self.accTimeShape = None
        self.accTfrm = None
        self.gpsWindowShape = None
        self.gpsFeaturesShape = None
        self.gpsTimeShape = None
        self.gpsTfrm = None
        self.lbsTfrm = None
        self.inputShape = None
        self.inputType = None
        self.timeShape = None
        self.timeType = None
        self.random = self.shl_args.train_args['randomize']
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.trainSize = 0
        self.valSize = 0
        self.testSize = 0
        self.train_days = []
        self.val_days = []
        self.test_days = []
        self.valPercentage = self.shl_args.train_args['val_percentage']
        self.accBags = None
        self.lbsBags = None
        self.gpsBags = None
        self.motorized = self.shl_args.train_args['motorized']
        self.n_classes = 5 if self.motorized else 8
        self.oversampling = self.shl_args.train_args['oversampling']

    def to_pandas(self, indices, motorized=True, includeGpsLoss=False):
        data = []

        for en, index in enumerate(indices):

            i = index[0]
            position = index[1][0][0]

            location = self.location[self.positions.index(self.whichGPS)][self.gpsBags[self.whichGPS][i]]
            vel = gps_features(location, self.gpsDuration)

            if vel == -1 and not includeGpsLoss:
                continue

            var, coef1Hz, coef2Hz, coef3Hz = acc_features(self.acceleration[self.accBags[i]], position=position)

            Lb = self.labels[self.lbsBags[i]][0] - 1
            if motorized:
                Lb = Lb if Lb < 4 else 4

            user, day, time = self.labels[self.lbsBags[i]][-3:]

            if en == 0:
                data = [[vel, var, coef1Hz, coef2Hz, coef3Hz, user, day, time, position, Lb]]

            else:
                data.append([vel, var, coef1Hz, coef2Hz, coef3Hz, user, day, time, position, Lb])

        data = pd.DataFrame(data,
                            columns=['vel', 'var', '1Hz', '2Hz', '3Hz', 'User', 'Day', 'Time', 'Position', 'Label'])

        return data

    def to_bags(self):

        bagMap = {
            'acc': [[] for _ in range(self.bags)],
            'labels': [i for i in range(self.bags)],
            'gps': {}
        }

        for i in range(self.bags):
            label = self.labels[i]
            sample_index = label[1]
            bagMap['acc'][i].append(sample_index)

        dailyGpsData = []
        gpsPivot = None
        if self.syncing == 'Past':
            gpsPivot = self.gpsDuration - 1
        elif self.syncing == 'Present':
            gpsPivot = self.gpsDuration // 2
        elif self.syncing == 'Future':
            gpsPivot = 0

        for index, position in enumerate(self.positions):

            bagMap['gps'][position] = [[] for _ in range(self.bags)]

            if position == self.whichGPS:

                dailyStart = 0
                NDaily = 0
                searchStart = 0

                for i, label in enumerate(self.labels):

                    user = label[-3]
                    day = label[-2]
                    time = label[-1]

                    if i == 0 or user != self.labels[i - 1][-3] or day != self.labels[i - 1][-2]:

                        dailyStart += NDaily
                        dailyGpsData = self.select_location(user,
                                                            day,
                                                            index,
                                                            dailyStart)

                        NDaily = dailyGpsData.shape[0]
                        searchStart = 0

                    else:
                        if len(bagMap['gps'][position][i - 1]):
                            searchStart = bagMap['gps'][position][i - 1][0] - dailyStart

                    found = False
                    for offset, GPS in enumerate(dailyGpsData[searchStart:]):

                        timeDistance = np.abs(GPS[gpsPivot, -1] - time)

                        if timeDistance > self.syncThreshold:
                            if found:
                                break

                        else:
                            found = True
                            filled = np.sum(np.count_nonzero(GPS == -1, axis=1) == 0)
                            if filled >= self.paddingThreshold:
                                bagMap['gps'][position][i].append(offset + searchStart + dailyStart)

            thisBagMap = []
            gpsBagSize = 1
            for i, bag in enumerate(bagMap['gps'][position]):

                if len(bag) > gpsBagSize:
                    divergence_list = []
                    gpsBag = self.location[index][bag]
                    for GPS in gpsBag:
                        timestamp = GPS[gpsPivot, -1]
                        div = np.abs(self.labels[i, -1] - timestamp)
                        divergence_list.append(div)

                    closest = np.argpartition(divergence_list, gpsBagSize)[:gpsBagSize]
                    thisBagMap.append([bag[c] for c in closest])

                elif len(bag) <= gpsBagSize:
                    thisBagMap.append(bag)

            bagMap['gps'][position] = thisBagMap

        self.accBags, self.lbsBags, self.gpsBags = bagMap['acc'], bagMap['labels'], bagMap['gps']

    def select_location(self, user, day, position, start):
        output = []
        found = False

        if user == 3 and day >= 2:
            day -= 1

        for loc_sample in self.location[position][start:]:

            if loc_sample[0, -3] == user and loc_sample[0, -2] == day:

                found = True
                output.append(loc_sample)

            elif found:
                break

        return np.array(output)

    def init_transformers(self, accTransfer=False, gpsTransfer=False, timeInfo=False):

        if not gpsTransfer:

            if self.useSpectro:
                self.accTfrm = spectrogramTransformer(
                    self.shl_args,
                    accTransfer=accTransfer,
                    accMIL=self.accMIL
                )

            else:
                self.accTfrm = temporalTransformer(
                    self.shl_args,
                    accTransfer=accTransfer,
                    accMIL=self.accMIL
                )

            self.accShape = self.accTfrm.get_shape()
            self.accTimeShape = self.accTfrm.get_time_shape()

        if not accTransfer:
            self.gpsTfrm = gpsTransformer(shl_args=self.shl_args, gpsTransfer=gpsTransfer)
            self.gpsWindowShape, self.gpsFeaturesShape = self.gpsTfrm.get_shape
            self.gpsTimeShape = self.gpsTfrm.get_time_shape()

        self.lbsTfrm = CategoricalTransformer(motorized=self.motorized)

        if gpsTransfer:
            self.inputShape = (self.gpsWindowShape, self.gpsFeaturesShape)
            self.inputType = (tf.float64, tf.float64)
            if timeInfo:
                self.timeShape = (self.gpsTimeShape, 3)
                self.timeType = (tf.float64, tf.float64)

        elif accTransfer:
            self.inputShape = (self.accShape, self.accBagSize * self.trainPerInstance)
            self.inputType = (tf.float64, tf.int32)
            if timeInfo:
                self.timeShape = (self.accTimeShape, 3)
                self.timeType = (tf.float64, tf.float64)

        else:
            self.inputShape = (
                self.accShape, self.gpsWindowShape, self.gpsFeaturesShape, self.accBagSize * self.trainPerInstance)
            self.inputType = (tf.float64, tf.float64, tf.float64, tf.int32)
            if timeInfo:
                self.timeShape = (self.accTimeShape, self.gpsTimeShape, 3)
                self.timeType = (tf.float64, tf.float64, tf.float64)

    def to_generator(self, is_val=False, is_test=False, accTransfer=False, gpsTransfer=False, timeInfo=False):

        if not is_test:
            if not is_val:
                indices = self.train_indices

            else:
                indices = self.val_indices

        else:
            indices = self.test_indices

        def gen():
            for index in indices:

                if not gpsTransfer:

                    bagIndex = index[0]
                    instancePositions = []
                    positions = []
                    for instance, posI in enumerate(index[1]):
                        for pos in posI:
                            instancePositions.append([instance, pos])
                            positions.append(pos)

                else:
                    bagIndex = index
                    instancePositions = None

                if not gpsTransfer:
                    accBag, accTime = self.accTfrm(self.acceleration[self.accBags[bagIndex]],
                                                   is_train=not (is_val or is_test),
                                                   position=instancePositions, timeInfo=timeInfo)

                if not accTransfer:
                    location = self.location[self.positions.index(self.whichGPS)][self.gpsBags[self.whichGPS][bagIndex]]
                    gpsSeries, gpsFeatures, gpsTime = self.gpsTfrm(location, timeInfo=timeInfo,
                                                                   is_train=not (is_val or is_test))

                y, yTime = self.lbsTfrm(self.labels[self.lbsBags[bagIndex]], timeInfo=timeInfo)

                if timeInfo:
                    if gpsTransfer:
                        yield (gpsSeries, gpsFeatures), y, (gpsTime, yTime)

                    elif accTransfer:
                        yield (accBag, positions), y, (accTime, yTime)

                    else:
                        yield (accBag, gpsSeries, gpsFeatures, positions), y, (accTime, gpsTime, yTime)

                else:
                    if gpsTransfer:
                        yield (gpsSeries, gpsFeatures), y

                    elif accTransfer:
                        yield (accBag, positions), y

                    else:
                        yield (accBag, gpsSeries, gpsFeatures, positions), y

        if timeInfo:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(self.inputType,
                              tf.float32,
                              self.timeType),
                output_shapes=(self.inputShape,
                               self.n_classes,
                               self.timeShape))

        else:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(self.inputType,
                              tf.float32),
                output_shapes=(self.inputShape,
                               self.n_classes))

    def batch_and_prefetch(self, train, val, test):

        return train.cache().shuffle(1000).repeat().batch(batch_size=self.trainBatchSize).prefetch(tf.data.AUTOTUNE), \
            val.cache().shuffle(1000).repeat().batch(batch_size=self.valBatchSize).prefetch(tf.data.AUTOTUNE), \
            test.cache().shuffle(1000).repeat().batch(batch_size=self.testBatchSize).prefetch(tf.data.AUTOTUNE)

    def split_train_val(self, dataIndices):

        originalIndices = dataIndices
        dataIndices = pd.DataFrame(dataIndices, columns=['index', 'user_label'])
        count = dataIndices['user_label'].value_counts()
        val_count = count * self.valPercentage
        val_count = val_count.astype('int32')
        val_indices = []

        for user_label, count in val_count.items():

            candidates = pd.DataFrame()
            tmp_count = count

            while candidates.empty:
                candidates = dataIndices[dataIndices['user_label'] == user_label].user_label.groupby(
                    [dataIndices.user_label, dataIndices.user_label.diff().ne(0).cumsum()]).transform('size').ge(
                    tmp_count).astype(int)
                candidates = pd.DataFrame(candidates)
                candidates = candidates[candidates['user_label'] == 1]
                tmp_count = int(tmp_count * 0.95)

            index = candidates.sample(random_state=1).index[0]
            val_indices.append(index)
            n_indices = 1
            up = 1
            down = 1
            length = dataIndices.shape[0]

            while n_indices < tmp_count - 1:

                if index + up < length and user_label == dataIndices.iloc[index + up]['user_label']:
                    val_indices.append(index + up)
                    up += 1
                    n_indices += 1

                if index - down >= 0 and user_label == dataIndices.iloc[index - down]['user_label']:
                    val_indices.append(index - down)
                    down += 1
                    n_indices += 1

        val_indices.sort()
        self.val_indices = [originalIndices.pop(i - shift)[0] for shift, i in enumerate(val_indices)]
        self.valSize = len(self.val_indices)
        self.train_indices = [x[0] for x in originalIndices]
        self.trainSize = len(self.train_indices)

        if self.random:
            random.shuffle(self.val_indices)
            random.shuffle(self.train_indices)

    def split(self, seed=1):
        self.test_indices = []
        if self.complete:

            with temp_seed(seed):
                n_days = len(self.files['1'])
                days = np.delete(np.arange(stop=n_days), self.nan_days)
                n_days = days.shape[0]
                test_indices = np.random.choice(n_days, n_days // 4, replace=False)
                self.test_days = days[test_indices]
                days = np.delete(days, test_indices)
                n_days = days.shape[0]
                val_indices = np.random.choice(n_days, n_days // 4, replace=False)
                self.val_days = days[val_indices]
                self.train_days = np.delete(days, val_indices)

            print(self.train_days)
            print(self.val_days)
            print(self.test_days)

            for index, (label, day) in enumerate(zip(self.labels[:, 0],
                                                     self.labels[:, -2])):

                if day in self.test_days:
                    self.testSize += 1
                    self.test_indices.append(index)

                if day in self.train_days:
                    self.trainSize += 1
                    self.train_indices.append(index)

                elif day in self.val_days:
                    self.valSize += 1
                    self.val_indices.append(index)

        else:

            train_val_indices = []
            train_val_size = 0

            for index, (label, user) in enumerate(zip(self.labels[:, 0],
                                                      self.labels[:, -3])):

                if user == self.testUser:
                    self.testSize += 1
                    self.test_indices.append(index)

                else:
                    train_val_size += 1
                    train_val_indices.append([index, user * 10 + label])

            if self.random:
                random.shuffle(self.test_indices)

            self.split_train_val(train_val_indices)

    def yToSequence(self, Model, accTransfer=False, gpsTransfer=False, prob=False, train=False):

        if train:
            true = []
            true_sequence = []
            Time = []
            time_sequence = []
            lengths = []
            length = 0

            for index, (label, day, time, user) in enumerate(zip(self.labels[:, 0],
                                                                 self.labels[:, -2],
                                                                 self.labels[:, -1],
                                                                 self.labels[:, -3])):

                include = (self.complete and day not in self.test_days) or (not self.complete and user != self.testUser)

                if include:
                    length += 1
                    true_sequence.append(min(label - 1, self.n_classes - 1))
                    time_sequence.append(time)

                    if index == self.bags - 1 or \
                            self.labels[index + 1][-1] - time > self.transThreshold or \
                            self.labels[index + 1][-2] != day or \
                            self.labels[index + 1][-3] != user:

                        if length != 0:
                            true.extend(true_sequence)
                            Time.extend(time_sequence)
                            lengths.append(length)

                        true_sequence = []
                        time_sequence = []
                        length = 0

            return None, true, lengths, Time

        else:
            predicted = []
            true = []
            true_sequence = []
            Time = []
            time_sequence = []
            lengths = []
            length = 0

            if not accTransfer and not gpsTransfer:
                n_features = 4
            else:
                n_features = 2

            if self.testPosition != 'all' or not self.multipleTest or gpsTransfer:
                nSeqs = 1
                seqPos = [[]]
                inputs = [[[] for _ in range(n_features)]]
            else:
                nSeqs = len(self.positions)
                seqPos = [[] for _ in range(nSeqs)]
                inputs = [[[] for _ in range(n_features)] for _ in range(nSeqs)]

            for index, (label, day, time, user) in enumerate(zip(self.labels[:, 0],
                                                                 self.labels[:, -2],
                                                                 self.labels[:, -1],
                                                                 self.labels[:, -3])):

                include = (self.complete and day in self.test_days) or (user == self.testUser and not self.complete)

                if include:
                    if self.testPosition != 'all':
                        pos = self.positionsDict[self.testPosition]

                        if not len(seqPos[0]):
                            seqPos[0] = [[pos] for _ in range(self.accBagSize)]

                        else:
                            seqPos[0].append([pos])

                    elif self.testBagPositions == 'same':
                        if not len(seqPos[0]):
                            seqPos = [[[s] for _ in range(self.accBagSize)] for s in range(nSeqs)]

                        else:
                            for s in range(nSeqs):
                                seqPos[s].append([s])

                    elif self.testBagPositions == 'random':
                        if not len(seqPos[0]):
                            seqPos = [random.sample(range(4), nSeqs) for _ in range(self.accBagSize)]
                            seqPos = list(map(list, zip(*seqPos)))
                            seqPos = np.reshape(seqPos, (nSeqs, self.accBagSize, 1)).tolist()

                        else:
                            pos = random.sample(range(4), nSeqs)
                            for s in range(nSeqs):
                                seqPos[s].append([pos[s]])

                    elif self.testBagPositions == 'variable':
                        probability = 0.3

                        if not len(seqPos[0]):
                            makeTransition = np.random.uniform(size=self.accBagSize - 1) < probability
                            transitions = np.argwhere(makeTransition).squeeze(-1) + 1

                            varyingPos = None
                            previousTransition = 0
                            for k in range(len(transitions) + 1):
                                if k == len(transitions):
                                    transition = self.accBagSize
                                else:
                                    transition = transitions[k]

                                stablePos = random.sample(range(nSeqs), nSeqs)
                                stablePos = [stablePos for _ in range(transition - previousTransition)]
                                if not varyingPos:
                                    varyingPos = stablePos

                                else:
                                    varyingPos.extend(stablePos)

                                previousTransition = transition

                            seqPos = list(map(list, zip(*varyingPos)))
                            seqPos = np.reshape(seqPos, (nSeqs, self.accBagSize, 1)).tolist()

                        else:
                            if np.random.uniform() < probability:
                                pos = random.sample(range(nSeqs), nSeqs)
                            else:
                                pos = [seqPos[s][-1][0] for s in range(nSeqs)]

                            for s in range(nSeqs):
                                seqPos[s].append([pos[s]])

                    for s in range(nSeqs):

                        position = seqPos[s][-self.accBagSize:]
                        Iposition = []
                        for instance, posI in enumerate(position):
                            for pos in posI:
                                Iposition.append([instance, pos])

                        if not gpsTransfer:
                            accBag, accTime = self.accTfrm(
                                self.acceleration[self.accBags[index]],
                                is_train=False,
                                position=Iposition
                            )

                        if not accTransfer:
                            location = self.location[self.positions.index(self.whichGPS)][
                                self.gpsBags[self.whichGPS][index]]
                            gpsSeries, gpsFeatures, gpsTime = self.gpsTfrm(location, is_train=False)
                            if gpsTransfer:
                                if gpsSeries[0, 0] == -10000000:
                                    continue

                        if s == 0:
                            length += 1
                            true_sequence.append(min(label - 1, self.n_classes - 1))
                            time_sequence.append(time)

                        if accTransfer:
                            inputs[s][0].append(accBag)
                            inputs[s][1].append(position)

                        elif gpsTransfer:
                            inputs[s][0].append(gpsSeries)
                            inputs[s][1].append(gpsFeatures)

                        else:
                            inputs[s][0].append(accBag)
                            inputs[s][1].append(gpsSeries)
                            inputs[s][2].append(gpsFeatures)
                            inputs[s][3].append(position)

                    if index == self.bags - 1 or \
                            self.labels[index + 1][-1] - time > self.transThreshold or \
                            self.labels[index + 1][-2] != day or \
                            self.labels[index + 1][-3] != user:

                        if length != 0:
                            for s in range(nSeqs):
                                true.extend(true_sequence)
                                Time.extend(time_sequence)
                                if prob:
                                    predicted.extend(Model.predict([np.array(inputs[s][i]) for i in range(n_features)],
                                                                   verbose=0))

                                else:
                                    predicted.extend(
                                        np.argmax(Model.predict([np.array(inputs[s][i]) for i in range(n_features)],
                                                                verbose=0), axis=1))
                                lengths.append(length)

                        seqPos = [[] for _ in range(nSeqs)]
                        inputs = [[[] for _ in range(n_features)] for _ in range(nSeqs)]
                        true_sequence = []
                        time_sequence = []
                        length = 0

            if not prob:
                predicted = np.array(predicted)
                predicted = np.reshape(predicted, (-1, 1))

            return predicted, true, lengths, Time

    def same_position(self, indices, multiple=True, pos=None):
        n = self.shl_args.train_args['accBagSize']
        positions = len(self.positions)
        posIndices = []

        if multiple:
            for index in indices:

                if not len(posIndices):
                    posIndices = [[index, [[pos] for _ in range(n)]] for pos in range(positions)]
                    continue

                posIndices.extend([[index, [[pos] for _ in range(n)]] for pos in range(positions)])

        else:
            if pos is not None:
                for index in indices:

                    if not len(posIndices):
                        posIndices = [[index, [[pos] for _ in range(n)]]]
                        continue

                    posIndices.append([index, [[pos] for _ in range(n)]])

            else:
                for index in indices:
                    tmpPos = random.randrange(positions)

                    if not len(posIndices):
                        posIndices = [[index, [[tmpPos] for _ in range(n)]]]
                        continue

                    posIndices.append([index, [[tmpPos] for _ in range(n)]])

        return posIndices

    def random_position(self, indices, multiple=True):
        n = self.shl_args.train_args['accBagSize']
        positions = len(self.positions)
        posIndices = []

        if multiple:
            for index in indices:

                pos = [random.sample(range(positions), positions) for _ in range(n)]
                pos = list(map(list, zip(*pos)))
                pos = np.reshape(pos, (positions, n, 1)).tolist()

                if not len(posIndices):
                    posIndices = [[index, pos[j]] for j in range(positions)]
                    continue

                posIndices.extend([[index, pos[j]] for j in range(positions)])

        else:
            for index in indices:

                if not len(posIndices):
                    posIndices = [[index, [[random.randrange(positions)] for _ in range(n)]]]
                    continue

                posIndices.append([index, [[random.randrange(positions)] for _ in range(n)]])

        return posIndices

    def variable_position(self, indices, multiple=True):
        n = self.shl_args.train_args['accBagSize']
        positions = len(self.positions)
        posIndices = []

        # if probability = 1.0 then transitionalPosition() = randomPosition()
        # if probability = 0.0 then transitionalPosition() = samePosition()
        probability = 0.3

        if multiple:
            for index in indices:
                makeTransition = np.random.uniform(size=n - 1) < probability
                transitions = np.argwhere(makeTransition).squeeze(-1) + 1

                varyingPos = None
                previousTransition = 0
                for k in range(len(transitions) + 1):
                    if k == len(transitions):
                        transition = n
                    else:
                        transition = transitions[k]

                    stablePos = random.sample(range(positions), positions)
                    stablePos = [stablePos for _ in range(transition - previousTransition)]
                    if not varyingPos:
                        varyingPos = stablePos

                    else:
                        varyingPos.extend(stablePos)

                    previousTransition = transition

                pos = list(map(list, zip(*varyingPos)))
                pos = np.reshape(pos, (positions, n, 1)).tolist()

                if not len(posIndices):
                    posIndices = [[index, pos[j]] for j in range(positions)]
                    continue

                posIndices.extend([[index, pos[j]] for j in range(positions)])

        else:
            for index in indices:

                makeTransition = np.random.uniform(size=n - 1) < 0.5
                transitions = np.argwhere(makeTransition).squeeze(-1) + 1

                pos = None
                previousTransition = 0
                for k in range(len(transitions) + 1):
                    if k == len(transitions):
                        transition = n
                    else:
                        transition = transitions[k]

                    stablePos = random.sample(range(positions), 1)
                    stablePos = [stablePos for _ in range(transition - previousTransition)]
                    if not pos:
                        pos = stablePos

                    else:
                        pos.extend(stablePos)

                    previousTransition = transition

                if not len(posIndices):
                    posIndices = [[index, pos]]
                    continue

                posIndices.append([index, pos])

        return posIndices

    def assign_position(self, accTransfer=False, decisionTree=False):

        if self.testPosition != 'all':
            pos = self.positionsDict[self.testPosition]
            self.test_indices = self.same_position(self.test_indices, multiple=False, pos=pos)

        else:
            if self.testBagPositions == 'same':
                self.test_indices = self.same_position(self.test_indices, multiple=self.multipleTest)

            elif self.testBagPositions == 'random':
                self.test_indices = self.random_position(self.test_indices, multiple=self.multipleTest)

            elif self.testBagPositions == 'variable':
                self.test_indices = self.variable_position(self.test_indices, multiple=self.multipleTest)

        if self.trainPosition != 'all':
            pos = self.positionsDict[self.trainPosition]
            self.val_indices = self.same_position(self.val_indices, multiple=False, pos=pos)
            self.train_indices = self.same_position(self.train_indices, multiple=False, pos=pos)

        else:
            if self.trainBagPositions == 'same':
                self.val_indices = self.same_position(self.val_indices, multiple=self.multipleVal)

                if accTransfer or decisionTree:
                    self.train_indices = self.same_position(self.train_indices, multiple=self.multipleTrain)

                else:
                    multiple = self.multipleTrain if self.oversampling else False
                    self.train_indices = self.same_position(self.train_indices, multiple=multiple)

            elif self.trainBagPositions == 'random':
                self.val_indices = self.random_position(self.val_indices, multiple=self.multipleVal)

                if accTransfer or decisionTree:
                    self.train_indices = self.random_position(self.train_indices, multiple=self.multipleTrain)

                else:
                    multiple = self.multipleTrain if self.oversampling else False
                    self.train_indices = self.random_position(self.train_indices, multiple=multiple)

            elif self.trainBagPositions == 'variable':
                self.val_indices = self.variable_position(self.val_indices, multiple=self.multipleVal)

                if accTransfer or decisionTree:
                    self.train_indices = self.variable_position(self.train_indices, multiple=self.multipleTrain)

                else:
                    multiple = self.multipleTrain if self.oversampling else False
                    self.train_indices = self.variable_position(self.train_indices, multiple=multiple)

        self.testSize = len(self.test_indices)
        self.valSize = len(self.val_indices)
        self.trainSize = len(self.train_indices)

        if self.random:
            random.shuffle(self.test_indices)
            random.shuffle(self.val_indices)
            random.shuffle(self.train_indices)

    def get_gps_gaps(self):

        null_loc = []
        for i, loc_bag in enumerate(self.gpsBags[self.whichGPS]):
            if not loc_bag:
                null_loc.append(i)

        return null_loc

    def delete_gps_gaps(self, nulls):
        self.test_indices = [test_index for test_index in self.test_indices if test_index not in nulls]
        self.val_indices = [val_index for val_index in self.val_indices if val_index not in nulls]
        self.train_indices = [train_index for train_index in self.train_indices if train_index not in nulls]
        self.testSize = len(self.test_indices)
        self.valSize = len(self.val_indices)
        self.trainSize = len(self.train_indices)

    def toCSVs(self, filepath, motorized, includeGpsLoss=False):

        self.to_bags()
        self.split()
        self.assign_position(decisionTree=True)

        train = self.to_pandas(self.train_indices, motorized=motorized, includeGpsLoss=includeGpsLoss)
        test = self.to_pandas(self.test_indices, motorized=motorized, includeGpsLoss=includeGpsLoss)
        val = self.to_pandas(self.val_indices, motorized=motorized, includeGpsLoss=includeGpsLoss)

        train_val = pd.concat([train, val], ignore_index=False)
        train_val['in'] = train_val.index
        train_val.sort_values(['User', 'Day', 'Time', 'in'], inplace=True, ignore_index=True)

        try:
            os.makedirs(filepath)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        train_path = os.path.join(filepath, 'train' + str(self.testUser) + '.csv')
        val_path = os.path.join(filepath, 'val' + str(self.testUser) + '.csv')
        test_path = os.path.join(filepath, 'test' + str(self.testUser) + '.csv')
        train_val_path = os.path.join(filepath, 'train_val' + str(self.testUser) + '.csv')

        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        train_val.to_csv(train_val_path, index=False)

        return train, val, test, train_val

    def __call__(self, accTransfer=False,
                 gpsTransfer=False,
                 timeInfo=False,
                 batch_prefetch=True,
                 seed=1):

        self.initialize()

        self.init_transformers(
            accTransfer=accTransfer,
            gpsTransfer=gpsTransfer,
            timeInfo=timeInfo)

        self.to_bags()
        self.split(seed=seed)

        if gpsTransfer:
            nulls = self.get_gps_gaps()
            self.delete_gps_gaps(nulls)

        else:
            self.assign_position(accTransfer=accTransfer)

        train = self.to_generator(
            accTransfer=accTransfer,
            gpsTransfer=gpsTransfer,
            timeInfo=timeInfo)

        val = self.to_generator(
            is_val=True,
            accTransfer=accTransfer,
            gpsTransfer=gpsTransfer,
            timeInfo=timeInfo)

        test = self.to_generator(
            is_test=True,
            accTransfer=accTransfer,
            gpsTransfer=gpsTransfer,
            timeInfo=timeInfo)

        if batch_prefetch:
            return self.batch_and_prefetch(train, val, test)

        else:
            return train, val, test
