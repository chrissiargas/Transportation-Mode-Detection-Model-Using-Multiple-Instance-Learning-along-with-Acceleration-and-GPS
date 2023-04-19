import contextlib
import random
import pandas as pd
from configParser import Parser
from extractData import extractData
from buildData import buildData
from transformers import *

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

    def __init__(self, regenerate=False, verbose=False):

        parser = Parser()
        self.shl_args = parser.get_args()
        self.verbose = verbose

        if not regenerate:
            xData = extractData(self.shl_args)

            if not xData.found:

                bData = buildData(args=self.shl_args, verbose=verbose)

                bData()
                del bData

                self.acceleration, self.labels, self.location = xData()

            else:
                self.acceleration, self.labels, self.location = xData()

            del xData

        else:

            bData = buildData(args=self.shl_args,
                               verbose=verbose,
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

        self.complete = (self.shl_args.data_args['dataset'] == 'CompleteUser1')

        if self.complete:

            self.positions = ['Hips']

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

        if self.complete:
            self.whichGPS = 'Hips'

        else:
            self.whichGPS = 'Hand'

        self.testUser = self.shl_args.train_args['test_user']
        self.bags = self.labels.shape[0]
        self.n_labels = 8
        self.accBagSize = self.shl_args.train_args['accBagSize']
        self.random_position = self.shl_args.train_args['random_position']
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
        self.postprocessing = self.shl_args.train_args['post']
        self.syncing = self.shl_args.data_args['sync']
        self.gpsDuration = self.shl_args.data_args['locDuration']
        self.syncThreshold = self.shl_args.train_args['pair_threshold']
        self.paddingThreshold = self.gpsDuration
        self.useSpectro = self.shl_args.train_args['use_spectrograms']
        self.accShape = None
        self.accTimeShape = None
        self.accTfrm = None
        self.gpsSeriesShape = None
        self.gpsFeaturesShape = None
        self.gpsTimeShape = None
        self.gpsTfrm = None
        self.lbsTfrm = None
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

    def to_bags(self):

        tmp_location = []

        bag_map = {
            'acc': [[] for _ in range(self.bags)],
            'labels': [i for i in range(self.bags)],
            'gps': {}
        }

        if self.syncing == 'Past':
            gps_pivot = self.gpsDuration - 1
        elif self.syncing == 'Present':
            gps_pivot = self.gpsDuration // 2
        elif self.syncing == 'Future':
            gps_pivot = 0
        else:
            gps_pivot = self.gpsDuration - 1

        pivot = 0
        i = 0

        while pivot < self.bags:
            label = self.labels[pivot]
            sample_index = label[1]
            bag_map['acc'][i].append(sample_index)

            pivot += 1
            i += 1

        for pos_index, pos_name in enumerate(self.positions):

            bag_map['gps'][pos_name] = [[] for _ in range(self.bags)]

            if pos_name == self.whichGPS:

                start = 0
                tmp_n = 0

                for i, label in enumerate(self.labels):

                    bag_user = label[-3]
                    bag_day = label[-2]
                    bag_time = label[-1]

                    if i == 0 or bag_user != self.labels[i - 1][-3] or bag_day != self.labels[i - 1][-2]:

                        start += tmp_n
                        tmp_location = self.select_location(bag_user,
                                                            bag_day,
                                                            pos_index,
                                                            start)

                        tmp_n = tmp_location.shape[0]
                        begin = 0

                    else:
                        if len(bag_map['gps'][pos_name][i-1]):
                            begin = bag_map['gps'][pos_name][i-1][0]-start

                    found = False
                    for offset, location in enumerate(tmp_location[begin:]):

                        distance = np.abs(location[gps_pivot, -1] - bag_time)

                        if distance > self.syncThreshold:
                            if found:
                                break

                        else:
                            found = True
                            filled = np.sum(np.count_nonzero(location == -1, axis=1) == 0)
                            if filled >= self.paddingThreshold:
                                bag_map['gps'][pos_name][i].append(offset + begin + start)

            tmp_bag_map = []

            for i, bag in enumerate(bag_map['gps'][pos_name]):

                if len(bag) > 1:
                    divergence_list = []
                    bag_loc = self.location[pos_index][bag]
                    for loc in bag_loc:
                        timestamp = loc[gps_pivot, -1]
                        div = np.abs(self.labels[i, -1] - timestamp)
                        divergence_list.append(div)

                    min_indices = np.argpartition(divergence_list, 1)[:1]
                    tmp_bag_map.append([bag[index] for index in min_indices])

                elif len(bag) <= 1:
                    tmp_bag_map.append(bag)

            bag_map['gps'][pos_name] = tmp_bag_map

        return bag_map

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
                    accTransfer = accTransfer,
                    accMIL = self.accMIL
                )

            self.accShape = self.accTfrm.get_shape()
            if timeInfo:
                self.accTimeShape = self.accTfrm.get_time_shape()

        if not accTransfer:
            self.gpsTfrm = gpsTransformer(shl_args=self.shl_args,
                                          locTransfer=gpsTransfer)
            self.gpsSeriesShape, self.gpsFeaturesShape = self.gpsTfrm.get_shape
            if timeInfo:
                self.gpsTimeShape = self.gpsTfrm.get_time_shape()

        self.lbsTfrm = CategoricalTransformer()

        if gpsTransfer:
            self.inputShape = (self.gpsSeriesShape, self.gpsFeaturesShape)
            self.inputType = (tf.float64, tf.float64)
            if timeInfo:
                self.timeShape = (self.gpsTimeShape, (3))
                self.timeType = (tf.float64, tf.float64)

        elif accTransfer:
            self.inputShape = (self.accShape, self.accBagSize)
            self.inputType = (tf.float64, tf.int32)
            if timeInfo:
                self.timeShape = (self.accTimeShape, (3))
                self.timeType = (tf.float64, tf.float64)

        else:
            self.inputShape = (self.accShape, self.gpsSeriesShape, self.gpsFeaturesShape, self.accBagSize)
            self.inputType = (tf.float64, tf.float64, tf.float64, tf.int32)
            if timeInfo:
                self.timeShape = (self.accTimeShape, self.gpsTimeShape, (3))
                self.timeType = (tf.float64, tf.float64, tf.float64)

    def accFeatures(self, acceleration, position):

        if self.complete:
            positions = {
                'Hips': 0
            }

        else:
            positions = {
                'Torso': 0,
                'Hips': 1,
                'Bag': 2,
                'Hand': 3
            }

        pos_i = 3 * positions[position]

        magnitude = np.sqrt(np.sum(acceleration[:, :, pos_i:pos_i + 3] ** 2,
                                   axis=2))[0]

        var = np.var(magnitude)

        freq_acc = np.fft.fft(magnitude)
        freq_magnitude = np.power(np.abs(freq_acc), 2)

        coef1Hz = freq_magnitude[1]
        coef2Hz = freq_magnitude[2]
        coef3Hz = freq_magnitude[3]

        acc_features = [var, coef1Hz, coef2Hz, coef3Hz, acceleration[0][0][-3],acceleration[0][0][-2]]

        return acc_features

    def calc_haversine_dis(self, lat, lon, moment):

        point1 = (lat[moment - 1], lon[moment - 1])
        point2 = (lat[moment], lon[moment])
        return great_circle(point1, point2).m

    def calc_haversine_vel(self, lat, lon, t, moment):
        hvs_dis = self.calc_haversine_dis(lat, lon, moment)
        return 3600. * hvs_dis / (t[moment] - t[moment - 1])

    def haversine_velocity(self, pos_location, duration):
        time_signal = pos_location[:, -1]
        x_signal = pos_location[:, 1]
        y_signal = pos_location[:, 2]

        vel_signal = np.zeros((duration - 1))

        for moment in range(1, duration):
            vel_signal[moment - 1] = self.calc_haversine_vel(x_signal,
                                                             y_signal,
                                                             time_signal,
                                                             moment)

        return vel_signal

    def locFeatures(self, location):

        if self.complete:
            positions = {
                'Hips': 0
            }

        else:
            positions = {
                'Torso': 0,
                'Hips': 1,
                'Bag': 2,
                'Hand': 3
            }

        pos_name = self.whichGPS

        pos_location = location[positions[pos_name]]

        if np.size(pos_location):
            pos_location = pos_location[0]

        else:
            return -1, -1, -1, -1, -1

        for location_timestamp in pos_location:
            if location_timestamp[0] == -1.:
                return -1, -1, -1, -1, -1

        velocity = self.haversine_velocity(pos_location, self.gpsDuration)

        return velocity[0], pos_location[-1][-1], pos_location[-1][0], pos_location[-1][1], pos_location[-1][2]

    def to_pandas(self, is_val=False, is_test=False, motorized_class=True):

        if not is_test:

            if not is_val:
                indices = self.train_indices

            else:

                indices = self.val_indices

        else:

            indices = self.test_indices

        data = []

        for en, index in enumerate(indices):

            i = index[0]

            position = self.positions[index[1][0]]

            locBag = []
            for pos_i, pos in enumerate(self.positions):
                locBag.append(self.location[pos_i][self.gps_bags[pos][i]])

            LocFeature, LocTime, Accuracy, Lat, Lon = self.locFeatures(locBag)

            if LocFeature == -1:
                continue

            del locBag

            AccFeatures = self.accFeatures(self.acceleration[self.acc_bags[i]],
                                           position=position)

            Lb = self.labels[self.lbs_bags[i]][0] - 1
            if motorized_class:
                Lb = Lb if Lb < 4 else 4

            Time = self.labels[self.lbs_bags[i]][-1]

            if en == 0:
                data = [[LocFeature, Accuracy, Lat, Lon, *AccFeatures, LocTime, Time, position, Lb]]

            else:
                data.append([LocFeature, Accuracy, Lat, Lon, *AccFeatures, LocTime, Time, position, Lb])

        data = pd.DataFrame(data, columns=['vel', 'acc', 'lat', 'lon', 'var', '1Hz', '2Hz', '3Hz', 'User', 'Day', 'GPS Time', 'Label Time', 'Position', 'Label'])

        return data

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
                    i = index[0]
                    position = index[1]

                else:
                    i = index
                    position = None

                if not gpsTransfer:
                    if timeInfo:
                        accBag, accTime = self.accTfrm(self.acceleration[self.acc_bags[i]],
                                                       is_train=not (is_val or is_test),
                                                       position=position, timeInfo=timeInfo)
                    else:
                        try:
                            accBag = self.accTfrm(self.acceleration[self.acc_bags[i]], is_train = not (is_val or is_test),
                                                  position = position)
                        except:
                            print(i)

                            print(position)

                            # print(self.test_indices)

                if not accTransfer:
                    if timeInfo:
                        location = self.location[self.positions.index(self.whichGPS)][self.gps_bags[self.whichGPS][i]]
                        gpsSeries, gpsFeatures, gpsTime = self.gpsTfrm(location, timeInfo=timeInfo, is_train = not (is_val or is_test))
                    else:
                        location = self.location[self.positions.index(self.whichGPS)][self.gps_bags[self.whichGPS][i]]
                        gpsSeries, gpsFeatures = self.gpsTfrm(location, is_train = not (is_val or is_test))

                if timeInfo:
                    y, yTime = self.lbsTfrm(self.labels[self.lbs_bags[i]], timeInfo=timeInfo)
                else:
                    y = self.lbsTfrm(self.labels[self.lbs_bags[i]])

                if timeInfo:
                    if gpsTransfer:
                        yield (gpsSeries, gpsFeatures), y, (gpsTime, yTime)

                    elif accTransfer:
                        yield (accBag, position), y, (accTime, yTime)

                    else:
                        yield (accBag, gpsSeries, gpsFeatures, position), y, \
                            (accTime, gpsTime, yTime)

                else:
                    if gpsTransfer:
                        yield (gpsSeries, gpsFeatures), y

                    elif accTransfer:
                        yield (accBag, position), y

                    else:
                        yield (accBag, gpsSeries, gpsFeatures, position), y

        if timeInfo:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(self.inputType,
                              tf.float32,
                              self.timeType),
                output_shapes=(self.inputShape,
                               (self.n_labels),
                               self.timeShape)
            )

        else:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(self.inputType,
                              tf.float32),
                output_shapes=(self.inputShape,
                               (self.n_labels))
            )

    def batch_and_prefetch(self, train, val, test):

        return train.cache().shuffle(1000).repeat() \
                   .batch(batch_size=self.trainBatchSize) \
                   .prefetch(tf.data.AUTOTUNE), \
               val.cache().shuffle(1000).repeat() \
                   .batch(batch_size=self.valBatchSize).prefetch(tf.data.AUTOTUNE), \
               test.cache().shuffle(1000).repeat() \
                   .batch(batch_size=self.testBatchSize).prefetch(tf.data.AUTOTUNE)

    def split_train_val(self, dataIndices):

        seed = 1
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

            index = candidates.sample(random_state=seed).index[0]
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

    def split_train_val_test(self, seed=1):
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

    def postprocess(self, Model):

        predicted = []
        true = []
        true_sequence = []
        lengths = []
        length = 0
        positions = len(self.positions)
        inputs = [[[] for _ in range(4)] for _ in range(positions)]

        for index, (label, day, time, user) in enumerate(zip(self.labels[:-1, 0],
                                                             self.labels[:-1, -2],
                                                             self.labels[:-1, -1],
                                                             self.labels[:-1, -3])):

            if (self.complete and day in self.test_days) or (user == self.testUser and not self.complete):

                if self.random_position:
                    pos = [random.sample(range(positions), positions) for _ in range(self.accBagSize)]
                    pos = list(map(list, zip(*pos)))

                else:
                    pos = [[p for _ in range(self.accBagSize)] for p in range(positions)]

                for pos_j, position in enumerate(pos):

                    accBag = self.accTfrm(
                        self.acceleration[self.acc_bags[index]],
                        is_train=False,
                        position=position
                    )

                    location = self.location[self.positions.index(self.whichGPS)][self.gps_bags[self.whichGPS][index]]
                    gpsSeries, gpsFeatures = self.gpsTfrm(location, is_train=False)

                    if pos_j == 0:
                        length += 1
                        true_sequence.append(label - 1)

                    inputs[pos_j][0].append(accBag)
                    inputs[pos_j][1].append(gpsSeries)
                    inputs[pos_j][2].append(gpsFeatures)
                    inputs[pos_j][3].append(position)

                if self.labels[index + 1][-1] - time > self.syncThreshold \
                        or self.labels[index + 1][-2] != day or self.labels[index + 1][-3] != user:

                    for pos_j in range(positions):
                        true.extend(true_sequence)
                        predicted.extend(
                            np.argmax(Model.predict([np.array(inputs[pos_j][i]) for i in range(4)], verbose=0), axis=1))
                        lengths.append(length)

                    inputs = [[[] for _ in range(4)] for _ in range(positions)]
                    true_sequence = []
                    length = 0

        predicted = np.array(predicted)
        predicted = np.reshape(predicted, (-1, 1))
        true = np.array(true)
        return predicted, true, lengths

    def sensor_position(self, accTransfer=False, gpsTransfer=False, randomTree=False):

        pos_test_indices = []
        pos_val_indices = []
        pos_train_indices = []
        n = self.shl_args.train_args['accBagSize']

        if not self.random_position:
            positions = len(self.positions)

            if not gpsTransfer:

                for i, test_index in enumerate(self.test_indices):

                    if i == 0:
                        pos_test_indices = [[test_index, [pos for _ in range(n)]] for pos in range(positions)]
                        continue

                    pos_test_indices.extend([[test_index, [pos for _ in range(n)]] for pos in range(positions)])

                self.test_indices = pos_test_indices

                if not randomTree and self.random:
                    random.shuffle(self.test_indices)

                self.testSize = len(self.test_indices)

                for i, val_index in enumerate(self.val_indices):

                    if i == 0:
                        pos_val_indices = [[val_index, [pos for _ in range(n)]] for pos in range(positions)]
                        continue

                    pos_val_indices.extend([[val_index, [pos for _ in range(n)]] for pos in range(positions)])

                self.val_indices = pos_val_indices

                if not randomTree and self.random:
                    random.shuffle(self.val_indices)

                self.valSize = len(self.val_indices)

            if accTransfer or randomTree:

                for i, train_index in enumerate(self.train_indices):

                    if i == 0:
                        pos_train_indices = [[train_index, [pos for _ in range(n)]] for pos in range(positions)]
                        continue

                    pos_train_indices.extend([[train_index, [pos for _ in range(n)]] for pos in range(positions)])

                self.train_indices = pos_train_indices

                if not randomTree and self.random:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

            elif not gpsTransfer:

                for i, train_index in enumerate(self.train_indices):

                    pos = random.randrange(positions)

                    if i == 0:
                        pos_train_indices = [[train_index, [pos for _ in range(n)]]]
                        continue

                    pos_train_indices.append([train_index, [pos for _ in range(n)]])

                self.train_indices = pos_train_indices

                if not randomTree and self.random:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

        else:
            positions = len(self.positions)

            if not gpsTransfer:
                for i, test_index in enumerate(self.test_indices):

                    pos = [random.sample(range(positions), positions) for _ in range(n)]
                    pos = list(map(list, zip(*pos)))

                    if i == 0:
                        pos_test_indices = [[test_index, pos[j]] for j in range(positions)]
                        continue

                    pos_test_indices.extend([[test_index, pos[j]] for j in range(positions)])

                self.test_indices = pos_test_indices

                if not randomTree and self.random:
                    random.shuffle(self.test_indices)

                self.testSize = len(self.test_indices)

                for i, val_index in enumerate(self.val_indices):
                    pos = [random.sample(range(positions), positions) for _ in range(n)]
                    pos = list(map(list, zip(*pos)))

                    if i == 0:
                        pos_val_indices = [[val_index, pos[j]] for j in range(positions)]
                        continue

                    pos_val_indices.extend([[val_index, pos[j]] for j in range(positions)])

                self.val_indices = pos_val_indices

                if not randomTree and self.random:
                    random.shuffle(self.val_indices)

                self.valSize = len(self.val_indices)

            if accTransfer or randomTree:

                for i, train_index in enumerate(self.train_indices):
                    pos = [random.sample(range(positions), positions) for _ in range(n)]
                    pos = list(map(list, zip(*pos)))

                    if i == 0:
                        pos_train_indices = [[train_index, pos[j]] for j in range(positions)]
                        continue

                    pos_train_indices.extend([[train_index, pos[j]] for j in range(positions)])

                self.train_indices = pos_train_indices

                if not randomTree and self.random:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

            elif not gpsTransfer:

                for i, train_index in enumerate(self.train_indices):

                    if i == 0:
                        pos_train_indices = [[train_index, [random.randrange(positions) for _ in range(n)]]]
                        continue

                    pos_train_indices.append([train_index, [random.randrange(positions) for _ in range(n)]])

                self.train_indices = pos_train_indices

                if not randomTree and self.random:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

    def get_loc_nulls(self, bags):

        null_loc = []
        for i, loc_bag in enumerate(bags['gps'][self.whichGPS]):
            if not loc_bag:
                null_loc.append(i)

        return null_loc

    def delete_nulls(self, nulls):
        self.test_indices = [test_index for test_index in self.test_indices if test_index not in nulls]
        self.val_indices = [val_index for val_index in self.val_indices if val_index not in nulls]
        self.train_indices = [train_index for train_index in self.train_indices if train_index not in nulls]
        self.testSize = len(self.test_indices)
        self.valSize = len(self.val_indices)
        self.trainSize = len(self.train_indices)

    def __call__(self, accTransfer=False,
                 gpsTransfer=False,
                 randomTree=False,
                 timeInfo=False,
                 batch_prefetch=True,
                 seed=1):

        if randomTree:

            motorized = True
            bags = self.to_bags()
            self.acc_bags, self.lbs_bags, self.gps_bags = bags['acc'], bags['labels'], bags['gps']
            del bags

            self.split_train_val_test()
            self.sensor_position(randomTree=True)

            train = self.to_pandas(motorized_class=motorized)
            test = self.to_pandas(is_test=True, motorized_class=motorized)
            val = self.to_pandas(is_val=True, motorized_class=motorized)

            train_path = 'train' + str(self.testUser) + '.csv'
            val_path = 'val' + str(self.testUser) + '.csv'
            test_path = 'test' + str(self.testUser) + '.csv'

            train.to_csv(train_path, index=False)
            val.to_csv(val_path, index=False)
            test.to_csv(test_path, index=False)

            return train, val, test

        if not randomTree:

            self.init_transformers(
                accTransfer=accTransfer,
                gpsTransfer=gpsTransfer,
                timeInfo=timeInfo
            )

            bags = self.to_bags()

            self.acc_bags, self.lbs_bags, self.gps_bags = bags['acc'], bags['labels'], bags['gps']

            self.split_train_val_test(seed=seed)

            if gpsTransfer:
                nulls = self.get_loc_nulls(bags)
                del bags
                self.delete_nulls(nulls)

            self.sensor_position(accTransfer=accTransfer,
                                 gpsTransfer=gpsTransfer)

            train = self.to_generator(
                accTransfer=accTransfer,
                gpsTransfer=gpsTransfer,
                timeInfo=timeInfo
            )

            val = self.to_generator(
                is_val=True,
                accTransfer=accTransfer,
                gpsTransfer=gpsTransfer,
                timeInfo=timeInfo
            )

            test = self.to_generator(
                is_test=True,
                accTransfer=accTransfer,
                gpsTransfer=gpsTransfer,
                timeInfo=timeInfo
            )

            if batch_prefetch:
                return self.batch_and_prefetch(train, val, test)

            else:
                return train, val, test
