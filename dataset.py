import contextlib
import random

import numpy as np
import pandas as pd
import sklearn.metrics

from configParser import Parser
from extractData import extractData
from preprocessing import preprocessData
import tensorflow as tf
from transformers import *
from sklearn.model_selection import train_test_split
from collections import Counter, OrderedDict
from tqdm import tqdm
from hmmlearn import hmm

np.set_printoptions(precision=14)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class SignalsDataset:

    def __init__(self,
                 regenerate=False,
                 deleteFolders=False,
                 verbose=False
                 ):

        parser = Parser()
        self.shl_args = parser.get_args()
        self.random_position = self.shl_args.train_args['random_position']
        self.verbose = verbose

        if not regenerate:
            exData = extractData(self.shl_args)

            if not exData.found:

                ppData = preprocessData(args=self.shl_args,
                                        verbose=verbose)

                ppData()
                del ppData

                self.acceleration, \
                self.labels, \
                self.location = exData(delete_dst=deleteFolders,
                                       delete_tmp=deleteFolders,
                                       delete_final=deleteFolders)




            else:
                self.acceleration, \
                self.labels, \
                self.location = exData(delete_dst=deleteFolders,
                                       delete_tmp=deleteFolders,
                                       delete_final=deleteFolders)

            del exData


        else:

            ppData = preprocessData(args=self.shl_args,
                                    verbose=verbose,
                                    delete_dst=True,
                                    delete_tmp=True,
                                    delete_final=True,
                                    delete_filter=True)

            ppData()

            del ppData

            exData = extractData(self.shl_args)

            self.acceleration, \
            self.labels, \
            self.location = exData(delete_dst=deleteFolders,
                                   delete_tmp=deleteFolders,
                                   delete_final=deleteFolders)

            del exData

        self.complete = (self.shl_args.data_args['dataset'] == 'CompleteUser1')
        self.bags = self.labels.shape[0]
        self.trainBatchSize = self.shl_args.train_args['trainBatchSize']
        self.valBatchSize = self.shl_args.train_args['valBatchSize']
        self.testBatchSize = self.shl_args.train_args['testBatchSize']

        self.n_labels = 8

        self.complete = (self.shl_args.data_args['dataset'] == 'CompleteUser1')

        if self.complete:

            self.pos = ['Hips']

            self.acc_positions = ['Hips']

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

            self.pos = [
                'Torso',
                'Hips',
                'Bag',
                'Hand'
            ]

            self.acc_positions = ['Torso', 'Hips', 'Bag', 'Hand']

        if self.complete:
            self.gps_pos = 'Hips'

        else:
            self.gps_pos = 'Hand'

    def to_bags(self):

        bag_map = {
            'acc_bags': [[] for _ in range(self.bags)],
            'lbs_bags': [i for i in range(self.bags)],
            'loc_bags': {}
        }

        sync = self.shl_args.data_args['sync']
        gps_duration = self.shl_args.data_args['locDuration']

        if sync == 'Past':
            gps_pivot = gps_duration - 1
        elif sync == 'Present':
            gps_pivot = gps_duration // 2
        elif sync == 'Future':
            gps_pivot = 0

        pivot = 0
        i = 0

        while pivot < self.bags:
            label = self.labels[pivot]
            sample_index = label[1]
            bag_map['acc_bags'][i].append(sample_index)

            pivot += 1
            i += 1

        for pos_index, pos_name in enumerate(self.pos):

            bag_map['loc_bags'][pos_name] = [[] for _ in range(self.bags)]

            if pos_name == self.gps_pos:

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
                        if len(bag_map['loc_bags'][pos_name][i-1]):
                            begin = bag_map['loc_bags'][pos_name][i-1][0]-start

                    found = False
                    for offset, location in enumerate(tmp_location[begin:]):

                        distance = np.abs(location[gps_pivot, -1] - bag_time)

                        if distance > self.shl_args.train_args['pair_threshold']:
                            if found:
                                break

                        else:
                            found = True
                            filled = np.sum(np.count_nonzero(location == -1, axis=1) == 0)
                            if filled >= self.shl_args.train_args['padding_threshold']:
                                bag_map['loc_bags'][pos_name][i].append(offset + begin + start)

            # offset = 0
            # last = False
            # bag_map['loc_bags'][pos_name] = [[] for _ in range(self.bags)]
            # start = 0
            # tmp_n = 0
            #
            # for i, label in enumerate(self.labels):
            #
            #     if last and offset == tmp_n:
            #         break
            #
            #     bag_user = label[-3]
            #     bag_day = label[-2]
            #     bag_time = label[-1]
            #
            #     if i == 0 or bag_user != self.labels[i - 1][-3] or \
            #             bag_day != self.labels[i - 1][-2]:
            #
            #         if bag_user == 3 and bag_day == 3:
            #             last = True
            #
            #         start += tmp_n
            #         tmp_location = self.select_location(bag_user,
            #                                             bag_day,
            #                                             pos_index,
            #                                             start)
            #
            #         offset = 0
            #         tmp_n = tmp_location.shape[0]
            #
            #     begin = offset
            #     while offset < tmp_n:
            #         if bag_time < tmp_location[offset, 0, -1]:
            #             offset = begin
            #             break
            #
            #         elif tmp_location[offset, 0, -1] <= bag_time <= tmp_location[offset, -1, -1]:
            #
            #             filled = np.sum(np.count_nonzero(tmp_location[offset] == -1, axis=1) == 0)
            #
            #             if filled < self.shl_args.train_args['padding_threshold']:
            #
            #                 pass
            #
            #             distance = np.abs(tmp_location[offset, locPos, -1] - bag_time)
            #             if distance > self.shl_args.train_args['pair_threshold']:
            #                 pass
            #             else:
            #                 bag_map['loc_bags'][pos_name][i].append(offset + start)
            #
            #             offset += 1
            #
            #         elif offset == tmp_n - 1 or tmp_location[offset, -1, -1] < bag_time < tmp_location[
            #             offset + 1, 0, -1]:
            #             offset = begin
            #             break
            #
            #         elif bag_time >= tmp_location[offset + 1, 0, -1]:
            #             offset += 1
            #             begin += 1

            tmp_bag_map = []

            for i, bag in enumerate(bag_map['loc_bags'][pos_name]):

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

            bag_map['loc_bags'][pos_name] = tmp_bag_map

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

    def init_transformers(self, accTransfer=False, locTransfer=False, timeInfo=False):

        use_spectro = self.shl_args.train_args['use_spectrograms']
        accMIL = self.shl_args.train_args['seperate_MIL']
        n_instances = self.shl_args.train_args['accBagSize']

        if not locTransfer:

            if use_spectro:
                self.accTfrm = SpectogramAccTransform(
                    self.shl_args,
                    accTransfer=accTransfer,
                    accMIL=accMIL
                )

            else:
                self.accTfrm = TemporalAccTransform(
                    self.shl_args,
                    accTransfer = accTransfer,
                    accMIL = accMIL
                )

            self.accShape = self.accTfrm.get_shape()
            if timeInfo:
                self.accTimeShape = self.accTfrm.get_time_shape()

        if not accTransfer:
            self.locTfrm = TemporalLocationTransform(shl_args=self.shl_args,
                                                     locTransfer=locTransfer)
            self.locSignalsShape, self.locFeaturesShape = self.locTfrm.get_shape()
            if timeInfo:
                self.locTimeShape = self.locTfrm.get_time_shape()

        self.lbsTfrm = CategoricalTransform()

        if locTransfer:
            self.inputShape = (self.locSignalsShape, self.locFeaturesShape)
            self.inputType = (tf.float64, tf.float64)
            if timeInfo:
                self.timeShape = (self.locTimeShape, (3))
                self.timeType = (tf.float64, tf.float64)

        elif accTransfer:
            self.inputShape = (self.accShape, n_instances)
            self.inputType = (tf.float64, tf.int32)
            if timeInfo:
                self.timeShape = (self.accTimeShape, (3))
                self.timeType = (tf.float64, tf.float64)

        else:
            self.inputShape = (self.accShape, self.locSignalsShape, self.locFeaturesShape, n_instances)
            self.inputType = (tf.float64, tf.float64, tf.float64, tf.int32)
            if timeInfo:
                self.timeShape = (self.accTimeShape, self.locTimeShape, (3))
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

        pos_name = self.gps_pos

        pos_location = location[positions[pos_name]]

        if np.size(pos_location):
            pos_location = pos_location[0]

        else:
            return -1, -1, -1, -1, -1

        for location_timestamp in pos_location:
            if location_timestamp[0] == -1.:
                print('a')
                print(pos_location)
                return -1, -1, -1, -1, -1

        velocity = self.haversine_velocity(pos_location, self.shl_args.data_args['locDuration'])
        # print(pos_location)
        # print(velocity)

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


            position = self.acc_positions[index[1][0]]

            # print(i)
            # print(position)

            locBag = []
            for pos_i, pos in enumerate(self.pos):
                locBag.append(self.location[pos_i][self.loc_bags[pos][i]])

            LocFeature, LocTime, Accuracy, Lat, Lon = self.locFeatures(locBag)

            if LocFeature == -1:
                continue

            del locBag

            AccFeatures = self.accFeatures(self.acceleration[self.acc_bags[i]],
                                           position=position)

            # print(self.labels[self.lbs_bags[i]])

            Lb = self.labels[self.lbs_bags[i]][0] - 1
            if motorized_class:
                Lb = Lb if Lb < 4 else 4

            Time = self.labels[self.lbs_bags[i]][-1]


            if en == 0:
                data = [[LocFeature, Accuracy, Lat, Lon, *AccFeatures, LocTime, Time, position, Lb]]

            else:
                data.append([LocFeature, Accuracy, Lat, Lon, *AccFeatures, LocTime, Time, position, Lb])



        #
        # if DHMM:
        #     df_data = [pd.DataFrame(data[i::4], columns=['vel', 'acc var', 'acc DFT 1Hz', 'acc DFT 2Hz', 'acc DFT 3Hz', 'GPS time', 'Label time'],
        #                             dtype=float) for i in range(4)]
        #
        #     df_labels = [pd.DataFrame(labels[i::4], columns=['label'], dtype=int) for i in range(4)]
        #
        #     time = pd.DataFrame(time[0::4], columns=['time'], dtype=float)
        #
        #     dT_threshold = 2000
        #     time['dT'] = time['time'].diff().abs()
        #     split = time.index[time['dT'] > dT_threshold].tolist()
        #
        #     last_check = 0
        #     split_data = []
        #     split_lbs = []
        #     for index in split:
        #         split_data.extend([df_data[i].loc[last_check:index - 1] for i in range(4)])
        #         split_lbs.extend([df_labels[i].loc[last_check:index - 1] for i in range(4)])
        #         last_check = index
        #
        #     if not is_test:
        #         classes = [i for i in range(5)] if motorized_class else [i for i in range(self.n_labels)]
        #
        #         transition_mx = None
        #         for i, seq in enumerate(split_lbs):
        #             seq_ = seq
        #             seq_['label_'] = seq_.shift(-1)
        #
        #             groups = seq_.groupby(['label', 'label_'])
        #             counts = {i[0]: len(i[1]) for i in groups}
        #
        #             matrix = pd.DataFrame()
        #
        #             for x in classes:
        #                 matrix[x] = pd.Series([counts.get((x, y), 0) for y in classes], index=classes)
        #
        #             if i != 0:
        #                 transition_mx = transition_mx.add(matrix)
        #
        #             else:
        #                 transition_mx = matrix
        #
        #         transition_mx["sum"] = transition_mx.sum(axis=1)
        #         transition_mx = transition_mx.div(transition_mx["sum"], axis=0)
        #         transition_mx = transition_mx.drop(columns=['sum'])
        #         transition_mx = transition_mx.values.tolist()
        #
        #         return split_data, split_lbs, transition_mx
        #
        #     return split_data, split_lbs


        data = pd.DataFrame(data, columns=['vel', 'acc', 'lat', 'lon', 'var', '1Hz', '2Hz', '3Hz', 'User', 'Day', 'GPS Time', 'Label Time', 'Position', 'Label'])

        return data

    def to_generator(self, is_val=False, is_test=False, accTransfer=False, locTransfer=False, timeInfo=False):

        if not is_test:
            if not is_val:
                indices = self.train_indices

            else:
                indices = self.val_indices

        else:
            indices = self.test_indices

        def gen():

            for index in indices:

                if not locTransfer:
                    i = index[0]
                    position = index[1]

                else:
                    i = index
                    position = None

                if not locTransfer:
                    if timeInfo:
                        transformedAccBag, accTime = self.accTfrm(self.acceleration[self.acc_bags[i]]
                                                         , is_train=not (is_val or is_test),
                                                         position=position, timeInfo=timeInfo)
                    else:
                        transformedAccBag = self.accTfrm(self.acceleration[self.acc_bags[i]]
                                                         ,is_train = not (is_val or is_test),
                                                         position = position)

                if not accTransfer:
                    if timeInfo:
                        location = self.location[self.pos.index(self.gps_pos)][self.loc_bags[self.gps_pos][i]]
                        transformedLocSignalsBag, transformedLocFeaturesBag, locTime = self.locTfrm(location, timeInfo=timeInfo,
                                                                                           is_train = not (is_val or is_test))
                    else:
                        location = self.location[self.pos.index(self.gps_pos)][self.loc_bags[self.gps_pos][i]]
                        transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(location,
                                                                                           is_train = not (is_val or is_test))

                if timeInfo:
                    y, yTime = self.lbsTfrm(self.labels[self.lbs_bags[i]], timeInfo=timeInfo)
                else:
                    y = self.lbsTfrm(self.labels[self.lbs_bags[i]])

                if timeInfo:
                    if locTransfer:
                        yield (transformedLocSignalsBag, transformedLocFeaturesBag), y, (locTime, yTime)

                    elif accTransfer:
                        yield (transformedAccBag, position), y, (accTime, yTime)

                    else:
                        yield (transformedAccBag, transformedLocSignalsBag, transformedLocFeaturesBag, position), y, \
                            (accTime, locTime, yTime)

                else:
                    if locTransfer:
                        yield (transformedLocSignalsBag, transformedLocFeaturesBag), y

                    elif accTransfer:
                        yield (transformedAccBag, position), y

                    else:
                        yield (transformedAccBag, transformedLocSignalsBag, transformedLocFeaturesBag, position), y

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

        val_percentage = self.shl_args.train_args['val_percentage']
        randomize = self.shl_args.train_args['randomize']
        seed = 1


        originalIndices = dataIndices
        dataIndices = pd.DataFrame(dataIndices, columns=['index', 'user_label'])

        count = dataIndices['user_label'].value_counts()

        val_count = count * val_percentage
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

        if randomize:
            random.shuffle(self.val_indices)
            random.shuffle(self.train_indices)

    def split_train_val_test(self, seed=1):

        randomize = self.shl_args.train_args['randomize']
        test_user = self.shl_args.train_args['test_user']
        self.test_indices = []
        self.testSize = 0

        if self.complete:
            self.train_indices = []
            self.trainSize = 0

            self.val_indices = []
            self.valSize = 0

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

                if index not in self.run_indices:
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

                if user == test_user:
                    self.testSize += 1
                    self.test_indices.append(index)

                else:
                    train_val_size += 1
                    train_val_indices.append([index, user * 10 + label])

            if randomize:
                random.shuffle(self.test_indices)

            self.split_train_val(train_val_indices)

    def postprocess(self, Model, fit=False):

        if fit:

            bagged = self.shl_args.data_args['bagging']
            size = self.shl_args.data_args['accBagSize'] if bagged else None
            stride = self.shl_args.data_args['accBagStride'] if bagged else None

            trans_threshold = self.shl_args.train_args['transition_threshold']

            predicted = []
            true = []
            true_sequence = []

            length = 0

            inputs = [[[] for _ in range(3)] for _ in range(len(self.acc_positions))]
            classes = [i for i in range(self.n_labels)]

            transition_mx = pd.DataFrame(
                np.zeros(shape=(self.n_labels, self.n_labels))
            )

            for index, (label, day, time, user) in enumerate(zip(self.labels[:-1, 0],
                                                                 self.labels[:-1, -2],
                                                                 self.labels[:-1, -1],
                                                                 self.labels[:-1, -3])):

                if index not in self.run_indices:

                    if (self.complete and day in self.val_days) or (not self.complete and user == self.val_user):

                        for pos_j, position in enumerate(self.acc_positions):

                            transformedAccSignalsBag = self.accTfrm(
                                self.acceleration[self.acc_bags[index]]
                                , is_train=False,
                                position=position, bagged=bagged,
                                size=size, stride=stride
                            )

                            locBag = []
                            for pos_i, position in enumerate(self.pos):
                                locBag.append(self.location[pos_i][self.loc_bags[position][index]])

                            transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(locBag, is_train=False)

                            if pos_j == 0:
                                length += 1
                                true_sequence.append(label - 1)

                            inputs[pos_j][0].append(transformedAccSignalsBag)
                            inputs[pos_j][1].append(transformedLocSignalsBag)
                            inputs[pos_j][2].append(transformedLocFeaturesBag)

                        if self.labels[index + 1][-1] - time > 3 * trans_threshold \
                                or self.labels[index + 1][-2] != day or self.labels[index + 1][-3] != user:

                            for pos_j in range(len(self.acc_positions)):
                                true.extend(true_sequence)
                                predicted.extend(
                                    np.argmax(Model.call([np.array(inputs[pos_j][i]) for i in range(3)]), axis=1))

                            true_sequence = pd.DataFrame(true_sequence, columns=['label'])
                            true_sequence['label_'] = true_sequence.shift(-1)

                            groups = true_sequence.groupby(['label', 'label_'])
                            counts = {i[0]: len(i[1]) for i in groups}

                            matrix = pd.DataFrame()

                            for x in classes:
                                matrix[x] = pd.Series([counts.get((x, y), 0) for y in classes], index=classes)

                            transition_mx = transition_mx.add(matrix)

                            inputs = [[[] for _ in range(3)] for _ in range(len(self.acc_positions))]
                            true_sequence = []
                            length = 0

            confusion_mx = pd.DataFrame(sklearn.metrics.confusion_matrix(predicted, true))

            return transition_mx, confusion_mx

        else:
            test_user = self.shl_args.train_args['test_user']

            bagged = self.shl_args.data_args['bagging']
            size = self.shl_args.train_args['accBagSize'] if bagged else None

            trans_threshold = self.shl_args.train_args['transition_threshold']

            predicted = []
            true = []
            true_sequence = []
            lengths = []

            length = 0

            n_pos = len(self.acc_positions)

            inputs = [[[] for _ in range(4)] for _ in range(len(self.acc_positions))]



            for index, (label, day, time, user) in enumerate(zip(self.labels[:-1, 0],
                                                                 self.labels[:-1, -2],
                                                                 self.labels[:-1, -1],
                                                                 self.labels[:-1, -3])):

                if index not in self.run_indices:

                    if (self.complete and day in self.test_days) or (user == test_user and not self.complete):

                        if self.random_position:
                            pos = [random.sample(range(n_pos), n_pos) for _ in range(size)]
                            pos = list(map(list, zip(*pos)))

                        else:
                            pos = [[p for _ in range(size)] for p in range(n_pos)]

                        for pos_j, position in enumerate(pos):

                            transformedAccSignalsBag = self.accTfrm(
                                self.acceleration[self.acc_bags[index]]
                                ,is_train=False,
                                position=position
                            )


                            location = self.location[self.pos.index(self.gps_pos)][self.loc_bags[self.gps_pos][index]]
                            transformedLocSignalsBag, transformedLocFeaturesBag = self.locTfrm(location, is_train=False)

                            if pos_j == 0:
                                length += 1
                                true_sequence.append(label - 1)

                            inputs[pos_j][0].append(transformedAccSignalsBag)
                            inputs[pos_j][1].append(transformedLocSignalsBag)
                            inputs[pos_j][2].append(transformedLocFeaturesBag)
                            inputs[pos_j][3].append(position)

                        if self.labels[index + 1][-1] - time > trans_threshold \
                                or self.labels[index + 1][-2] != day or self.labels[index + 1][-3] != user:

                            for pos_j in range(n_pos):
                                true.extend(true_sequence)
                                # print(length)
                                predicted.extend(
                                    np.argmax(Model.predict([np.array(inputs[pos_j][i]) for i in range(4)], verbose=0), axis=1))
                                lengths.append(length)

                            # print(true[-length:])
                            # print(predicted[-length:])
                            inputs = [[[] for _ in range(4)] for _ in range(n_pos)]
                            true_sequence = []
                            length = 0

            predicted = np.array(predicted)
            predicted = np.reshape(predicted, (-1, 1))
            true = np.array(true)
            return predicted, true, lengths

    def sensor_position(self, accTransfer=False, locTransfer=False, randomTree=False):

        n = self.shl_args.train_args['accBagSize']
        randomize = self.shl_args.train_args['randomize']

        if not self.random_position:
            positions = len(self.acc_positions)

            if not locTransfer:

                for i, test_index in enumerate(self.test_indices):
                    if i == 0:
                        pos_test_indices = [[test_index, [pos for _ in range(n)]] for pos in range(positions)]
                        continue

                    pos_test_indices.extend([[test_index, [pos for _ in range(n)]] for pos in range(positions)])

                self.test_indices = pos_test_indices

                if not randomTree and randomize:
                    random.shuffle(self.test_indices)

                self.testSize = len(self.test_indices)

                for i, val_index in enumerate(self.val_indices):
                    if i == 0:
                        pos_val_indices = [[val_index, [pos for _ in range(n)]] for pos in range(positions)]
                        continue

                    pos_val_indices.extend([[val_index, [pos for _ in range(n)]] for pos in range(positions)])

                self.val_indices = pos_val_indices

                if not randomTree and randomize:
                    random.shuffle(self.val_indices)

                self.valSize = len(self.val_indices)


            if accTransfer or randomTree:

                for i, train_index in enumerate(self.train_indices):
                    if i == 0:
                        pos_train_indices = [[train_index, [pos for _ in range(n)]] for pos in range(positions)]
                        continue

                    pos_train_indices.extend([[train_index, [pos for _ in range(n)]] for pos in range(positions)])

                self.train_indices = pos_train_indices

                if not randomTree and randomize:
                    random.shuffle(self.train_indices)
                self.trainSize = len(self.train_indices)

            elif not locTransfer:
                for i, train_index in enumerate(self.train_indices):
                    pos = random.randrange(positions)
                    if i == 0:
                        pos_train_indices = [[train_index, [pos for _ in range(n)]]]
                        continue

                    pos_train_indices.append([train_index, [pos for _ in range(n)]])

                self.train_indices = pos_train_indices

                if not randomTree and randomize:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

        else:
            positions = len(self.acc_positions)

            if not locTransfer:

                for i, test_index in enumerate(self.test_indices):
                    pos = [random.sample(range(positions), positions) for _ in range(n)]
                    pos = list(map(list, zip(*pos)))
                    if i == 0:
                        pos_test_indices = [[test_index, pos[j]] for j in range(positions)]
                        continue

                    pos_test_indices.extend([[test_index, pos[j]] for j in range(positions)])

                self.test_indices = pos_test_indices

                if not randomTree and randomize:
                    random.shuffle(self.test_indices)

                self.testSize = len(self.test_indices)

                for i, val_index in enumerate(self.val_indices):
                    pos = [random.sample(range(positions), positions) for _ in range(n)]
                    pos = list(map(list, zip(*pos)))

                    if i == 0:
                        pos_val_indices = [[val_index, pos[i]] for i in range(positions)]
                        continue

                    pos_val_indices.extend([[val_index, pos[i]] for i in range(positions)])

                self.val_indices = pos_val_indices

                if not randomTree and randomize:
                    random.shuffle(self.val_indices)

                self.valSize = len(self.val_indices)

            if accTransfer or randomTree:
                for i, train_index in enumerate(self.train_indices):
                    pos = [random.sample(range(positions), positions) for _ in range(n)]
                    pos = list(map(list, zip(*pos)))

                    if i == 0:
                        pos_train_indices = [[train_index, pos[i]] for i in range(positions)]
                        continue

                    pos_train_indices.extend([[train_index, pos[i]] for i in range(positions)])

                self.train_indices = pos_train_indices

                if not randomTree and randomize:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

            elif not locTransfer:
                for i, train_index in enumerate(self.train_indices):

                    if i == 0:
                        pos_train_indices = [[train_index, [random.randrange(positions) for _ in range(n)]]]
                        continue

                    pos_train_indices.append([train_index, [random.randrange(positions) for _ in range(n)]])

                self.train_indices = pos_train_indices

                if not randomTree and randomize:
                    random.shuffle(self.train_indices)

                self.trainSize = len(self.train_indices)

    def get_loc_nulls(self, bags):

        if self.shl_args.train_args['gpsPosition'] == None:
            if self.complete:
                position = 'Hips'

            else:
                position = 'Hand'

        else:
            position = self.shl_args.train_args['gpsPosition']

        null_loc = []
        for i, loc_bag in enumerate(bags['loc_bags'][position]):
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
                 locTransfer=False,
                 randomTree=False,
                 timeInfo=False,
                 batch_prefetch=True,
                 seed=1):

        if randomTree:

            motorized = self.shl_args.train_args['motorized']

            bags = self.to_bags()

            self.acc_bags, self.lbs_bags, self.loc_bags = bags['acc_bags'], bags['lbs_bags'], bags['loc_bags']

            del bags

            self.run_indices = []
            self.split_train_val_test(DHMM=DHMM)

            self.sensor_position(randomTree=True, DHMM=DHMM)

            test_user =  self.shl_args.train_args['test_user']

            train = self.to_pandas(motorized_class=motorized)

            test = self.to_pandas(is_test=True, motorized_class=motorized)

            val = self.to_pandas(is_val=True, motorized_class=motorized)

            train_path = 'train' + str(test_user) + '.csv'
            val_path = 'val' + str(test_user) + '.csv'
            test_path = 'test' + str(test_user) + '.csv'

            train.to_csv(train_path, index=False)

            val.to_csv(val_path, index=False)

            test.to_csv(test_path, index=False)

            return train, val, test

        if not randomTree:

            self.init_transformers(
                accTransfer=accTransfer,
                locTransfer=locTransfer,
                timeInfo=timeInfo
            )

            bags = self.to_bags()

            self.acc_bags, self.lbs_bags, self.loc_bags = bags['acc_bags'], bags['lbs_bags'], bags['loc_bags']

            self.split_train_val_test(seed=seed)

            if locTransfer:
                nulls = self.get_loc_nulls(bags)
                del bags
                self.delete_nulls(nulls)

            self.sensor_position(accTransfer=accTransfer,
                                 locTransfer=locTransfer)

            train = self.to_generator(
                is_val=False,
                accTransfer=accTransfer,
                locTransfer=locTransfer,
                timeInfo=timeInfo
            )
            val = self.to_generator(
                is_val=True,
                accTransfer=accTransfer,
                locTransfer=locTransfer,
                timeInfo=timeInfo
            )
            test = self.to_generator(
                is_test=True,
                accTransfer=accTransfer,
                locTransfer=locTransfer,
                timeInfo=timeInfo
            )

            if batch_prefetch:
                return self.batch_and_prefetch(train, val, test)

            else:
                return train, val, test
