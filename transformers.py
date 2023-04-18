import copy
import numpy as np
from geopy.distance import great_circle
from scipy.interpolate import CubicSpline
import tensorflow as tf
from scipy import signal
from scipy import interpolate


def GenerateRandomCurves(N, sigma=0.2, knot=4, xyz=False):
    if not xyz:
        xx = ((np.arange(0, N, (N - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        x_range = np.arange(N)
        cs_x = CubicSpline(xx[:], yy[:])
        return np.array([cs_x(x_range)]).transpose()

    else:
        xx = (np.ones((3, 1)) * (np.arange(0, N, (N - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, 3))
        x_range = np.arange(N)
        cs_x = CubicSpline(xx[:, 0], yy[:, 0])
        cs_y = CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = CubicSpline(xx[:, 2], yy[:, 2])
        return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def DistortTimesteps(N, sigma=0.2, xyz=False):
    if not xyz:
        tt = GenerateRandomCurves(N, sigma)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [(N - 1) / tt_cum[-1]]
        tt_cum[:] = tt_cum[:] * t_scale
        return tt_cum

    else:
        tt = GenerateRandomCurves(N, sigma, xyz=xyz)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [(N - 1) / tt_cum[-1, 0], (N - 1) / tt_cum[-1, 1], (N - 1) / tt_cum[-1, 2]]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
        tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
        return tt_cum


def DA_TimeWarp(N, sigma=0.2, xyz=False):
    if not xyz:
        tt_new = DistortTimesteps(N, sigma)
        tt_new = np.squeeze(tt_new)
        x_range = np.arange(N)
        return tt_new, x_range


    else:
        tt_new = DistortTimesteps(N, sigma, xyz)
        x_range = np.arange(N)
        return tt_new, x_range


def DA_Permutation(N, nPerm=4, minSegLength=10, xyz=False):
    if not xyz:

        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, N - minSegLength, nPerm - 1))
            segs[-1] = N
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False

        return segs


def permutate(signal, N, segs, idx, nPerm=4):
    pp = 0
    X_new = np.zeros(N)

    for ii in range(nPerm):
        x_temp = signal[segs[idx[ii]]:segs[idx[ii] + 1]]
        X_new[pp:pp + len(x_temp)] = x_temp
        pp += len(x_temp)

    return X_new


def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axis_angle_to_rotation_matrix_3d_vectorized(axis, angle))


def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    x, y, z = axes

    n = np.sqrt(x * x + y * y + z * z)
    x = x / n
    y = y / n
    z = z / n

    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    return np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])


class CategoricalTransform:
    def __init__(self):
        self.encoding = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            self.encoding[i, i] = 1.

    def __call__(self, label, timeInfo = False):
        if timeInfo:
            return self.encoding[label[0] - 1], label[-3:]
        else:
            return self.encoding[label[0] - 1]


class TemporalAccTransform:
    def __init__(self,
                 shl_args=None,
                 accTransfer=False,
                 accMIL=False,
                 processing=False):

        self.shl_args = shl_args
        self.transfer = accTransfer
        self.MIL = accMIL
        self.processing = processing

        if shl_args.train_args['acc_signals'] is None:
            self.snl = ['Acc_norm']

        else:
            self.snl = shl_args.train_args['acc_signals']

        self.base_acc_signals = {
            'Acc_x': 0,
            'Acc_y': 1,
            'Acc_z': 2
        }

        self.sec_acc_signals = [
            'Acc_norm',
            'Jerk'
        ]

        self.acc_aug_list = shl_args.train_args['acc_norm_augmentation']
        self.acc_aug_params = shl_args.train_args['acc_norm_aug_params']
        self.acc_xyz_aug_list = shl_args.train_args['acc_xyz_augmentation']
        self.acc_xyz_aug_params = shl_args.train_args['acc_xyz_aug_params']

        self.freq = int(1. / shl_args.data_args['accSamplingRate'])

        self.length = shl_args.train_args['accDuration']
        self.stride = self.shl_args.train_args['accBagStride']
        self.channels = len(shl_args.train_args['acc_signals'])
        self.n_bags = self.shl_args.train_args['accBagSize']

    def get_shape(self):
        if self.transfer and not self.MIL:
            return self.length, self.channels

        else:
            return self.n_bags, self.length, self.channels

    def get_time_shape(self):
        if self.transfer and not self.MIL:
            return self.length, 3

        else:
            return self.n_bags, self.length, 3

    def __call__(self, acceleration, is_train=True, position=None, timeInfo=False):

        if self.processing:
            signals = {}

        else:
            if self.transfer and not self.MIL:
                acceleration = np.array([acceleration[0][(self.n_bags - 1) * self.stride:]])
                position = [position[-1]]

            else:
                acceleration = np.array(
                    [acceleration[0][
                     (self.n_bags - i - 1) * self.stride: (self.n_bags - i - 1) * self.stride + self.length]
                     for i in range(self.n_bags)]
                )

        acc_xyz = np.array([acceleration[i, :, 3 * pos_i: 3 * pos_i + 3] for i, pos_i in enumerate(position)])

        if timeInfo:
            time = acceleration[:, :, -3:]

        if is_train and self.acc_xyz_aug_list:

            for acc_aug, param in zip(self.acc_xyz_aug_list, self.acc_xyz_aug_params):

                if acc_aug == 'Jittering':

                    noise = np.random.normal(0., param, size=acc_xyz.shape[1:])
                    acc_xyz = np.array([acc + noise for acc in acc_xyz])

                elif acc_aug == 'TimeWarp':

                    tt_new, x_range = DA_TimeWarp(self.length, param)
                    acc_xyz = np.array([np.array(
                        [np.interp(x_range, tt_new, acc[:, orientation]) for orientation in
                         range(3)]).transpose() for acc in acc_xyz])

                elif acc_aug == 'Permutation':

                    segs = DA_Permutation(self.length, nPerm=param, minSegLength=200)
                    idx = np.random.permutation(param)
                    acc_xyz = np.array(
                        [np.array([permutate(acc[:, orientation], self.length, segs, idx, param) for orientation in
                                   range(3)]).transpose()
                         for acc in acc_xyz]
                    )

                elif acc_aug == 'Rotation':

                    acc_xyz = np.array([
                        DA_Rotation(acc) for acc in acc_xyz])

        for signal_index, signal_name in enumerate(self.snl):

            if signal_name in self.base_acc_signals:
                acc_i = self.base_acc_signals[signal_name]
                acc_signal = acc_xyz[:, :, acc_i]

            else:

                if signal_name == 'Acc_norm':

                    acc_signal = np.sqrt(np.sum(acc_xyz ** 2, axis=2))

                elif signal_name == 'Jerk':

                    J = np.array(
                        [np.array(
                            [(acc[1:, orientation] - acc[:-1, orientation]) * self.freq for orientation in
                             range(3)]).transpose()
                         for acc in acc_xyz]
                    )

                    acc_signal = np.sqrt(np.sum(J ** 2, axis=2))

                    if not self.processing:

                        acc_signal = np.concatenate((acc_signal, np.zeros((acc_signal.shape[0], 1))), axis=1)


            if self.processing:
                signals[signal_name] = acc_signal

                del acc_signal

            else:
                if signal_index == 0:
                    outputs = acc_signal[:, :, np.newaxis]

                else:
                    outputs = np.concatenate(
                        (outputs, acc_signal[:, :, np.newaxis]),
                        axis=2
                    )

        if self.processing:
            return signals

        else:
            if self.n_bags:
                n_null = self.n_bags - outputs.shape[0]

                if n_null > 0:
                    extra_nulls = np.zeros((n_null, self.length, self.channels))
                    outputs = np.concatenate((outputs, extra_nulls),
                                             axis=0)




                if self.transfer and not self.MIL:
                    outputs = outputs[0]

                    if timeInfo:
                        time = time[0]

            if timeInfo:
                return outputs, time

            else:
                return outputs

class SpectogramAccTransform():

    def __init__(self,
                 shl_args,
                 out_size=(48, 48),
                 accTransfer=False,
                 accMIL=False):

        self.shl_args = shl_args

        self.freq = int(1. / self.shl_args.data_args['accSamplingRate'])
        self.duration_window = int(self.shl_args.train_args['specto_window'] * self.freq)
        self.duration_overlap = int(self.shl_args.train_args['specto_overlap'] * self.freq)
        self.complete = (self.shl_args.data_args['dataset'] == 'CompleteUser1')

        self.signal_name_list = self.shl_args.train_args['acc_signals']
        self.out_size = out_size
        self.concat = self.shl_args.train_args['acc_concat']
        self.shape = self.shl_args.train_args['acc_shape']
        self.channel = self.shl_args.train_args['acc_channel']
        self.n_bags = self.shl_args.train_args['accBagSize']
        self.transfer = accTransfer
        self.MIL = accMIL
        self.aug_list = self.shl_args.train_args['specto_augment']
        self.aug_list = [] if self.aug_list is None else self.aug_list

        self.temp_tfrm = TemporalAccTransform(shl_args=shl_args,
                                              processing = True)

        self.length = self.shl_args.train_args['accDuration']
        self.stride = self.shl_args.train_args['accBagStride']

        self.channels = len(self.shl_args.train_args['acc_signals'])
        self.height, self.width = self.out_size

        self.frequency_masking_param = self.shl_args.train_args['frequency_masking_param']
        self.frequency_mask_num = self.shl_args.train_args['frequency_mask_num']
        self.time_masking_param = self.shl_args.train_args['time_masking_param']
        self.time_mask_num = self.shl_args.train_args['time_mask_num']

    def random_masking_init(self, max_num=800):
        N = self.out_size[1] * self.out_size[0]
        num = np.random.randint(0, max_num)
        arr = np.array([0] * num + [1] * (N - num))

        np.random.shuffle(arr)

        masks = np.reshape(arr, newshape=(self.out_size[0], self.out_size[1]))

        def random_masking(spectrogram):
            masked_spectrogram = spectrogram

            masked_spectrogram = tf.multiply(masked_spectrogram, masks)

            return np.array(masked_spectrogram)

        return random_masking

    def frequency_masking_init(self):

        n, v = self.out_size[1], self.out_size[0]

        for i in range(self.frequency_mask_num):
            f = tf.random.uniform([], 0, self.frequency_masking_param, dtype=tf.int32)
            v = tf.cast(v, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, v - f, dtype=tf.int32)

            mask = tf.concat(
                (
                    tf.ones(shape=(1, 1, n, v - f0 - f)),
                    tf.zeros(shape=(1, 1, n, f)),
                    tf.ones(shape=(1, 1, n, f0))
                ), axis=3
            )

            if i == 0:
                masks = mask

            else:
                masks = tf.concat((masks, mask), axis=0)

        def frequency_masking(spectrogram):

            masked_spectrogram = spectrogram

            for i in range(self.frequency_mask_num):
                masked_spectrogram = tf.multiply(masked_spectrogram, masks[i])

            return np.array(masked_spectrogram)

        return frequency_masking

    def time_masking_init(self):
        n, v = self.out_size[1], self.out_size[0]

        for i in range(self.time_mask_num):
            t = tf.random.uniform([], 0, self.time_masking_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, n - t, dtype=tf.int32)

            mask = tf.concat(
                (
                    tf.ones(shape=(1, 1, n - t0 - t, v)),
                    tf.zeros(shape=(1, 1, t, v)),
                    tf.ones(shape=(1, 1, t0, v))
                ), axis=2
            )

            if i == 0:
                masks = mask

            else:
                masks = tf.concat((masks, mask), axis=0)

        def time_masking(spectrogram):
            masked_spectrogram = spectrogram
            for i in range(self.time_mask_num):
                masked_spectrogram = tf.multiply(masked_spectrogram, masks[i])

            return np.array(masked_spectrogram)

        return time_masking

    def augmentation_init(self):
        aug_functions = []

        for aug in self.aug_list:

            if aug == 'frequencyMask':
                aug_functions.append(self.frequency_masking_init())

            if aug == 'timeMask':
                aug_functions.append(self.time_masking_init())

            if aug == 'randomMask':
                aug_functions.append(self.random_masking_init())

        def augment(spectrogram):
            for aug_function in aug_functions:
                spectrogram = aug_function(spectrogram)

            return spectrogram

        return augment

    def log_inter(self, spectrograms, freq, time):

        samples = spectrograms.shape[0]
        out_f, out_t = self.out_size
        out_spectrograms = np.zeros((samples, out_f, out_t), dtype=np.float64)

        log_f = np.log(freq + freq[1])

        log_f_normalized = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
        f = out_f * log_f_normalized

        t_normalized = (time - time[0]) / (time[-1] - time[0])
        t = out_t * t_normalized

        f_i = np.arange(out_f)
        t_i = np.arange(out_t)

        for i, spectrogram in enumerate(spectrograms):
            spectrogram_fn = interpolate.interp2d(t, f, spectrogram, copy=False)
            out_spectrograms[i, :, :] = spectrogram_fn(f_i, t_i)

        return out_spectrograms

    def get_shape(self):

        if self.shape == '2D':

            if self.concat == 'Depth':

                if self.transfer and not self.MIL:
                    return self.height, self.width, self.channels

                return self.n_bags, self.height, self.width, self.channels

            elif self.concat == 'Frequency':

                self.height *= self.channels

                if self.transfer and not self.MIL:
                    return self.height, self.width, 1

                return self.n_bags, self.height, self.width, 1

        if self.shape == '1D':

            self.channels *= self.width

            if self.transfer and not self.MIL:
                return self.height, self.channels

            return self.n_bags, self.height, self.channels

    def get_time_shape(self):
        if self.transfer and not self.MIL:
            return self.length, 3

        return self.n_bags, self.length, 3

    def __call__(self, acceleration, is_train=True, position=None, timeInfo=False):

        if self.transfer and not self.MIL:
            acceleration = np.array([acceleration[0][(self.n_bags - 1) * self.stride:]])
            position = [position[-1]]

        else:

            acceleration = np.array(
                [acceleration[0][(self.n_bags - i - 1) * self.stride:(self.n_bags - i - 1) * self.stride + self.length]
                 for i in range(self.n_bags)])

        if timeInfo:
            time = acceleration[:, :, -3:]

        signals = self.temp_tfrm(acceleration,
                                 is_train=is_train,
                                 position=position)

        del acceleration

        if is_train:
            augment = self.augmentation_init()

        signal_index = 0

        for signal_name in signals.keys():

            f, t, spectrograms = signal.spectrogram(signals[signal_name],
                                                    fs=self.freq,
                                                    nperseg=self.duration_window,
                                                    noverlap=self.duration_overlap)

            spectrograms = self.log_inter(spectrograms, f, t)

            if is_train:
                spectrograms = augment(spectrograms)

            np.log(spectrograms + 1e-10, dtype=np.float64, out=spectrograms)

            if self.shape == '2D':

                if self.concat == 'Depth':

                    if signal_index == 0:
                        outputs = spectrograms[:, :, :, np.newaxis]

                    else:
                        outputs = np.concatenate((outputs,
                                                  spectrograms[:, :, :, np.newaxis]),
                                                axis=3)

                elif self.concat == 'Frequency':

                    if signal_index == 0:
                        outputs = spectrograms[:, :, :, np.newaxis]

                    else:
                        outputs = np.concatenate((outputs,
                                                  spectrograms[:, :, :, np.newaxis]),
                                                 axis=1)

            if self.shape == '1D':

                if self.channel == 'Frequency':

                    if signal_index == 0:
                        outputs = spectrograms.transpose((0, 2, 1))

                    else:
                        outputs = np.concatenate((outputs,
                                                  spectrograms.transpose((0,2,1))),
                                                 axis=2)

                if self.channel == 'Time':

                    if signal_index == 0:
                        outputs = spectrograms

                    else:
                        outputs = np.concatenate((outputs,
                                                  spectrograms),
                                                 axis=2)

            signal_index += 1

        if self.n_bags:

            n_null = self.n_bags - outputs.shape[0]
            if n_null > 0:
                if self.shape == '2D':
                    if self.concat == 'Depth':
                        extra_nulls = np.zeros((n_null, self.height, self.width, self.channels))

                    elif self.concat == 'Frequency':
                        extra_nulls = np.zeros((n_null, self.height, self.width, 1))

                if self.shape == '1D':
                    extra_nulls = np.zeros((n_null, self.height, self.channels))

                outputs = np.concatenate((outputs, extra_nulls),
                                         axis=0)

            if self.transfer and not self.MIL:
                outputs = outputs[0]
                if timeInfo:
                    time = time[0]

        if timeInfo:
            return outputs, time

        else:
            return outputs


class TemporalLocationTransform:
    def __init__(self, shl_args=None, locTransfer=False):

        self.shl_args = shl_args
        self.earthR = 6372.

        self.location_noise = shl_args.train_args['location_noise']
        self.complete = (shl_args.data_args['dataset'] == 'CompleteUser1')

        self.transfer = locTransfer

        self.loc_signals = {
            'Acc': 0,
            'Lat': 1,
            'Long': 2,
            'Alt': 3,
            'GPS': 4
        }

        self.n_bags = 1
        self.time_features = shl_args.train_args['time_features']
        self.statistical_features = shl_args.train_args['statistical_features']
        self.point_features = shl_args.train_args['point_features']
        self.mask = shl_args.train_args['mask']
        self.mean = True if 'Mean' in self.statistical_features else False
        self.var = True if 'Var' in self.statistical_features else False
        self.length = self.shl_args.data_args['locDuration']
        self.channels = len(self.time_features)
        self.threshold = self.shl_args.train_args['padding_threshold']
        if not self.threshold:
            self.threshold = self.length
        self.syncing = self.shl_args.train_args['sync']


        if self.syncing == 'Past':
            self.pivot = self.length - 1
        elif self.syncing == 'Present':
            self.pivot = self.length // 2
        elif self.syncing == 'Future':
            self.pivot = 0
        else:
            self.pivot = self.length - 1


    def get_shape(self):

        if 'Jerk' in self.time_features:
            self.totalLength = self.length - 3

        elif 'Acceleration' in self.time_features or 'BearingRate' in self.time_features or 'Movability' in self.time_features:
            self.totalLength = self.length - 2

        elif 'Distance' in self.time_features or 'Velocity' in self.time_features or 'Bearing' in self.time_features:
            self.totalLength = self.length - 1

        elif any(signal in self.time_features for signal in self.loc_signals):
            self.totalLength = self.length

        if self.transfer:
            signals_dims = (self.totalLength, self.channels)

        else:
            signals_dims = (self.n_bags, self.totalLength, self.channels)

        self.n_stat_features = 0
        self.n_point_features = 0

        for feature in self.statistical_features:
            if feature in ['TotalDistance', 'TotalVelocity', 'TotalWalk']:
                self.n_stat_features += 1

            elif feature in ['Min', 'Max', 'Mean', 'Var', 'Quantile', 'Skewness', 'Kurtosis']:
                self.n_stat_features += len(self.time_features)

        for feature in self.point_features:
            if feature in ['Accuracy','Velocity','Acceleration','Jerk','BearingRate','Movability']:
                self.n_point_features += 1

        self.features = self.n_stat_features + self.n_point_features
        if self.transfer:
            feature_dims = self.features
        else:
            feature_dims = (self.n_bags, self.features)

        return signals_dims, feature_dims

    def get_time_shape(self):
        if self.transfer:
            return self.length, 3

        else:
            return self.n_bags, self.length, 3

    def add_noise(self, acc, lat, lon, alt, moment):
        noise_radius = np.random.normal(0.,
                                        acc[moment] * self.shl_args.train_args['noise_std_factor'])

        noise_theta = np.random.uniform(0., 1.) * np.pi
        noise_phi = np.random.uniform(0., 2.) * np.pi

        noise_lat = noise_radius * np.cos(noise_phi) * np.sin(noise_theta)
        noise_lon = noise_radius * np.sin(noise_phi) * np.sin(noise_theta)
        noise_alt = noise_radius * np.cos(noise_theta)

        m = 180. / (self.earthR * 1000. * np.pi)
        new_lat = lat[moment] + noise_lat * m
        new_lon = lon[moment] + noise_lon * m / np.cos(lat[moment] * (np.pi / 180.))
        new_alt = alt[moment] + noise_alt
        return np.array([new_lat, new_lon, new_alt])

    def noisy(self, pos_location, duration):

        return np.array(
            [np.array([self.add_noise(acc, x, y, z, moment) for moment in range(duration)]) for acc, x, y, z in zip(
                pos_location[:, :, 0], pos_location[:, :, 1], pos_location[:, :, 2], pos_location[:, :, 3]
            )])

    def get_distance(self, lat, lon, moment):

        point1 = (lat[moment - 1], lon[moment - 1])
        point2 = (lat[moment], lon[moment])

        return great_circle(point1, point2).m

    def get_velocity(self, lat, lon, t, moment, kmh = False):
        f = 3600. if kmh else 1000.
        hvs_dis = self.get_distance(lat, lon, moment)
        return f * hvs_dis / (t[moment] - t[moment - 1])  # m/s

    def get_acceleration(self, lat, lon, t, moment, kmh = False):
        f = 3600. if kmh else 1000.
        v1 = self.get_velocity(lat, lon, t, moment - 1, kmh=False)
        v2 = self.get_velocity(lat, lon, t, moment, kmh=False)
        return f * (v2 - v1) / (t[moment] - t[moment - 1])

    def get_jerk(self, lat, lon, t, moment, kmh = False):
        f = 3600. if kmh else 1000.
        a1 = self.get_acceleration(lat, lon, t, moment)
        a2 = self.get_acceleration(lat, lon, t, moment - 1)
        return f * (a2 - a1) / (t[moment] - t[moment - 1])

    def get_bearing(self, lat, lon, t, moment):

        y = np.sin(lon[moment] - lon[moment - 1]) * np.cos(lat[moment])
        x = np.cos(lat[moment - 1]) * np.sin(lat[moment]) - \
            np.sin(lat[moment - 1]) * np.cos(lon[moment] - lon[moment - 1]) * np.cos(lat[moment])

        angle = (np.degrees(np.arctan2(y, x)) + 360) % 360

        return 1000. * angle / (t[moment] - t[moment - 1])

    def get_bearing_rate(self, lat, lon, t, moment):
        angle1 = self.get_bearing(lat, lon, t, moment - 1)
        angle2 = self.get_bearing(lat, lon, t, moment)
        return 1000. * (angle2 - angle1) / (t[moment] - t[moment - 1])

    def get_movability(self, lat, lon, moment):
        point1 = (lat[moment - 2], lon[moment - 2])
        point2 = (lat[moment - 1], lon[moment - 1])
        point3 = (lat[moment], lon[moment])

        distance =  great_circle(point1, point3).m
        displacement = great_circle(point1, point2).m + great_circle(point2, point3).m

        return distance / (displacement + 1e-10)

    def distance(self, location, duration):
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        dis_signal = np.array([[self.get_distance(x, y, moment) for moment in range(1, duration)]
                                    for x, y in zip(lats,lons)])

        return dis_signal

    def velocity(self, location, samples, duration):
        if samples:
            time = location[:, :, -1]
            lats = location[:, :, 1]
            lons = location[:, :, 2]

            vel_signal = np.array([[self.get_velocity(x, y, t, moment) for moment in range(1, duration)]
                                    for x, y, t in zip(lats,lons,time)])

            return vel_signal
        else:
            return np.zeros((0, duration - 1))

    def acceleration(self, location, samples, duration):
        if samples:
            time = location[:, :, -1]
            lats = location[:, :, 1]
            lons = location[:, :, 2]

            acc_signal = np.array([[self.get_acceleration(x, y, t, moment) for moment in range(2, duration)]
                                   for x, y, t in zip(lats,lons,time)])

            return acc_signal
        else:
            return np.zeros((0, duration - 2))

    def jerk(self, location, duration):

        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        jerk_signal = np.array([[self.get_jerk(x, y, t, moment) for moment in range(3, duration)]
                                   for x, y, t in zip(lats,lons,time)])

        return jerk_signal

    def bearing(self, pos_location, duration):

        time = pos_location[:, :, -1]
        lats = pos_location[:, :, 1]
        lons = pos_location[:, :, 2]

        bear_signal = np.array([[self.get_bearing(x, y, t, moment) for moment in range(1, duration)]
                                for x,y,t in zip(lats,lons,time)])

        return bear_signal

    def bearing_rate(self, pos_location, duration):

        time = pos_location[:, :, -1]
        lats = pos_location[:, :, 1]
        lons = pos_location[:, :, 2]

        BR_signal = np.array([[self.get_bearing_rate(x, y, t, moment) for moment in range(2, duration)]
                                 for x,y,t in zip(lats,lons,time)])

        return BR_signal

    def cropping(self, location_bag):
        max_front_pad = 0
        max_end_pad = 0
        after = 0
        before = 0

        for location_window in location_bag:

            for after in range(1, self.length - self.pivot + 1):
                if after == self.length - self.pivot:
                    break
                if location_window[self.pivot + after][0] == -1:
                    break

            end_padding = self.length - self.pivot - after

            for before in range(1, self.pivot + 2):
                if before == self.pivot + 1:
                    break
                if location_window[self.pivot - before][0] == -1:
                    break

            front_padding = self.pivot - before + 1

            max_front_pad = max(front_padding, max_front_pad)
            max_end_pad = max(end_padding, max_end_pad)

        cropped_length = self.length - max_front_pad - max_end_pad

        if cropped_length < self.threshold:
            return np.zeros((0, *location_bag.shape[1:])), None, None, self.length

        return np.delete(location_bag, [*[i for i in range(max_front_pad)], *[-i - 1 for i in range(max_end_pad)]], 1), \
               max_front_pad, \
               max_end_pad, \
               cropped_length

    def __call__(self, location, is_train=True, timeInfo=False):

        location, front_pad, end_pad, length = self.cropping(location)
        samples = location.shape[0]

        if self.location_noise and is_train and np.size(location):
            location[:, :, 1:4] = self.noisy(location, length)

        statistical_features = None
        aggregated_features = None
        time_series = None
        signals = None

        if timeInfo:
            time = location[:, :, -3:]

        for index, time_feature in enumerate(self.time_features):
            if time_feature in self.loc_signals:
                time_series = location[:, :, self.loc_signals[time_feature]]
            else:
                if time_feature == 'Distance':

                    time_series = self.distance(location, length)

                elif time_feature == 'Velocity':

                    time_series = self.velocity(location, samples, length)

                elif time_feature == 'Acceleration':

                    time_series = self.acceleration(location, samples, length)

                elif time_feature == 'Jerk':

                    time_series = self.jerk(location, length)

                elif time_feature == 'Bearing':

                    time_series = self.bearing(location, length)

                elif time_feature == 'BearingRate':

                    time_series = self.bearing_rate(location, length)

                if self.mean:
                    statistical_features = np.mean(time_series, axis=1)[:, np.newaxis] if statistical_features is None \
                        else np.concatenate([statistical_features, np.mean(time_series, axis=1)[:, np.newaxis]], axis=1)

                if self.var:
                    statistical_features = np.var(time_series, axis=1)[:, np.newaxis] if statistical_features is None \
                        else np.concatenate([statistical_features, np.var(time_series, axis=1)[:, np.newaxis]], axis=1)


            if np.size(time_series):
                time_series = np.pad(time_series, [(0, 0), (front_pad, end_pad)],
                                     mode='constant', constant_values=self.mask)

            if index == 0:
                signals = time_series[:, :, np.newaxis]

            else:
                signals = np.concatenate(
                    (signals[:, -self.totalLength:, :],
                     time_series[:, -self.totalLength:, np.newaxis]),
                    axis=2
                )

        init = False
        for feature_index, feature_name in enumerate(self.point_features):

            if feature_name == 'Velocity':

                Velocity = np.zeros((samples, 1))
                for i, positions in enumerate(location):
                    Velocity[i, 0] = self.get_velocity(positions[:,1], positions[:,2], positions[:,3], positions[:,-1], -1)

                if not init:
                    init = True
                    aggregated_features = Velocity

                else:
                    aggregated_features = np.concatenate([aggregated_features, Velocity], axis=1)

            if feature_name == 'Acceleration':

                Acceleration = np.zeros((samples, 1))
                for i, positions in enumerate(location):

                    Acceleration[i, 0] = self.get_acceleration(positions[:,1], positions[:,2], positions[:,3], positions[:,-1], -1)

                if not init:
                    init = True
                    aggregated_features = Acceleration

                else:
                    aggregated_features = np.concatenate([aggregated_features, Acceleration], axis=1)

            if feature_name == 'BearingRate':

                BearingRate = np.zeros((samples, 1))
                for i, positions in enumerate(location):

                    BearingRate[i, 0] = self.get_bearing_rate(positions[:,1], positions[:,2], positions[:,-1], -1)

                if not init:
                    init = True
                    aggregated_features = BearingRate

                else:
                    aggregated_features = np.concatenate([aggregated_features, BearingRate], axis=1)

            if feature_name == 'Movability':

                Movability = np.zeros((samples, 1))
                for i, positions in enumerate(location):

                    Movability[i, 0] = self.get_movability(positions[:,1], positions[:,2], -1)

                if not init:
                    init = True
                    aggregated_features = Movability

                else:
                    aggregated_features = np.concatenate([aggregated_features, Movability], axis=1)

            if feature_name == 'Jerk':

                Jerk = np.zeros((samples, 1))
                for i, positions in enumerate(location):

                    Jerk[i, 0] = self.get_jerk(positions[:,1], positions[:,2], positions[:,3], positions[:,-1], -1)

                if not init:
                    init = True
                    aggregated_features = Jerk

                else:
                    aggregated_features = np.concatenate([aggregated_features, Jerk], axis=1)

        done = False
        for feature_index, feature_name in enumerate(self.statistical_features):
            if feature_name in ['Mean', 'Var']:
                if done:
                    continue

                else:
                    if not init:
                        init = True
                        aggregated_features = statistical_features

                    else:
                        aggregated_features = np.concatenate([aggregated_features, statistical_features], axis=1)

                    done = True

            else:
                if feature_name == 'TotalWalk':
                    dis_signal = self.distance(location, length)

                    TotalWalk = np.zeros((samples, 1))
                    for i, (positions, distances) in enumerate(zip(location, dis_signal)):
                        displacement = np.sum(distances)

                        point1 = (positions[0, 1], positions[0, 2])
                        point2 = (positions[-1, 1], positions[-1, 2])
                        total_distance = great_circle(point1, point2).m

                        TotalWalk[i, 0] = total_distance / (displacement + 1e-10)

                    if not init:
                        init = True
                        aggregated_features = TotalWalk

                    else:
                        aggregated_features = np.concatenate([aggregated_features, TotalWalk], axis=1)

        if self.n_bags:
            n_null = self.n_bags - samples

            if n_null > 0:
                extra_nulls = np.zeros((n_null, self.totalLength, self.channels)) + self.mask
                signals = np.concatenate((signals, extra_nulls),
                                         axis=0)

                extra_nulls = np.zeros((n_null, self.features)) + self.mask
                aggregated_features = np.concatenate((aggregated_features, extra_nulls),
                                          axis=0)

                if timeInfo:
                    extra_nulls = np.zeros((n_null, self.length, 3)) + self.mask
                    time = np.concatenate((time, extra_nulls),
                                          axis=0)

        if self.transfer:
            signals = signals[0]
            aggregated_features = aggregated_features[0]
            if timeInfo:
                time = time[0]

        if timeInfo:
            return signals, aggregated_features, time

        else:
            return signals, aggregated_features

