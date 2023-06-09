import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.interpolate import interp2d
from augment import *
from gpsProcessing import *
from mySpectrogram import LogBands, my_tvs, my_tvs2
from math import floor, ceil


class CategoricalTransformer:
    def __init__(self, motorized=False):
        self.motorized = motorized
        self.n_classes = 5 if motorized else 8
        self.encoding = np.zeros((self.n_classes, self.n_classes), dtype=np.float32)
        for i in range(self.n_classes):
            self.encoding[i, i] = 1.

    def __call__(self, label, timeInfo=False):
        if timeInfo:
            return self.encoding[min(label[0] - 1, self.n_classes - 1)], label[-3:]
        else:
            return self.encoding[min(label[0] - 1, self.n_classes - 1)], None


class temporalTransformer:
    def __init__(self,
                 shl_args=None,
                 accTransfer=False,
                 accMIL=False,
                 preprocessing=False):

        self.transfer = accTransfer
        self.MIL = accMIL
        self.preprocessing = preprocessing
        self.snl = shl_args.train_args['acc_signals']
        self.augmentations = shl_args.train_args['acc_augmentation']
        self.freq = int(1. / shl_args.data_args['accSamplingRate'])
        self.length = shl_args.train_args['accDuration']
        self.stride = shl_args.train_args['accBagStride']
        self.channels = len(shl_args.train_args['acc_signals'])
        self.bagSize = shl_args.train_args['accBagSize']
        self.syncing = shl_args.train_args['sync']
        self.bagPositions = shl_args.train_args['train_bag_positions']
        self.posPerInstance = 4 if self.bagPositions == 'all' else 1
        self.pyramid = shl_args.train_args['pyramid']

        self.rawSignals = {
            'Acc_x': 0,
            'Acc_y': 1,
            'Acc_z': 2
        }

        self.virtualSignals = [
            'Acc_norm',
            'Jerk'
        ]

        if self.syncing == 'Past':
            self.pivot = self.bagSize - 1
        elif self.syncing == 'Present':
            self.pivot = self.bagSize // 2
        elif self.syncing == 'Future':
            self.pivot = 0

    def get_shape(self):
        if self.transfer and not self.MIL:
            return self.length, self.channels

        else:
            return self.bagSize * self.posPerInstance, self.length, self.channels

    def get_time_shape(self):
        if self.transfer and not self.MIL:
            return self.length, 3

        else:
            return self.bagSize * self.posPerInstance, self.length, 3

    def __call__(self, acceleration, is_train=True, position=None, timeInfo=False):

        signals = None
        outputs = None
        time = None

        if self.preprocessing:
            if self.pyramid:
                signals = [{} for _ in range(self.bagSize)]
            else:
                signals = {}

        else:
            if self.transfer and not self.MIL:
                acceleration = np.array(
                    [acceleration[0][self.pivot * self.stride: self.pivot * self.stride + self.length]])
                position = [position[0]]

            else:
                acceleration = np.array(
                    [acceleration[0][i * self.stride: i * self.stride + self.length] for i in range(self.bagSize)]
                )

        if type(acceleration) == list:
            for l, layer in enumerate(acceleration):
                layerPos = position[:self.bagSize - l]
                accXYZ = np.array([layer[i, :, 3 * pos: 3 * pos + 3] for i, pos in layerPos])

                if timeInfo:
                    time = acceleration[:, :, -3:]

                signals[l] = self.generate_signals(accXYZ, is_train)

        else:
            accXYZ = np.array([acceleration[i, :, 3 * pos: 3 * pos + 3] for i, pos in position])

            if timeInfo:
                time = acceleration[:, :, -3:]

            signals = self.generate_signals(accXYZ, is_train)

        if self.preprocessing:
            return signals

        else:
            if self.bagSize:
                n_null = self.bagSize - outputs.shape[0]

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
                return outputs, None

    def generate_signals(self, accXYZ, is_train=True):

        if self.preprocessing:
            outputs = {}

        if is_train and self.augmentations:

            for augmentation in zip(self.augmentations):

                if augmentation == 'Jittering':
                    noise = np.random.normal(0., 1., size=accXYZ.shape[1:])
                    accXYZ = np.array([acc + noise for acc in accXYZ])

                elif augmentation == 'TimeWarp':
                    tt_new, x_range = DA_TimeWarp(self.length, 1.)
                    accXYZ = np.array([np.array(
                        [np.interp(x_range, tt_new, acc[:, orientation]) for orientation in
                         range(3)]).transpose() for acc in accXYZ])

                elif augmentation == 'Permutation':
                    nPerm = 4
                    segs = DA_Permutation(self.length, nPerm=nPerm, minSegLength=200)
                    idx = np.random.permutation(nPerm)
                    accXYZ = np.array(
                        [np.array([permutate(acc[:, orientation], self.length, segs, idx, nPerm) for orientation in
                                   range(3)]).transpose()
                         for acc in accXYZ]
                    )

                elif augmentation == 'Rotation':
                    accXYZ = np.array([
                        DA_Rotation(acc) for acc in accXYZ])

        for thisSignal in self.snl:

            if thisSignal in self.rawSignals:
                s_i = self.rawSignals[thisSignal]
                signal = accXYZ[:, :, s_i]

            else:

                if thisSignal == 'Acc_norm':

                    signal = np.sqrt(np.sum(accXYZ ** 2, axis=2))

                elif thisSignal == 'Jerk':

                    J = np.array(
                        [np.array(
                            [(acc[1:, orientation] - acc[:-1, orientation]) * self.freq for orientation in
                             range(3)]).transpose()
                         for acc in accXYZ]
                    )

                    signal = np.sqrt(np.sum(J ** 2, axis=2))

                    if not self.preprocessing:
                        signal = np.concatenate((signal, np.zeros((signal.shape[0], 1))), axis=1)

            if self.preprocessing:
                outputs[thisSignal] = signal
                del signal

            else:
                if outputs is None:
                    outputs = signal[:, :, np.newaxis]

                else:
                    outputs = np.concatenate(
                        (outputs, signal[:, :, np.newaxis]),
                        axis=2
                    )

        return outputs


class spectrogramTransformer:

    def __init__(self,
                 shl_args,
                 out_size=(48, 48),
                 accTransfer=False,
                 accMIL=False):

        self.freq = int(1. / shl_args.data_args['accSamplingRate'])
        self.nperseg = int(shl_args.train_args['specto_window'] * self.freq)
        self.noverlap = int(shl_args.train_args['specto_overlap'] * self.freq)
        self.complete = shl_args.data_args['dataset'] == 'CompleteUser1'
        self.snl = shl_args.train_args['acc_signals']
        self.out_size = out_size
        self.concat2D = shl_args.train_args['acc_concat']
        self.dimension = shl_args.train_args['acc_shape']
        self.concat1D = shl_args.train_args['acc_channel']
        self.transfer = accTransfer
        self.MIL = accMIL
        self.augmentations = shl_args.train_args['specto_augment']
        self.length = shl_args.train_args['accDuration']
        self.stride = shl_args.train_args['accBagStride']
        self.channels = len(shl_args.train_args['acc_signals'])
        self.height, self.width = self.out_size
        self.syncing = shl_args.train_args['sync']
        self.log = shl_args.train_args['freq_interpolation'] == 'log'
        self.mySpectro = False
        self.bagPositions = shl_args.train_args['train_bag_positions']
        self.bagSize = shl_args.train_args['accBagSize']
        self.posPerInstance = 4 if self.bagPositions == 'all' else 1
        self.pyramid = shl_args.train_args['pyramid']
        if self.pyramid:
            self.totalBagSize = int(self.bagSize * (self.bagSize + 1) / 2)

        else:
            self.totalBagSize = self.bagSize

        self.temp_tfrm = temporalTransformer(shl_args=shl_args,
                                             preprocessing=True)

        if self.syncing == 'Past':
            self.pivot = self.bagSize - 1
        elif self.syncing == 'Present':
            self.pivot = self.bagSize // 2
        elif self.syncing == 'Future':
            self.pivot = 0

    def log_inter(self, spectrograms, freq, time):

        samples = spectrograms.shape[0]
        out_f, out_t = self.out_size
        out_spectrograms = np.zeros((samples, out_f, out_t), dtype=np.float64)

        if self.log:
            log_f = np.log(freq + freq[1])
            log_f_normalized = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
            f = out_f * log_f_normalized

        else:
            f_normalized = (freq - freq[0]) / (freq[-1] - freq[0])
            f = out_f * f_normalized

        t_normalized = (time - time[0]) / (time[-1] - time[0])
        t = out_t * t_normalized

        f_i = np.arange(out_f)
        t_i = np.arange(out_t)

        for i, spectro in enumerate(spectrograms):
            spectrogram_fn = interp2d(t, f, spectro, copy=False)
            out_spectrograms[i, :, :] = spectrogram_fn(f_i, t_i)

        return out_spectrograms

    def get_shape(self):

        if self.dimension == '2D':

            if self.concat2D == 'Depth':

                if self.transfer and not self.MIL:
                    return self.height, self.width, self.channels

                return self.totalBagSize * self.posPerInstance, self.height, self.width, self.channels

            elif self.concat2D == 'Frequency':

                self.height *= self.channels

                if self.transfer and not self.MIL:
                    return self.height, self.width, 1

                return self.totalBagSize * self.posPerInstance, self.height, self.width, 1

        if self.dimension == '1D':

            self.channels *= self.width

            if self.transfer and not self.MIL:
                return self.height, self.channels

            return self.totalBagSize * self.posPerInstance, self.height, self.channels

    def get_time_shape(self):
        if self.transfer and not self.MIL:
            return self.length, 3

        return self.totalBagSize * self.posPerInstance, self.length, 3

    def __call__(self, acceleration, is_train=True, position=None, timeInfo=False):

        masking = None
        outputs = None
        time = None

        if self.transfer and not self.MIL:
            accBag = np.array([acceleration[0][self.pivot * self.stride: self.pivot * self.stride + self.length]])
            position = [position[0]]

        else:
            if self.pyramid:
                accBag = []

                for j in range(1, self.bagSize + 1):
                    layer = np.array(
                        [acceleration[0][i * self.stride: i * self.stride + j * self.length]
                         for i in range(self.bagSize - j + 1)]
                    )

                    accBag.append(layer)

            else:
                accBag = np.array([acceleration[0][i * self.stride: i * self.stride + self.length]
                                         for i in range(self.bagSize)])

        if timeInfo:
            time = acceleration[:, :, -3:]

        signals = self.temp_tfrm(accBag,
                                 is_train=is_train,
                                 position=position)

        del accBag

        if is_train:
            masking = Masking(self.augmentations, self.out_size)

        if type(signals) == list:
            for signalsLayer in signals:
                spectroLayer = self.generate_spectrograms(signalsLayer, masking, is_train)

                if outputs is None:
                    outputs = spectroLayer
                else:
                    outputs = np.concatenate((outputs, spectroLayer), axis=0)

        else:
            outputs = self.generate_spectrograms(signals, masking, is_train)

        if self.totalBagSize:

            n_null = self.totalBagSize - outputs.shape[0]
            if n_null > 0:
                extra_nulls = None

                if self.dimension == '2D':
                    if self.concat2D == 'Depth':
                        extra_nulls = np.zeros((n_null, self.height, self.width, self.channels))

                    elif self.concat2D == 'Frequency':
                        extra_nulls = np.zeros((n_null, self.height, self.width, 1))

                if self.dimension == '1D':
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
            return outputs, None

    def generate_spectrograms(self, signals, masking, is_train):
        outputs = None

        for thisSignal in signals.keys():

            if self.mySpectro:
                samples = signals[thisSignal].shape[0]
                out_f, out_t = self.out_size
                thisSpectrogram = np.zeros((samples, out_f, out_t), dtype=np.float64)

                nfft = 2 * out_f - 1

                for i, signal in enumerate(signals[thisSignal]):
                    if self.log:
                        nfft = 100 * (2 * out_f - 1)
                        t1, w1, Sxx1 = my_tvs2(signal, wsize=50, num_of_windows=out_t, nfft=nfft)
                        log_bands = LogBands(len(w1))
                        thisSpectrogram[i, :, :] = log_bands.apply(Sxx1)

                    else:
                        _, _, Sxx = my_tvs2(signal, wsize=nfft, num_of_windows=out_t, nfft=nfft)
                        thisSpectrogram[i, :, :] = Sxx

            else:

                f, t, thisSpectrogram = spectrogram(signals[thisSignal],
                                                    fs=self.freq,
                                                    nperseg=self.nperseg,
                                                    noverlap=self.noverlap)

                thisSpectrogram = self.log_inter(thisSpectrogram, f, t)

            if is_train:
                thisSpectrogram = masking(thisSpectrogram)

            np.log(thisSpectrogram + 1e-10, dtype=np.float64, out=thisSpectrogram)

            if self.dimension == '2D':
                if self.concat2D == 'Depth':
                    if outputs is None:
                        outputs = thisSpectrogram[:, :, :, np.newaxis]

                    else:
                        outputs = np.concatenate((outputs,
                                                  thisSpectrogram[:, :, :, np.newaxis]),
                                                 axis=3)

                elif self.concat2D == 'Frequency':
                    if outputs is None:
                        outputs = thisSpectrogram[:, :, :, np.newaxis]
                    else:
                        outputs = np.concatenate((outputs,
                                                  thisSpectrogram[:, :, :, np.newaxis]),
                                                 axis=1)

            if self.dimension == '1D':
                if self.concat1D == 'Frequency':
                    if outputs is None:
                        outputs = thisSpectrogram.transpose((0, 2, 1))

                    else:
                        outputs = np.concatenate((outputs,
                                                  thisSpectrogram.transpose((0, 2, 1))),
                                                 axis=2)

                if self.concat1D == 'Time':
                    if outputs is None:
                        outputs = thisSpectrogram

                    else:
                        outputs = np.concatenate((outputs,
                                                  thisSpectrogram),
                                                 axis=2)

        return outputs


class gpsTransformer:
    def __init__(self, shl_args=None, gpsTransfer=False):

        self.earthR = 6372.
        self.augmentation = shl_args.train_args['gps_augmentation']
        self.complete = (shl_args.data_args['dataset'] == 'CompleteUser1')
        self.transfer = gpsTransfer
        self.bagSize = 1
        self.timeFeatures = shl_args.train_args['time_features']
        self.statFeatures = shl_args.train_args['statistical_features']
        self.pointFeatures = shl_args.train_args['point_features']
        self.maskValue = shl_args.train_args['mask']
        self.mean = True if 'Mean' in self.statFeatures else False
        self.var = True if 'Var' in self.statFeatures else False
        self.length = shl_args.data_args['locDuration']
        self.channels = len(self.timeFeatures)
        self.padLimit = shl_args.train_args['padding_threshold']
        if not self.padLimit:
            self.padLimit = self.length
        self.syncing = shl_args.train_args['sync']
        self.finalLength = None
        self.featureSize = None
        self.symmetric = shl_args.train_args['symmetric']

        self.gpsFeatures = {
            'Acc': 0,
            'Lat': 1,
            'Long': 2,
            'Alt': 3,
            'GPS': 4
        }

        if self.syncing == 'Past':
            self.pivot = self.length - 1
        elif self.syncing == 'Present':
            self.pivot = self.length // 2
        elif self.syncing == 'Future':
            self.pivot = 0

    @property
    def get_shape(self):

        if 'Acceleration' in self.timeFeatures or 'BearingRate' in self.timeFeatures:
            self.finalLength = self.length - 4 if self.symmetric else self.length - 2

        elif 'Distance' in self.timeFeatures or 'Velocity' in self.timeFeatures:
            self.finalLength = self.length - 2 if self.symmetric else self.length - 1

        elif any(gpsFeature in self.timeFeatures for gpsFeature in self.gpsFeatures):
            self.finalLength = self.length

        if self.transfer:
            windowShape = (self.finalLength, self.channels)

        else:
            windowShape = (self.bagSize, self.finalLength, self.channels)

        nStat = 0
        nPoint = 0
        for feature in self.statFeatures:
            if feature in ['TotalDistance', 'TotalVelocity', 'TotalMovability']:
                nStat += 1

            elif feature in ['Min', 'Max', 'Mean', 'Var']:
                nStat += self.channels

        for feature in self.pointFeatures:
            if feature in ['Accuracy', 'Velocity', 'Acceleration', 'BearingRate']:
                nPoint += 1

        self.featureSize = nStat + nPoint
        if self.transfer:
            featureShape = self.featureSize
        else:
            featureShape = (self.bagSize, self.featureSize)

        return windowShape, featureShape

    def get_time_shape(self):
        if self.transfer:
            return self.length, 3

        else:
            return self.bagSize, self.length, 3

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

        if cropped_length < self.padLimit:
            return np.zeros((0, *location_bag.shape[1:])), None, None, self.length

        return np.delete(location_bag, [*[i for i in range(max_front_pad)], *[-i - 1 for i in range(max_end_pad)]], 1), \
            max_front_pad, \
            max_end_pad, \
            cropped_length

    def __call__(self, location, is_train=True, timeInfo=False):

        location, front_pad, end_pad, length = self.cropping(location)
        samples = location.shape[0]

        if self.augmentation and is_train and np.size(location):
            location[:, :, 1:4] = noisy(location, length)

        statisticalFeatures = None
        features = None
        sequence = None
        window = None
        time = None

        if timeInfo:
            time = location[:, :, -3:]

        for timeFeature in self.timeFeatures:

            if timeFeature in self.gpsFeatures:
                sequence = location[:, :, self.gpsFeatures[timeFeature]]

            else:
                if timeFeature == 'Distance':

                    sequence = distance(location, samples, length, self.symmetric)

                elif timeFeature == 'Velocity':

                    sequence = velocity(location, samples, length, self.symmetric)

                elif timeFeature == 'Acceleration':

                    sequence = acceleration(location, samples, length, self.symmetric)

                elif timeFeature == 'BearingRate':

                    sequence = bearing_rate(location, samples, length, self.symmetric)

                if self.mean:
                    statisticalFeatures = np.mean(sequence, axis=1)[:, np.newaxis] if statisticalFeatures is None \
                        else np.concatenate([statisticalFeatures, np.mean(sequence, axis=1)[:, np.newaxis]], axis=1)

                if self.var:
                    statisticalFeatures = np.var(sequence, axis=1)[:, np.newaxis] if statisticalFeatures is None \
                        else np.concatenate([statisticalFeatures, np.var(sequence, axis=1)[:, np.newaxis]], axis=1)

            if np.size(sequence):
                sequence = np.pad(sequence, [(0, 0), (front_pad, end_pad)],
                                  mode='constant',
                                  constant_values=self.maskValue)

            if window is None:
                if not self.symmetric:
                    window = sequence[:, -self.finalLength:, np.newaxis]

                else:
                    len = sequence.shape[1]
                    start = int(len // 2 - floor(self.finalLength / 2.))
                    end = int(len // 2 + ceil(self.finalLength / 2.))
                    window = sequence[:, start:end, np.newaxis]

            else:
                if not self.symmetric:
                    window = np.concatenate((window, sequence[:, -self.finalLength:, np.newaxis]), axis=2)

                else:
                    len = sequence.shape[1]
                    start = int(len // 2 - floor(self.finalLength / 2.))
                    end = int(len // 2 + ceil(self.finalLength / 2.))
                    window = np.concatenate((window, sequence[:, start:end, np.newaxis]), axis=2)

        done = False
        for statFeature in self.statFeatures:
            if statFeature in ['Mean', 'Var']:
                if done:
                    continue

                else:
                    if features is None:
                        features = statisticalFeatures

                    else:
                        features = np.concatenate([features, statisticalFeatures], axis=1)

                    done = True

            else:
                if statFeature == 'TotalMovability':
                    distances = distance(location, samples, length, False)

                    Movability = np.zeros((samples, 1))
                    for i, (positions, eachDistance) in enumerate(zip(location, distances)):
                        totalDisplacement = np.sum(eachDistance)
                        point1 = (positions[0, 1], positions[0, 2])
                        point2 = (positions[-1, 1], positions[-1, 2])
                        totalDistance = great_circle(point1, point2).m

                        Movability[i, 0] = totalDistance / (totalDisplacement + 1e-10)

                    if features is None:
                        features = Movability

                    else:
                        features = np.concatenate([features, Movability], axis=1)

        if self.bagSize:
            n_null = self.bagSize - samples

            if n_null > 0:
                extra_nulls = np.zeros((n_null, self.finalLength, self.channels)) + self.maskValue
                window = np.concatenate((window, extra_nulls), axis=0)

                extra_nulls = np.zeros((n_null, self.featureSize)) + self.maskValue
                features = np.concatenate((features, extra_nulls), axis=0)

                if timeInfo:
                    extra_nulls = np.zeros((n_null, self.length, 3)) + self.maskValue
                    time = np.concatenate((time, extra_nulls), axis=0)

        if self.transfer:
            window = window[0]
            features = features[0]
            if timeInfo:
                time = time[0]

        if timeInfo:
            return window, features, time

        else:
            return window, features, None
