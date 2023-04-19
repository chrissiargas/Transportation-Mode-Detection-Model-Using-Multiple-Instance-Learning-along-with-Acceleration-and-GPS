import numpy as np
from scipy.interpolate import CubicSpline
import tensorflow as tf


def GenerateRandomCurves(N, sigma=0.2, knot=4, xyz=False):
    if not xyz:
        xx = (np.arange(0, N, (N - 1) / (knot + 1))).transpose()
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
    segs = None
    if not xyz:
        bWhile = True
        while bWhile:
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


def random_masking_init(out_size):
    max_num = 800
    N = out_size[1] * out_size[0]
    num = np.random.randint(0, max_num)
    arr = np.array([0] * num + [1] * (N - num))

    np.random.shuffle(arr)

    masks = np.reshape(arr, newshape=(out_size[0], out_size[1]))

    def random_masking(spectrogram):
        masked_spectrogram = spectrogram

        masked_spectrogram = tf.multiply(masked_spectrogram, masks)

        return np.array(masked_spectrogram)

    return random_masking


def frequency_masking_init(out_size):
    frequency_masking_param = 5
    frequency_mask_num = 2
    n, v = out_size[1], out_size[0]
    masks = None

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], 0, frequency_masking_param, dtype=tf.int32)
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

        for i in range(frequency_mask_num):
            masked_spectrogram = tf.multiply(masked_spectrogram, masks[i])

        return np.array(masked_spectrogram)

    return frequency_masking


def time_masking_init(out_size):
    time_masking_param = 5
    time_mask_num = 2
    n, v = out_size[1], out_size[0]
    masks = None

    for i in range(time_mask_num):
        t = tf.random.uniform([], 0, time_masking_param, dtype=tf.int32)
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
        for i in range(time_mask_num):
            masked_spectrogram = tf.multiply(masked_spectrogram, masks[i])

        return np.array(masked_spectrogram)

    return time_masking


def Masking(augmentations, out_size):
    aug_functions = []

    for augmentation in augmentations:

        if augmentation == 'frequencyMask':
            aug_functions.append(frequency_masking_init(out_size))

        if augmentation == 'timeMask':
            aug_functions.append(time_masking_init(out_size))

        if augmentation == 'randomMask':
            aug_functions.append(random_masking_init(out_size))

    def augment(spectrogram):
        for aug_function in aug_functions:
            spectrogram = aug_function(spectrogram)

        return spectrogram

    return augment


def add_noise(acc, lat, lon, alt, moment):
    factor = 1.
    earthR = 6372.

    noise_radius = np.random.normal(0., acc[moment] * factor)

    noise_theta = np.random.uniform(0., 1.) * np.pi
    noise_phi = np.random.uniform(0., 2.) * np.pi

    noise_lat = noise_radius * np.cos(noise_phi) * np.sin(noise_theta)
    noise_lon = noise_radius * np.sin(noise_phi) * np.sin(noise_theta)
    noise_alt = noise_radius * np.cos(noise_theta)

    m = 180. / (earthR * 1000. * np.pi)
    new_lat = lat[moment] + noise_lat * m
    new_lon = lon[moment] + noise_lon * m / np.cos(lat[moment] * (np.pi / 180.))
    new_alt = alt[moment] + noise_alt
    return np.array([new_lat, new_lon, new_alt])


def noisy(pos_location, duration):

    return np.array(
        [np.array([add_noise(acc, x, y, z, moment) for moment in range(duration)]) for acc, x, y, z in zip(
            pos_location[:, :, 0], pos_location[:, :, 1], pos_location[:, :, 2], pos_location[:, :, 3]
        )])



