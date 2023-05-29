import numpy as np
from geopy.distance import great_circle
from gpsProcessing import get_velocity, get_distance


def acc_features(acceleration, position):

    pos_i = 3 * position

    magnitude = np.sqrt(np.sum(acceleration[:, :, pos_i:pos_i + 3] ** 2, axis=2))[0]

    var = np.var(magnitude)

    freq_acc = np.fft.fft(magnitude)
    freq_magnitude = np.power(np.abs(freq_acc), 2)

    coef1Hz = freq_magnitude[1]
    coef2Hz = freq_magnitude[2]
    coef3Hz = freq_magnitude[3]

    return var, coef1Hz, coef2Hz, coef3Hz


def velocity(pos_location, duration):

    time = pos_location[:, -1]
    lats = pos_location[:, 1]
    lons = pos_location[:, 2]

    sequence = np.zeros((duration - 1))

    for moment in range(1, duration):
        sequence[moment - 1] = get_velocity(lats, lons, time, moment)

    return sequence


def gps_features(location, duration):

    if location.shape[0]:
        location = location[0]

    else:
        return -1

    for timestamp in location:
        if timestamp[0] == -1.:
            return -1

    v = velocity(location, duration)

    return v[0]
