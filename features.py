import numpy as np
from geopy.distance import great_circle


def acc_features(acceleration, position, positions):

    pos_i = 3 * positions[position]

    magnitude = np.sqrt(np.sum(acceleration[:, :, pos_i:pos_i + 3] ** 2,
                               axis=2))[0]

    var = np.var(magnitude)

    freq_acc = np.fft.fft(magnitude)
    freq_magnitude = np.power(np.abs(freq_acc), 2)

    coef1Hz = freq_magnitude[1]
    coef2Hz = freq_magnitude[2]
    coef3Hz = freq_magnitude[3]

    acc_features = [var, coef1Hz, coef2Hz, coef3Hz, acceleration[0][0][-3], acceleration[0][0][-2]]

    return acc_features


def calc_haversine_dis(lat, lon, moment):
    point1 = (lat[moment - 1], lon[moment - 1])
    point2 = (lat[moment], lon[moment])
    return great_circle(point1, point2).m


def calc_haversine_vel(lat, lon, t, moment):
    hvs_dis = calc_haversine_dis(lat, lon, moment)
    return 3600. * hvs_dis / (t[moment] - t[moment - 1])


def haversine_velocity(pos_location, duration):
    time_signal = pos_location[:, -1]
    x_signal = pos_location[:, 1]
    y_signal = pos_location[:, 2]

    vel_signal = np.zeros((duration - 1))

    for moment in range(1, duration):
        vel_signal[moment - 1] = calc_haversine_vel(x_signal,
                                                     y_signal,
                                                     time_signal,
                                                     moment)

    return vel_signal


def gps_features(location, positions, whichGPS, duration):

    pos_name = whichGPS

    pos_location = location[positions[pos_name]]

    if np.size(pos_location):
        pos_location = pos_location[0]

    else:
        return -1, -1, -1, -1, -1

    for location_timestamp in pos_location:
        if location_timestamp[0] == -1.:
            return -1, -1, -1, -1, -1

    velocity = haversine_velocity(pos_location, duration)

    return velocity[0], pos_location[-1][-1], pos_location[-1][0], pos_location[-1][1], pos_location[-1][2]