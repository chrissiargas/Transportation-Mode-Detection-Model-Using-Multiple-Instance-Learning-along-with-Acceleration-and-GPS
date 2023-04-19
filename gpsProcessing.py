from geopy.distance import great_circle
import numpy as np


def get_distance(lat, lon, moment):
    point1 = (lat[moment - 1], lon[moment - 1])
    point2 = (lat[moment], lon[moment])
    return great_circle(point1, point2).m


def get_velocity(lat, lon, t, moment, kmh=False):
    f = 3600. if kmh else 1000.
    hvs_dis = get_distance(lat, lon, moment)
    return f * hvs_dis / (t[moment] - t[moment - 1])  # m/s


def get_acceleration(lat, lon, t, moment, kmh=False):
    f = 3600. if kmh else 1000.
    v1 = get_velocity(lat, lon, t, moment - 1, kmh=False)
    v2 = get_velocity(lat, lon, t, moment, kmh=False)
    return f * (v2 - v1) / (t[moment] - t[moment - 1])


def get_jerk(lat, lon, t, moment, kmh=False):
    f = 3600. if kmh else 1000.
    a1 = get_acceleration(lat, lon, t, moment)
    a2 = get_acceleration(lat, lon, t, moment - 1)
    return f * (a2 - a1) / (t[moment] - t[moment - 1])


def get_bearing(lat, lon, moment):
    y = np.sin(lon[moment] - lon[moment - 1]) * np.cos(lat[moment])
    x = np.cos(lat[moment - 1]) * np.sin(lat[moment]) - \
        np.sin(lat[moment - 1]) * np.cos(lon[moment] - lon[moment - 1]) * np.cos(lat[moment])

    angle = (np.degrees(np.arctan2(y, x)) + 360) % 360

    return angle


def get_bearing_rate(lat, lon, t, moment):
    angle1 = get_bearing(lat, lon, moment - 1)
    angle2 = get_bearing(lat, lon, moment)
    return 1000. * abs(angle2 - angle1) / (t[moment] - t[moment - 1])


def distance(location, duration):
    lats = location[:, :, 1]
    lons = location[:, :, 2]

    sequence = np.array([[get_distance(x, y, moment) for moment in range(1, duration)]
                         for x, y in zip(lats, lons)])

    return sequence


def velocity(location, samples, duration):
    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_velocity(x, y, t, moment) for moment in range(1, duration)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence
    else:
        return np.zeros((0, duration - 1))


def acceleration(location, samples, duration):
    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_acceleration(x, y, t, moment) for moment in range(2, duration)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence
    else:
        return np.zeros((0, duration - 2))


def jerk(location, samples, duration):
    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_jerk(x, y, t, moment) for moment in range(3, duration)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence
    else:
        return np.zeros((0, duration - 3))


def bearing(location, samples, duration):
    if samples:
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_bearing(x, y, moment) for moment in range(1, duration)]
                             for x, y in zip(lats, lons)])

        return sequence
    else:
        return np.zeros((0, duration - 1))


def bearing_rate(location, samples, duration):
    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_bearing_rate(x, y, t, moment) for moment in range(2, duration)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence

    else:
        return np.zeros((0, duration - 2))
