from geopy.distance import great_circle
import numpy as np


def get_distance(lat, lon, moment, symmetric=False):
    if not symmetric:
        point1 = (lat[moment - 1], lon[moment - 1])
        point2 = (lat[moment], lon[moment])

    else:
        point1 = (lat[moment - 1], lon[moment - 1])
        point2 = (lat[moment + 1], lon[moment + 1])

    return great_circle(point1, point2).m


def get_velocity(lat, lon, t, moment, symmetric=False):
    f = 1000.

    dis = get_distance(lat, lon, moment, symmetric)
    if not symmetric:
        return f * dis / (t[moment] - t[moment - 1])  # m/s

    else:
        return f * dis / (t[moment + 1] - t[moment - 1])


def get_acceleration(lat, lon, t, moment, symmetric=False):
    f = 1000.

    if not symmetric:
        v1 = get_velocity(lat, lon, t, moment - 1, symmetric)
        v2 = get_velocity(lat, lon, t, moment, symmetric)
        return f * (v2 - v1) / (t[moment] - t[moment - 1])

    else:
        v1 = get_velocity(lat, lon, t, moment - 1, symmetric)
        v2 = get_velocity(lat, lon, t, moment + 1, symmetric)
        return f * (v2 - v1) / (t[moment + 1] - t[moment - 1])


def get_bearing(lat, lon, moment, symmetric=False):

    if not symmetric:
        y = np.sin(lon[moment] - lon[moment - 1]) * np.cos(lat[moment])
        x = np.cos(lat[moment - 1]) * np.sin(lat[moment]) - \
            np.sin(lat[moment - 1]) * np.cos(lon[moment] - lon[moment - 1]) * np.cos(lat[moment])

        angle = (np.degrees(np.arctan2(y, x)) + 360) % 360

        return angle

    else:
        y = np.sin(lon[moment + 1] - lon[moment - 1]) * np.cos(lat[moment + 1])
        x = np.cos(lat[moment - 1]) * np.sin(lat[moment + 1]) - \
            np.sin(lat[moment - 1]) * np.cos(lon[moment + 1] - lon[moment - 1]) * np.cos(lat[moment + 1])

        angle = (np.degrees(np.arctan2(y, x)) + 360) % 360

        return angle


def get_bearing_rate(lat, lon, t, moment, symmetric=False):
    f = 1000.

    if not symmetric:
        angle1 = get_bearing(lat, lon, moment - 1, symmetric)
        angle2 = get_bearing(lat, lon, moment, symmetric)
        return f * abs(angle2 - angle1) / (t[moment] - t[moment - 1])

    else:
        angle1 = get_bearing(lat, lon, moment - 1, symmetric)
        angle2 = get_bearing(lat, lon, moment + 1, symmetric)
        return f * abs(angle2 - angle1) / (t[moment + 1] - t[moment - 1])


def distance(location, samples, duration, symmetric=False):
    end = duration - 1 if symmetric else duration

    if samples:
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_distance(x, y, moment, symmetric) for moment in range(1, end)]
                             for x, y in zip(lats, lons)])

        return sequence

    else:
        return np.zeros((0, end - 1))


def velocity(location, samples, duration, symmetric=False):
    end = duration - 1 if symmetric else duration

    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_velocity(x, y, t, moment, symmetric) for moment in range(1, end)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence
    else:
        return np.zeros((0, end - 1))


def acceleration(location, samples, duration, symmetric=False):
    end = duration - 2 if symmetric else duration

    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_acceleration(x, y, t, moment, symmetric) for moment in range(2, end)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence
    else:
        return np.zeros((0, end - 2))


def bearing_rate(location, samples, duration, symmetric=False):
    end = duration - 2 if symmetric else duration

    if samples:
        time = location[:, :, -1]
        lats = location[:, :, 1]
        lons = location[:, :, 2]

        sequence = np.array([[get_bearing_rate(x, y, t, moment, symmetric) for moment in range(2, end)]
                             for x, y, t in zip(lats, lons, time)])

        return sequence

    else:
        return np.zeros((0, end - 2))
