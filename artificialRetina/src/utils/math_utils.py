import numpy as np


def clamp(x, a, b):
    return max(a, min(x, b))


def clampclr(clr: np.ndarray):
    return np.array([min(x, 255) for x in clr])


def lerp(a, b, t):
    return (1-t)*a+t*b


def inverse_lerp(a, b, x):
    return (x-a)/(b-a)


def remap(x, a, b, c, d):
    return (b-x)/(b-a)*c+(x-a)/(b-a)*d
