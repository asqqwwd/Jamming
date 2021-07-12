import numpy as np
from utils.mplot import MPlot
from scipy import signal


def run():
    # h = [0] * 9 + [0.1] + [0.01] * 9
    h = [0] * 4000 + [1] + [0.8] * 100 + [0.5] * 100 + [0.3] * 100 + [0.1] * 100
    # t = np.arange(0, 200)
    # a = np.zeros(400)
    # a[150:350] = 1
    # b = np.convolve(a, h)

    # h_1 = [0] * 20 + [1] + list(map(lambda x: -x, h[-20:]))
    # c = np.convolve(a, h_1)
    # d = np.convolve(c[20:-20], h)

    # MPlot.subplot([a, b[20:-20], d])

    fs = 16000
    t = np.linspace(0, 1, fs)
    a = np.sin(2 * np.pi * 100 * t)
    b = np.sin(2 * np.pi * 400 * t)
    c = np.concatenate((a, b))

    # MPlot.plot(np.convolve(a, h))
    MPlot.subplot([c, np.convolve(c, h)])
