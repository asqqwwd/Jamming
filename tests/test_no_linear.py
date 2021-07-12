import numpy as np
from utils.mplot import MPlot
import scipy.signal as signal


def fun(x):
    return x


def run():
    fs = 48000
    length = 1
    t = np.linspace(0, length, num=int(fs * length))

    a =  np.sin(2 * np.pi * 10e3 * t+np.sin(2 * np.pi * 500 * t))
    # b = np.sin(2 * np.pi * 40e3 * t)
    # h = [0] + [1] + [0.5]
    h = [0] * 400 + [1] + [0.8] * 100 + [-0.5] * 100 + [0.3] * 100 + [0.1] * 100
    b = np.convolve(a, h, mode="same")
    # MPlot.subplot_specgram([np.power(a, 2),np.power(b, 2)], fs)
    MPlot.subplot_specgram([a,b], fs)
    # MPlot.plot_specgram(2 * np.pi * 20e3 * t + a, fs)
    # MPlot.subplot_specgram([b, np.power(b, 3)], fs, tps=["db", "db"])

    # MPlot.plot(b-a,fs)
    # MPlot.subplot_specgram([a,b,b-a],fs,tps=["mag","mag","mag"])
    # MPlot.subplot_specgram_plus([a,b,b-a],fs)

    # c = np.sin(2 * np.pi * 40 * t)
    # d = np.sin(2 * np.pi * 60 * t)
    # e = np.sin(2 * np.pi * 100 * t)

    # w_bases = np.linspace(1, 100, 30)
    # random_noise = np.zeros(fs)
    # random_factors = np.random.randint(2, size=30) * 2 - 1
    # for i in range(30):
    #     random_noise += random_factors[i] * np.sin(2 * np.pi * w_bases[i] * t)
    # random_noise = random_noise / np.max(np.abs(random_noise))
    # MPlot.plot(random_noise, fs)
