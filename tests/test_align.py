import numpy as np
from utils.mplot import MPlot
from utils.mfft import MFFT
import matplotlib.pyplot as plt
import scipy.signal as signal


def run():
    fs = 96000
    t = np.linspace(0, 0.01, num=fs)
    freq = np.fft.fftfreq(t.shape[-1])
    a = np.sin(2 * np.pi * 100 * t)
    b = np.concatenate((np.zeros(30), a[:-30]))

    MPlot.plot(b - a)

    aZxx = np.fft.fft(a)
    bZxx = np.fft.fft(b)
    MPlot.plot(np.fft.ifft((bZxx.real - aZxx.real) * aZxx.imag).real)

    aZxx = MFFT.divide_magphase(MFFT.stft(a, fs)[-1])
    bZxx = MFFT.divide_magphase(MFFT.stft(b, fs)[-1])
    MPlot.plot(MFFT.istft((bZxx[0] - aZxx[0]) * aZxx[1], fs))