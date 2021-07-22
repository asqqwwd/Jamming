from operator import length_hint
from matplotlib.pyplot import imread
from scipy.signal import lti, lsim, impulse, deconvolve
import numpy as np
from utils.mplot import MPlot
from utils.mfft import MFFT
from utils.resampler import Resampler
from scipy import signal


def run():
    lti_sys = lti([6, 5, 1], [1, 2, 3, 4])

    fs = 48000
    length = 1
    t = np.linspace(0, length, int(fs * length))
    a = np.sin(2 * np.pi * 10 * t) - np.sin(2 * np.pi * 20 * t) + np.sin(
        2 * np.pi * 30 * t)

    # tout, yout, xout = lsim(lti_sys, U=a, T=t)
    # MPlot.plot_together([a, yout])

    _, h = impulse(lti_sys)
    MPlot.plot(h)

    H_invert1 = 1 / np.abs(np.fft.fft(h))
    # MPlot.plot(H_invert1)

    # S, P = MFFT.divide_magphase(np.fft.fft(a))
    # a_manipulate = np.fft.ifft(
    #     MFFT.merge_magphase(S * signal.resample(H_invert1, len(a)), P)).real

    recovered, remainder = deconvolve(np.convolve(a, H_invert1),
                                      a)
    MPlot.subplot([h,recovered])
    # MPlot.plot_together([a, np.convolve(a, h)])
