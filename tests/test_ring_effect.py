from matplotlib.pyplot import imread, jet
import numpy as np
import scipy
from utils.mplot import MPlot
from scipy import signal
from utils.access import Access
from utils.mfft import MFFT
from tqdm import tqdm

# def run():
#     bh = Access.load_wave("./HW_Base5k.wav")[:4800]
#     b = Access.load_wave("./PB_Base5k.wav")[:4800]

#     S_bh, P_bh = MFFT.divide_magphase(np.fft.fft(bh))
#     S_b, P_b = MFFT.divide_magphase(np.fft.fft(b))

#     MPlot.subplot([S_bh, S_b])
#     S_h = np.zeros_like(S_bh)
#     for i in range(len(S_bh)):
#         if S_b[i] > 1e-8:
#             S_h[i] = S_bh[i] / S_b[i]
#         else:
#             S_h[i] = 0
#     # S_h[S_h > 100] = 0
#     MPlot.plot(S_h)

#     # MPlot.plot_together([S_bh, S_b * S_h])
#     MPlot.subplot([S_bh, S_b * S_h])

#     h = np.fft.ifft(MFFT.merge_magphase(S_h, P_bh)).real
#     MPlot.plot(h)

#     MPlot.plot_together([bh, np.convolve(b, h, mode="same")])

# def run():
#     fs = 48000
#     t = np.linspace(0, 1, fs - 1)
#     a = np.cos(2 * np.pi * 1000 * t)
#     b = np.cos(2 * np.pi * 3000 * t)

#     c = np.convolve(a, b, mode="full")
#     a_padding = align_length(a, len(c), tp=1)
#     b_padding = align_length(b, len(c), tp=1)
#     MPlot.subplot([
#         np.abs(np.fft.fft(c)),
#         np.abs(np.fft.fft(a_padding) * np.fft.fft(b_padding))
#     ])

#     MPlot.subplot([
#         np.angle(np.fft.fft(c)),
#         np.angle(np.fft.fft(a_padding) * np.fft.fft(b_padding))
#     ])

#     c_recovered = np.fft.ifft(np.fft.fft(a_padding) *
#                               np.fft.fft(b_padding)).real
#     MPlot.plot_together([c, c_recovered])

# def run():
#     fs = 8000
#     t = np.linspace(0, 1, fs)

#     t1 = (np.linspace(0, 1, 2000))
#     h = np.sin(2 * np.pi * 10 * t1 + 100 * t1**2) / (t1 + 0.1)**2
#     h = h / np.max(np.abs(h))
#     MPlot.plot(h)
#     x = np.cos(2 * np.pi * 200 * t)
#     noise = np.random.normal(0, 0.01, len(x) + len(h) - 1)

#     y = MFFT.convolve(x, h) + noise
#     MPlot.plot(y)

#     x_recoverd1 = MFFT.deconvolve(y, h)
#     x_recoverd2 = MFFT.wiener_deconvolve(y,h,x,noise)
#     MPlot.plot_together([x, x_recoverd2])


def run():
    x_raw = Access.load_wave("./PB_Impulse10_Base5k.wav")
    y_raw = Access.load_wave("./HW_Impulse10_Base5k.wav")
    MPlot.subplot([x_raw, y_raw])

    x = x_raw[48000:48000 + 4800]
    index = 51014
    error_rate = []
    for i in tqdm(range(12000)):
        y = y_raw[index:index + 4801 + i]
        h_recovered = MFFT.deconvolve(y, x)
        error_rate.append(np.var(y - np.convolve(x, h_recovered)))
    MPlot.plot(error_rate)
    print(np.argmin(error_rate), np.min(error_rate))

    x = x_raw[48000:48000 + 4800]
    y = y_raw[index:index + 4801 + np.argmin(error_rate)]
    h_recovered = MFFT.deconvolve(y, x)
    MPlot.plot_together([y, np.convolve(x, h_recovered)])
