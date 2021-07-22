from numpy.lib import RankWarning
from scipy.signal.signaltools import residue
from utils.iters import SlideWindow
import numpy as np
import scipy.signal as signal
from utils.mplot import MPlot
from utils.access import Access
from utils.iters import SlideWindow
from utils.resampler import Resampler
from utils.statistics import Statistics
import matplotlib.pyplot as plt
from tqdm import tqdm


def run():
    fs = 48000
    length = 12
    h = [0] * 4 + [0.5] + [0.1] * 3 + [0.2] * 1
    t = np.linspace(0, length, int(fs * length))

    index = 0
    raw = Access.load_wave("./HW_Base5k+Speaker.wav")

    # MPlot.plot_specgram(raw,fs,tp="mag")
    # raise ValueError("**")

    Statistics.eval_relative_DB(raw)
    Statistics.eval_stablity(raw)
    # Statistics.get_max_successive_fragment(raw)
    MPlot.plot(raw)
    base = raw[index:index + 480]
    raw = raw[index:]

    speaker = Access.load_wave_with_fs("./waves/raw/offer.wav", fs)
    speaker = repeat_align_length(speaker, int(fs * length))

    # x = np.random.normal(0, 1, int(fs * length))
    # d = np.convolve(x, h, mode="same") + speaker
    # d = x + speaker
    # MPlot.plot_together([
    #     repeat_align_length(base, int(fs * length)),
    #     repeat_align_length(construct_x(raw), int(fs * length))
    # ])
    x = repeat_align_length(construct_x(raw), int(fs * length))
    d = repeat_align_length(raw, int(fs * length))
    # MPlot.plot_together([x, d])
    # MPlot.subplot([x, d])

    # e = lms(x, d)
    e = rls(x, d)
    MPlot.plot_together([align_length(d, len(e), tp=1), e])

    # *结果保存*
    Access.save_wave(e,"./HW_Base5k+Speaker_Recovered.wav",1,2,fs)


# def run():
#     fs = 48000
#     length = 10
#     t = np.linspace(0, length, int(fs * length))

#     # Test 4
#     # *1.d读取*
#     d = Access.load_data("./Mi_WhiteNoise5k.npy")[90550:]
#     # MPlot.plot(d)

#     # *2.x读取*
#     x = Access.load_data("./PB_WhiteNoise5k.npy")
#     x = align_length(x,len(d))
#     # MPlot.subplot([d, x])
#     # raise ValueError("**")

#     # *3.自适应滤波*
#     e = lms(x, d)
#     MPlot.plot(e)

#     # *4.结果保存*
#     # Access.save_wave(d, "./repeat_noise_d_1_{}.wav".format(noise_rate), 1, 2,
#     #                  fs)
#     # Access.save_wave(e, "./repeat_noise_e_1_{}.wav".format(noise_rate), 1, 2,
#     #                  fs)


# 生成噪声基底
def generate_noise_bases(length, fs):
    t = np.linspace(0, length, int(fs * length) + 1)
    re = np.zeros(len(t))

    w_bases = np.arange(100, 5100, 100)
    rfs = 2 * np.random.randint(0, 2, len(w_bases)) - 1
    for w, rf in zip(w_bases, rfs):
        re += rf * np.sin(2 * np.pi * w * t)
    re = re / np.max(np.abs(re))
    return re[:-1]


def lms(x, d, N=256, mu=2e-2):  # N,mu=(4,2e-4) (256,2e-4)
    L = min(len(x), len(d))
    h = np.zeros(N)
    e = np.zeros(L - N)
    for n in tqdm(range(L - N)):
        x_n = x[n:n + N][::-1]
        d_n = d[n]
        y_n = np.dot(h, x_n.T)
        e_n = d_n - y_n
        h = h + mu * e_n * x_n
        e[n] = e_n
    return e


def rls(x, d, N=16, lmbd=0.999, delta=2e-6):
    L = min(len(x), len(d))
    lmbd_inv = 1 / lmbd
    h = np.zeros((N, 1))
    P = np.eye(N) / delta
    e = np.zeros(L - N)
    for n in tqdm(range(L - N)):
        x_n = np.array(x[n:n + N][::-1]).reshape(N, 1)
        d_n = d[n]
        y_n = np.dot(x_n.T, h)
        e_n = d_n - y_n
        g = np.dot(P, x_n)
        g = g / (lmbd + np.dot(x_n.T, g))
        h = h + e_n * g
        P = lmbd_inv * (P - np.dot(g, np.dot(x_n.T, P)))
        e[n] = e_n
    return e


def align_length(frames, base_length, tp=0):
    '''
    tp=0: left padding zeros
    tp=1: right padding zeros
    '''
    aligned_frames = None
    if len(frames) < base_length and tp == 0:
        aligned_frames = np.concatenate(
            (np.zeros(base_length - len(frames)), frames))
    if len(frames) < base_length and tp == 1:
        aligned_frames = np.concatenate(
            (frames, np.zeros(base_length - len(frames))))
    else:
        aligned_frames = frames[:base_length]
    return aligned_frames


def repeat_align_length(frames, base_length):
    re = np.tile(frames, base_length // len(frames) + 1)
    re = align_length(re, base_length)
    return re


def construct_x(raw):
    re = raw[:480]
    base = raw[:480]
    N = 1
    index_reg = []
    for w in SlideWindow(raw[480:], 480, 480):
        if np.sum(np.power(w - base, 2)) < 0.011:
            base = (N * base + w) / (N + 1)
            N += 1
            index_reg.append(1)
        else:
            index_reg.append(0)
        re = np.concatenate((re, base))
    # MPlot.plot(index_reg)
    return re