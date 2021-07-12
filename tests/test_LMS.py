from main import align_length
from scipy.signal.signaltools import residue
from utils.iters import SlideWindow
import numpy as np
import scipy.signal as signal
from utils.mplot import MPlot
from utils.access import Access
from utils.iters import SlideWindow
import matplotlib.pyplot as plt
from tqdm import tqdm


def run():
    fs = 48000
    length = 10
    t = np.linspace(0, length, int(fs * length))

    orign = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t) - np.sin(
        2 * np.pi * 400 * t)
    # orign = orign/np.max(np.abs(orign))
    noise = 0.8 * np.random.normal(0, 1, int(fs * length))
    lowpass_filter = signal.butter(8, 0.3, 'lowpass')  # 7.2k/24k=0.3 5/24k=0.2
    noise = signal.filtfilt(lowpass_filter[0], lowpass_filter[1], noise)

    # noise = generate_noise_bases(length,fs)

    mixer = orign + noise

    e = lms(noise,mixer)
    MPlot.plot(e)

    # e = rls(noise,mixer)
    # MPlot.plot(e)
    

    


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

def lms(x, d, N = 4, mu = 2e-4):  # N=4,256
    L = min(len(x),len(d))
    h = np.zeros(N)
    e = np.zeros(L-N)
    for n in tqdm(range(L-N)):
        x_n = x[n:n+N][::-1]
        d_n = d[n] 
        y_n = np.dot(h, x_n.T)
        e_n = d_n - y_n
        h = h + mu * e_n * x_n
        e[n] = e_n
    return e

def rls(x, d, N = 4, lmbd = 0.999, delta = 2e-4):
    L = min(len(x),len(d))
    lmbd_inv = 1/lmbd
    h = np.zeros((N, 1))
    P = np.eye(N)/delta
    e = np.zeros(L-N)
    for n in tqdm(range(L-N)):
        x_n = np.array(x[n:n+N][::-1]).reshape(N, 1)
        d_n = d[n] 
        y_n = np.dot(x_n.T, h)
        e_n = d_n - y_n
        g = np.dot(P, x_n)
        g = g / (lmbd + np.dot(x_n.T, g))
        h = h + e_n * g
        P = lmbd_inv*(P - np.dot(g, np.dot(x_n.T, P)))
        e[n] = e_n
    return e