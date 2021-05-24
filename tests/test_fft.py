import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def cal_value(f, spec_n, freq, epsilon=30):
    index_list = []
    for i, ff in enumerate(f):
        if abs(ff - freq) <= epsilon:
            index_list.append(i)
    return np.sum(spec_n[index_list])


def count_freq(f, Zxx, n, epsilon=0.00001):
    index_list = []
    for i, value in enumerate(np.abs(Zxx.T)[n]):
        if value > epsilon:
            index_list.append(i)
    return f[index_list]


def compress(f, spec_n, epsilon=0.00001):
    index_list = []
    for i, value in enumerate(spec_n):
        if value > epsilon:
            index_list.append(i)

    re = []
    for freq in np.arange(1000, 16000 + 1000, 1000):
        re.append(cal_value(f, spec_n, freq))
    re = np.array(re)
    return re


def cal_W(freq):
    W = np.array([])
    fs = 48000
    t = np.linspace(0, 1, num=fs)
    s = np.sin(2 * np.pi * freq * t)
    for i in range(16):
        f, _, Zxx = signal.stft(np.power(s, int(i + 1)),
                                fs=fs,
                                nperseg=1024 * 3)
        if len(W) == 0:
            W = np.expand_dims(compress(f,
                                        np.abs(Zxx.T)[Zxx.shape[1] // 2]), 1)
        else:
            W = np.concatenate(
                (W,
                 np.expand_dims(compress(f,
                                         np.abs(Zxx.T)[Zxx.shape[1] // 2]),
                                1)), 1)
    return W


def run():
    fs = 16000
    t = np.linspace(0, 2, num=fs * 2)
    s = np.sin(2 * np.pi * 1000 * t)
    res = np.zeros(len(s))
    factor = np.random.random(16)
    for i in range(16):
        res += factor[i] * np.power(s, int(i + 1))
    print(factor)

    W = cal_W(16)
    f, t, Zxx = signal.stft(res, fs=fs, nperseg=1024 * 3)  # 窗口越小，频率泄露越严重
    spec_n = np.expand_dims(compress(f, np.abs(Zxx.T)[Zxx.shape[1] // 2]), 1)
    print(np.linalg.inv(W) @ spec_n)

    # print(cal_value(f,Zxx,2000,Zxx.shape[1]//2))
    # print(cal_value(f,Zxx,4000,Zxx.shape[1]//2))
    # print(count_freq(f,Zxx,Zxx.shape[1]//2))

    # plt.figure(1)
    # f, t, Zxx = signal.stft(res, fs=fs, nperseg=1024*3)  # 窗口越小，频率泄露越严重
    # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap="rainbow")
    # plt.colorbar()
    # plt.ylim([f[1], f[-1]])

    plt.figure(2)
    plt.specgram(res, Fs=fs, scale_by_freq=True,
                 sides='default')  # 这里取的窗口应该差不多为256

    plt.show()


def run1():
    t = np.linspace(0, 2, num=96000 * 2)
    beta = 10
    s = np.sin(2 * np.pi * 50e3 * t +
               beta * np.sin(2 * np.pi * 10e3 * t)) + np.sin(2 * np.pi * 40e3)

    res = s
    # for i in range(1):
    #     res += np.power(s, int(i + 1))

    plt.figure(1)
    f, t, Zxx = signal.stft(res, fs=96000, nperseg=512)  # 窗口越小，频率泄露越严重
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap="rainbow")
    plt.colorbar()
    plt.ylim([f[1], f[-1]])

    plt.figure(2)
    plt.specgram(res, Fs=96000, scale_by_freq=True,
                 sides='default')  # 这里取的窗口应该差不多为256

    plt.show()


if __name__ == "__main__":
    run()