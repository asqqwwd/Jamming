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
    w = 100
    base_length = 0.01
    base_num = int(base_length * fs)

    filename_list = [
        "HW_Sin1k+Speaker.npy", "Mi_Sin1k+Speaker.npy",
        "HW_Base100+Speaker.npy", "Mi_Base100+Speaker.npy"
    ]
    # filename_list = [
    #     "Mi_Test.npy"
    # ]
    for filename in filename_list:
        m = Access.load_data(filename)
        # MPlot.plot_specgram(m,fs)
        # MPlot.plot(m)
        # raise ValueError("**")

        # # 非理想滤波滤波
        # lowpass_filter = signal.butter(8, 0.3, 'lowpass')  # 2*5k/48k=0.2
        # m = signal.filtfilt(lowpass_filter[0], lowpass_filter[1], m)
        # 理想滤波
        m_sp = np.fft.fft(m)
        S, P = np.abs(m_sp), np.exp(1.j * np.angle(m_sp))
        cut_off_freq = 5e3
        S[int(cut_off_freq / fs * len(m)):int((fs - cut_off_freq) / fs * len(m))] = 0
        m = np.fft.ifft(S * P).real

        # MPlot.subplot([m, m_filtered])
        # raise ValueError("**")

        # 信道估计
        index = 30377
        base = m[index:index + base_num]
        # MPlot.plot(base)
        # raise ValueError("**")

        # 谱减法
        m = m[index:]
        sw = SlideWindow(m, base_num, base_num)
        b_sp = 2 * np.fft.fft(base) / len(
            base)  # 对幅度谱做归一化，归一化后的画出来模长才能=对应频率系数，也可不做归一化
        b_s, b_p = np.abs(b_sp), np.exp(1.j * np.angle(b_sp))
        results = np.array([])
        for a in tqdm(sw):
            a_sp = 2 * np.fft.fft(a) / len(a)
            a_s, a_p = np.abs(a_sp), np.exp(1.j * np.angle(a_sp))
            # c_s = a_s**2 - b_s**2  # 最好不要用开方后用功率减，因为效果很差
            c_s = a_s - b_s
            c_s[c_s < 0] = 0  # 谱减法最简单的方式
            c_p = a_p
            # c_sp = c_s**.5 * c_p* len(a) / 2  # 如果用了开放功率谱减法，记得要恢复到幅度
            c_sp = c_s * c_p * len(a) / 2  # 如果做了归一化，记得ifft前要去归一化
            c = np.fft.ifft(
                c_sp)  # ifft后仍然为复数，虚部没有太大实际意义，其值反应了fft和ifft中间的计算误差，越小越好
            results = np.concatenate((results, c.real))
        # MPlot.plot(results)
        MPlot.plot_together([m, results])
        rvv = np.var(m) / np.var(results)
        print("{}: {}".format(filename, rvv))
        # Access.save_wave(results, "./HW_AncSin1k.wav", 1, 2, fs)


# 生成噪声基底
def generate_noise_bases(length, fs, w):
    t = np.linspace(0, length, int(fs * length))

    # Freq and mag
    rf = [0.61, 0.36, 0.99, 0.35, 0.21]
    re = rf[0] * np.sin(2 * np.pi * w * t) -\
            rf[1] * np.sin(2 * np.pi * 2 * w * t) +\
            rf[2] * np.sin(2 * np.pi * 4 * w * t) +\
            rf[3] * np.sin(2 * np.pi * 8 * w * t) -\
            rf[4] * np.sin(2 * np.pi * 16 * w * t)
    re = re / np.max(np.abs(re))

    return re
