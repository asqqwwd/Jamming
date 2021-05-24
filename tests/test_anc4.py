import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.signal as signal
from itertools import groupby

from scipy.signal.filter_design import butter

from utils.access import Access
from utils.pool import *
from utils.resampler import Resampler


class Anc():
    def __init__(self):
        self.freq_list = np.linspace(100, 1000, 19)
        self.W_list = []
        for w in self.freq_list:
            self.W_list.append(self._cal_W(w))
        self.Factor_list = []

    def _channel_simulation_slr(self, mixer_frames):
        """
        """
        # stft and divide
        f, _, Zxx = self._stft(mixer_frames)
        spec_mag, _ = self._divide_magphase(Zxx)

        # 寻找频率中心点
        gap = spec_mag.shape[1] / 19
        index_list = list(
            map(lambda x: int(x + gap / 2),
                np.linspace(0, spec_mag.shape[1], 19)))
        # 计算各项系数
        for i in index_list:
            self.Factor_list.append(
                np.linalg.inv(self.W_list[i]) @ np.expand_dims(
                    self._compress_dim(f, spec_mag.T[i]), 1))

    def _eliminate_noise(self, mixer_frames, noise_factor_list):
        """
        """
        # stft and divide
        f, t, Zxx = self._stft(mixer_frames)
        mf_spec_mag, mf_spec_phase = self._divide_magphase(Zxx)
        # 谱减法
        spec_n = np.zeros(mf_spec_mag.shape[0])
        for W, Factor, noise_factor in zip(self.W_list, self.Factor_list,
                                           noise_factor_list):
            spec_n += noise_factor * W @ Factor
        pf_spec_mag = spec_n.repeat(mf_spec_mag.shape[1], axis=1)

        return self._merge_magphase(np.abs(mf_spec_mag - pf_spec_mag),
                                    mf_spec_phase)

    def _stft(self, frames):
        """
        Return: (n_frame,n_fft)
        """
        # nperseg窗口大小等于nfft，若nperseg>nfft，则会自动填0，但这会导致无法还原
        # 窗口小，提升时间分辨率，降低频率分辨率
        return signal.stft(frames, fs=48000, nperseg=1024 * 6)

    def _istft(self, spectrogram):
        return signal.istft(spectrogram, fs=48000)

    def _divide_magphase(self, D):
        """Separate a complex-valued stft D into its magnitude (S)
        and phase (P) components, so that `D = S * P`."""
        S = np.abs(D)
        P = np.exp(1.j * np.angle(D))
        return S, P

    def _merge_magphase(self, S, P):
        """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
        return S * P

    def _location_by_burr(self, pool, raw_input_frames):
        # 1.找到毛刺起始和终止位置
        burr_index = self._find_burr(raw_input_frames)

        # 2.从last_input池中读取保存的上轮右半数据
        last_input_frames = pool.get_all()

        # 3.将本轮数据的右半保存至last_input池
        pool.put(raw_input_frames[burr_index:])

        # 4.拼接上轮右半数据核本轮左半数据
        joined_input_frames = np.concatenate(
            (last_input_frames, raw_input_frames[:burr_index]))

        return joined_input_frames

    def _find_burr(self, frames):
        # 过滤
        abs_of_frames = np.abs(frames)
        mean_pow = np.mean(abs_of_frames)
        index_list = []
        for i, value in enumerate(abs_of_frames):
            if value > mean_pow * 4:
                index_list.append(i)
        # 分组
        tmp = []
        fun = lambda x: x[1] - x[0]
        for k, g in groupby(enumerate(index_list), fun):
            l1 = [j for i, j in g]  # 连续数字的列表
            if len(l1) >= 2:
                tmp.append((min(l1), max(l1)))
        if len(tmp) == 0:
            logging.info("No found burr in this block!")
            return 0
        re = 0
        for i in range(1, len(tmp)):
            if tmp[i][1] - tmp[i][0] > tmp[re][1] - tmp[re][0]:
                re = i
        return tmp[re][0]  # 这里使用最大值定位效果比min好，但仍然使用最小值

    def _align_length(self, frames, base_frames):
        base_length = len(base_frames)
        aligned_frames = None
        if len(frames) < base_length:
            aligned_frames = np.concatenate(
                (frames, np.zeros(base_length - len(frames))))
        else:
            aligned_frames = frames[:base_length]
        return aligned_frames, base_frames

    def _cal_W(self, freq):
        W = np.array([])
        fs = 48000
        t = np.linspace(0, 1, num=fs)
        s = np.sin(2 * np.pi * freq * t)
        for i in range(16):
            f, _, Zxx = self._stft(s)
            if len(W) == 0:
                W = np.expand_dims(
                    self._compress_dim(f,
                                       np.abs(Zxx.T)[Zxx.shape[1] // 2]), 1)
            else:
                W = np.concatenate(
                    (W,
                     np.expand_dims(
                         self._compress_dim(f,
                                            np.abs(Zxx.T)[Zxx.shape[1] // 2]),
                         1)), 1)
        return W

    def _compress_dim(self, f, spec_n, epsilon=0.00001):
        index_list = []
        for i, value in enumerate(spec_n):
            if value > epsilon:
                index_list.append(i)

        re = []
        for freq in np.arange(1000, 16000 + 1000, 1000):
            re.append(self._sum_factor(f, spec_n, freq))
        re = np.array(re)
        return re

    def _sum_factor(self, f, spec_n, freq, epsilon=30):
        index_list = []
        for i, ff in enumerate(f):
            if abs(ff - freq) <= epsilon:
                index_list.append(i)
        return np.sum(spec_n[index_list])


def generate_noise(f_lower_bound, f_upper_bound, num_of_base, fs,
                   noise_length):
    noise_frames_count = int(fs * noise_length)
    random_factor = np.random.random(num_of_base)
    random_factor2 = np.random.random(num_of_base)
    random_factor = np.ones(num_of_base)  # Test

    # random_factor[-8:-3] = 0.5
    w_bases = np.linspace(f_lower_bound, f_upper_bound, num_of_base)
    t = np.linspace(0, noise_length, num=noise_frames_count)
    random_noise = np.zeros(noise_frames_count)
    for i in range(num_of_base):
        random_noise += random_factor[i] * np.sin(
            2 * np.pi * w_bases[i] * (t - np.pi * random_factor2[i] / 2))
    random_noise = random_noise / np.max(np.abs(random_noise))
    return random_noise


def run():
    anc = Anc()
    t = np.linspace(0, 1, num=48000)
    s = np.sin(2 * np.pi * 1000 * t)

    f, _, Zxx = anc._stft(s)
    plt.figure(1)
    plt.plot(anc._istft(Zxx)[-1])
    plt.show()


if __name__ == "__main__":
    run()