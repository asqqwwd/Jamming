import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from utils.access import Access
from utils.pool import *
from utils.resampler import Resampler


class Anc():
    def __init__(self):
        # LMMSE
        self.H = np.array([])
        self.V = np.array([])

    def channel_simulation_slr(self, mixer_frames, pure_frames, tp=1):
        """
        simple linear regression：认为mixer频率点的y只与pure对应频率x点有关
        tp: 0 => 估计mixer_k，1 => 估计pure_k
        Notes: 估计pure_k效果会好很多
        """
        # stft and divide
        mf_spec_mag, _ = self.divide_magphase(self.fft(mixer_frames))  # mixer
        pf_spec_mag, _ = self.divide_magphase(self.fft(pure_frames))  # pure
        # simple liner regression
        if tp == 0:
            X = mf_spec_mag.T
            Y = pf_spec_mag.T
        else:
            Y = mf_spec_mag.T
            X = pf_spec_mag.T
        beta1 = []
        beta0 = []
        for i in range(Y.shape[0]):  # 对每个频率带都做一元线性回归
            EX = np.mean(X[i])
            EY = np.mean(Y[i])
            DX = np.var(X[i])
            tmp = 0
            for j in range(Y.shape[1]):
                tmp += (X[i][j] - EX) * (Y[i][j] - EY)
            beta1.append(tmp / (DX * (Y.shape[1] - 1)))
            beta0.append(EY - beta1[-1] * EX)
        self.H = np.array(beta1)
        self.V = np.array(beta0)

        # Test
        n = 17
        plt.figure(2)
        plt.plot(X[n])
        plt.plot(Y[n])
        # n = 17
        # tmp1 = X[n]
        # tmp2 = Y[n]
        # plt.figure(2)
        # plt.scatter(tmp1, tmp2)
        # plt.plot(tmp1, beta1[n] * tmp1 + beta0[n])

    def channel_simulation_mlr(self, mixer_frames, pure_frames):
        """
        multiple linear regression：认为mixer频率点的y与pure对应频率x点以及附近频率有关
        """
        pass

    def eliminate_noise(self, mixer_frames, pure_frames, tp=1):
        """
        tp: 1 => k*mixer-pure; 2 => mixer-k*pure
        Notes: 不要在频域直接相减，变换到时域后在相减，在频域相减效果很差
        """
        # stft and divide
        mf_spec_mag, mf_spec_phase = self.divide_magphase(
            self.fft(mixer_frames))  # mixer
        pf_spec_mag, pf_spec_phase = self.divide_magphase(
            self.fft(pure_frames))  # pure
        re_spec_mag = np.zeros(shape=mf_spec_mag.shape)
        # filter
        if tp == 0:
            for i in range(re_spec_mag.shape[0]):
                re_spec_mag[i] = mf_spec_mag[i] * self.H + self.V
            # merge and istft
            return self.ifft(self.merge_magphase(re_spec_mag,
                                                 mf_spec_phase)) - pure_frames
        else:
            for i in range(re_spec_mag.shape[0]):
                re_spec_mag[i] = pf_spec_mag[i] * self.H + self.V
            # merge and istft
            return mixer_frames - self.ifft(
                self.merge_magphase(re_spec_mag, pf_spec_phase))

    def fft(self, frames):
        """
        Return: (n_frame,n_fft)
        """
        return librosa.stft(frames,
                            n_fft=2048,
                            hop_length=160,
                            window="hamming").T

    def ifft(self, frames_spectrum):
        return librosa.istft(frames_spectrum.T,
                             hop_length=160,
                             window="hamming")

    def divide_magphase(self, D, power=1):
        """Separate a complex-valued stft D into its magnitude (S)
        and phase (P) components, so that `D = S * P`."""
        S = np.abs(D)
        S **= power
        P = np.exp(1.j * np.angle(D))
        return S, P

    def merge_magphase(self, S, P):
        """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
        return S * P


def location_by_burr(pool, raw_input_frames):
    # 1.找到毛刺起始和终止位置
    burr_index = find_burr(raw_input_frames)

    # 2.从last_input池中读取保存的上轮右半数据
    last_input_frames = pool.get_all()

    # 3.将本轮数据的右半保存至last_input池
    pool.put(raw_input_frames[burr_index:])

    # 4.拼接上轮右半数据核本轮左半数据
    joined_input_frames = np.concatenate(
        (last_input_frames, raw_input_frames[:burr_index]))

    return joined_input_frames


def find_burr(frames):
    # 过滤
    abs_of_frames = np.abs(frames)
    mean_pow = np.mean(abs_of_frames)
    index_list = []
    for i, value in enumerate(frames):
        if value < mean_pow * 0.2:
            index_list.append(i)
    # 分组
    tmp = []
    fun = lambda x: x[1] - x[0]
    for k, g in groupby(enumerate(index_list), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 50:
            tmp.append((min(l1), max(l1)))
    if len(tmp) == 0:
        logging.info("No found burr in this block!")
        return 0
    re = 0
    for i in range(1, len(tmp)):
        if tmp[i][1] - tmp[i][0] > tmp[re][1] - tmp[re][0]:
            re = i
    return tmp[re][0]  # 这里使用最大值定位效果比min好，但仍然使用最小值


def generate_noise(f_lower_bound, f_upper_bound, num_of_base, fs,
                   noise_length):
    noise_frames_count = int(fs * noise_length)
    # random_factor = np.random.rand(num_of_base)
    random_factor = np.ones(num_of_base)  # Test
    # random_factor[-8:-3] = 0.5
    w_bases = np.linspace(f_lower_bound, f_upper_bound, num_of_base)
    t = np.linspace(0, noise_length, num=noise_frames_count)
    random_noise = np.zeros(noise_frames_count)
    for i in range(num_of_base):
        random_noise += random_factor[i] * np.sin(2 * np.pi * w_bases[i] * t)
    random_noise = random_noise / np.max(np.abs(random_noise))
    return random_noise


def locate_frames(frames, index, cache_pool):
    last_input_frames = cache_pool.get_all()
    cache_pool.put(frames[index:])
    return np.concatenate((last_input_frames, frames[:index]))


def run():
    data = Access.load_data("./input2.npy")
    fs = 16e3
    length = len(data) / fs
    t = np.linspace(0, length, num=len(data))
    cache_pool = PoolNoBlock()
    chirp = np.zeros(1600)
    noise = generate_noise(100, 1000, 30, fs, 1.9)
    chirp_noise = np.concatenate((chirp, noise))
    input_pool = PoolNoBlock()
    input_pool.put(data)

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(chirp_noise)

    flag = False
    anc = Anc()
    res = np.array([])
    last_burr_index = np.inf
    tmp = []
    while not input_pool.is_empty():
        frames = input_pool.get(int(fs * 2))

        # burr_index = find_burr(frames)

        # 过滤
        abs_of_frames = np.abs(frames)
        mean_pow = np.mean(abs_of_frames)
        print(mean_pow)
        for i, value in enumerate(abs_of_frames):
            if value < mean_pow * 0.2:
                tmp.append(value)
            else:
                tmp.append(0)
    plt.subplot(3,1,2)
    plt.plot(t,data)

        # if not flag:
        #     burr_index = find_burr(frames)
        #     if np.abs(last_burr_index - burr_index) < 30:
        #         flag = True
        #         print("***")
        #     last_burr_index = burr_index
        #     located_frames = locate_frames(frames,burr_index,cache_pool)
        # else:
        #     located_frames = locate_frames(frames,burr_index,cache_pool)
        # print(len(located_frames))

        # if not flag:
        #     anc.channel_simulation_slr(located_frames, chirp_noise)
        # else:
        #     r = anc.eliminate_noise(located_frames, chirp_noise)
        #     res = np.concatenate((res, r))

    
    plt.show()


if __name__ == "__main__":
    run()