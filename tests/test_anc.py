import time
import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.access import Access


class Anc():
    def __init__(self):
        # LMMSE
        self.H = np.array([])
        self.V = np.array([])

    def channel_simulation_ls(self, mixer_frames, pure_frames):
        # stft and divide
        mf_spec_mag, _ = self.divide_magphase(self.fft(mixer_frames))  # mixer
        pf_spec_mag, _ = self.divide_magphase(self.fft(pure_frames))  # pure
        # LS
        for i in range(mf_spec_mag.shape[0]):
            tmp = mf_spec_mag[i]
            tmp[tmp < 1e-6] = pf_spec_mag[i][tmp <
                                             1e-6]  # magnitude为非负，0值处增益直接为0
            if i == 30:
                self.H = pf_spec_mag[i] / tmp
                break

    def channel_simulation_lmmse(self, mixer_frames, pure_frames, SNR):
        # stft and divide
        mf_spec_mag, _ = self.divide_magphase(self.fft(mixer_frames))  # mixer
        pf_spec_mag, _ = self.divide_magphase(self.fft(pure_frames))  # pure
        # LS
        tmp = mf_spec_mag[0]
        tmp[tmp < 1e-6] == 1  # magnitude为非负，0值处直接等于pure幅值
        Hls = pf_spec_mag[0] / tmp
        # LMMSE
        M = len(mf_spec_mag[0])
        Rhh = self.get_rhh_simplify(5, M)
        self.H = Rhh @ np.linalg.inv(Rhh + 1 / SNR * np.eye(M)) @ Hls

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

    def get_rhh_simplify(self, L, M):
        """
        L: 信道固定时延
        M: 基频数量
        """
        re = np.zeros(shape=(M, M))
        for i in range(M):
            for j in range(M):
                if i == j:
                    re[i][j] = 1
                else:
                    re[i][j] = (1 - np.exp(-2.j * np.pi * L * (i - j) /
                                           M)) * M / 2.j * np.pi * L * (i - j)
        return re

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

    def get_avg_power(self, frames):
        n = len(frames)
        tot = 0
        for i in range(0, n):
            tot += frames[i]**2
        return tot / n


def generate_h(count, u, sigma):
    x = np.arange(-count // 2, count // 2)
    return 0.7 * (1 / np.sqrt(2 * np.pi * sigma * sigma)) * np.exp(
        -np.power(x - u, 2) / (2 * sigma * sigma)) + 0.3 * (1 / np.sqrt(
            2 * np.pi * sigma * sigma)) * np.exp(-np.power(x - u - 80, 2) /
                                                 (2 * sigma * sigma))


def generate_noise(f_lower_bound, f_upper_bound, num_of_base, fs,
                   noise_length):
    noise_frames_count = int(fs * noise_length)
    # random_factor = np.random.rand(num_of_base)
    random_factor = np.ones(num_of_base)  # Test
    random_factor[3:8] = 0
    random_factor[-8:-3] = 0.5
    w_bases = np.linspace(f_lower_bound, f_upper_bound, num_of_base)
    t = np.linspace(0, noise_length, num=noise_frames_count)
    random_noise = np.zeros(noise_frames_count)
    for i in range(num_of_base):
        random_noise += random_factor[i] * np.sin(2 * np.pi * w_bases[i] * t)
    random_noise = random_noise / np.max(np.abs(random_noise))

    return random_noise


def shift_signal(frames, shift_count):
    return np.concatenate((np.zeros(shift_count), frames[shift_count:]))


def run():
    fs = 16e3
    length = 4
    t = np.linspace(0, length, num=int(fs * length))
    s = generate_noise(100, 1000, 30, fs, length)  # signal
    h = generate_h(201, 0, 5)  # channel response
    s_h = np.convolve(shift_signal(s, 1600),
                      h)[len(h) // 2:len(s) + len(h) // 2]  # output signal
    v = np.random.normal(loc=0.0, scale=1.0,
                         size=len(t)) / 100  # ambient noise
    offer = Access.load_wave_with_fs("./waves/raw/offer.wav",
                                     16e3)[17000:len(s_h) +
                                           17000]  # ambient speaker
    offer[:len(offer) // 2] = 0
    x = s_h + v + 3 * offer  # mixer

    plt.figure(1)

    plt.subplot(6, 1, 1)
    plt.plot(t, s)

    plt.subplot(6, 1, 2)
    plt.plot(t, s_h)

    plt.subplot(6, 1, 3)
    plt.plot(h)

    plt.subplot(6, 1, 4)
    plt.plot(t, offer)

    plt.subplot(6, 1, 5)
    plt.plot(t, x)

    anc = Anc()

    # 信道估计
    SNR = anc.get_avg_power(x) / anc.get_avg_power(v)
    tp = 1
    # anc.channel_simulation_ls(x[:int(length * fs // 2)],
    #                           s[:int(length * fs // 2)])
    # anc.channel_simulation_lmmse(x[:int(length * fs // 2)],
    #                              s[:int(length * fs // 2)], SNR)
    anc.channel_simulation_slr(x[:int(length * fs // 2)],
                               s[:int(length * fs // 2)], tp)

    # 噪声去除
    res = anc.eliminate_noise(x[int(length * fs // 2):],
                              s[int(length * fs // 2):], tp)

    plt.figure(1)
    plt.subplot(6, 1, 6)
    plt.plot(t, np.concatenate((np.zeros(int(length * fs // 2)), res)))

    plt.show()

if __name__ == "__main__":
    run()