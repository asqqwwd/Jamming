import math, logging, global_var, threading, os, time
import numpy as np
import librosa
from utils.access import Access
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.signal as signal


class ActiveNoiseControl(threading.Thread):
    def __init__(self, nosie_lib, in_fs, in_channel, in_bit_depth,
                 chirp_length, nosie_length, snr_calculate_block_count,
                 simulation_block_count):
        threading.Thread.__init__(self)
        self.daemon = True
        self.exit_flag = False
        self.noise_lib = nosie_lib
        self.in_fs = in_fs
        self.in_channels = in_channel
        self.in_bit_depth = in_bit_depth
        self.frames_count = int((chirp_length + nosie_length) * in_fs)
        self.snr_calculate_block_count = snr_calculate_block_count
        self.simulation_block_count = simulation_block_count
        self.test_block = 1

        # Channel estimator
        self.w_bases = self.noise_lib.get_w_bases()
        self.Factor_list = []
        self.W_list = []
        for w in self.w_bases:
            self.W_list.append(self._cal_W(w))

        self.start()

    def run(self):
        # Test
        # start_index = self._find_start_burr(self.raw_input_frames)
        # print(start_index)
        # start_index -= 8000
        tmp = np.array([])
        while not self.exit_flag:
            # 1.读取输入音频数据。此过程会阻塞，直到有足够多的数据
            frames = global_var.raw_input_pool.get(self.frames_count)
            tmp = np.concatenate((tmp, frames))
            if global_var.run_time > 30:
                Access.save_data(tmp, "./input5.py")
                raise ValueError("***")

            # 2.输入定位
            chirp_noise = self.noise_lib.get_chirp_noise()
            located_frames = self._location_by_burr(frames)
            if abs(len(located_frames) - len(chirp_noise)) > 500:
                print("Located fail, continue")
                continue
            mf, pf = self._align_length(located_frames, chirp_noise)  # 不要去掉毛刺

            # 3.信道估计 or 噪声消除
            # if global_var.run_time < 10:
            #     self._channel_simulation(mf[1400:])  # 去除毛刺再输入
            # else:
            #     global_var.processed_input_pool.put(
            #         self._eliminate_noise(mf, pf))

    def stop(self):
        global_var.raw_input_pool.release()
        self.exit_flag = True
        self.join()

    def _channel_simulation(self, mixer_frames):
        # stft and divide
        _, _, Zxx = signal.stft(mixer_frames,
                                fs=self.in_fs,
                                nperseg=self.in_fs // 10,
                                noverlap=0)
        spec_mag, _ = self._divide_magphase(Zxx)
        # 寻找频率中心点
        gap = spec_mag.shape[1] / len(self.w_bases)
        index_list = list(
            map(lambda x: int(x + gap / 2),
                np.linspace(0, spec_mag.shape[1] - gap, len(self.w_bases))))
        # 计算各项系数
        for i, w in zip(index_list, self.w_bases):
            self.Factor_list.append(
                np.linalg.inv(self.W_list[i]) @ np.expand_dims(
                    self._compress_dim(self.in_fs, w, 16, spec_mag.T[i]), 1))

    def _eliminate_noise(self, mixer_frames, noise_factor_list):
        # stft and divide
        _, _, Zxx = self._stft(mixer_frames)
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
        return signal.stft(frames, fs=self.in_fs, nperseg=self.in_fs // 10)

    def _istft(self, spectrogram):
        return signal.istft(spectrogram, fs=self.in_fs)

    def _divide_magphase(self, D):
        """Separate a complex-valued stft D into its magnitude (S)
        and phase (P) components, so that `D = S * P`."""
        S = np.abs(D)
        P = np.exp(1.j * np.angle(D))
        return S, P

    def _merge_magphase(self, S, P):
        """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
        return S * P

    def _location_by_burr(self, raw_input_frames):
        # 1.找到毛刺起始和终止位置
        burr_index = self._find_burr(raw_input_frames)

        # 2.从last_input池中读取保存的上轮右半数据
        last_input_frames = global_var.last_input_pool.get_all()

        # 3.将本轮数据的右半保存至last_input池
        global_var.last_input_pool.put(raw_input_frames[burr_index:])

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

    def _cal_W(self, base_f):
        W = np.array([])
        t = np.linspace(0, 1, num=self.in_fs)
        s = np.sin(2 * np.pi * base_f * t)
        for i in range(16):
            f, _, Zxx = self._stft(np.pow(s, i))
            if len(W) == 0:
                W = np.expand_dims(
                    self._compress_dim(self.in_fs, base_f, 16,
                                       np.abs(Zxx.T)[Zxx.shape[1] // 2]), 1)
            else:
                W = np.concatenate(
                    (W,
                     np.expand_dims(
                         self._compress_dim(self.in_fs, base_f, 16,
                                            np.abs(Zxx.T)[Zxx.shape[1] // 2]),
                         1)), 1)
        return W

    def _compress_dim(self, fs, base_f, n, spec_n):
        """
        fs: sample rate
        base_f: base of frequency
        n: target dims after compress
        spec_n: spectrum at a certain moment
        """
        f_list = np.linspace(0, fs // 2, len(spec_n))
        new_spec_n = spec_n
        new_spec_n[new_spec_n < 1e-4] = 0
        re = []
        for freq in np.arange(base_f, n * base_f + base_f, base_f):
            index_list = []
            for i, f in enumerate(f_list):
                if abs(f - freq) <= 10:
                    index_list.append(i)
            re.append(np.sum(spec_n[index_list]))
        re = np.array(re)
        return re

    def _decompress_dim(self, fs, base_f, n, factors):
        """
        fs: sample rate
        base_f: base of frequency
        n: target dims after decompress
        factors: polynomial's factors
        """
        f_list = np.linspace(0, fs // 2, n)
        re = np.zeros(n)
        for freq, factor in zip(
                np.arange(base_f,
                          len(factors) * base_f + base_f, base_f), factors):
            index_list = []
            for i, f in enumerate(f_list):
                if abs(f - freq) <= 5:
                    index_list.append(i)
            print(index_list)
            re[index_list[len(index_list) // 2]] = factor
        return re
