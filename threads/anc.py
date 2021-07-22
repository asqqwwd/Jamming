import math, logging, global_var, threading, os, time
from utils.statistics import Statistics
from threads.nl import NoiseLib
from sklearn.cluster import KMeans

import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.signal as signal

from utils.access import Access
from utils.mplot import MPlot
from utils.iters import SlideWindow
from utils.mfft import MFFT


class ActiveNoiseControl(threading.Thread):
    def __init__(self, nosie_lib, in_fs, chirp_length, lowest_f, noise_num):
        threading.Thread.__init__(self)
        self.daemon = True
        self.exit_flag = False
        self.noise_lib = nosie_lib
        self.in_fs = in_fs
        self.chirp_length = chirp_length
        self.lowest_f = lowest_f
        self.noise_num = noise_num
        self.noise_frames_count = int(1 / lowest_f * in_fs)
        self.chirp_frames_count = int(chirp_length * in_fs)
        self.chirp_noises_frames_count = self.chirp_frames_count + self.noise_frames_count * self.noise_num
        self.last_max_index = np.inf
        self.located_flag1 = False
        self.located_flag2 = False
        self.simualted_flag = False

        self.noise_bases = []
        self.chirp_bases = []

        self.start()

    def run(self):
        # Test
        tmp = np.array([])
        while not self.exit_flag:
            # 1.读取输入音频数据。此过程会阻塞，直到有足够多的数据
            frames = global_var.raw_input_pool.get(
                self.chirp_noises_frames_count)
            print("Time: {:.2f}".format(global_var.run_time))

            # Test: Show and save data
            tmp = np.concatenate((tmp, frames))
            if global_var.run_time > 15:
                MPlot.plot(tmp)
                # tmp = tmp[20000:]
                Statistics.eval_relative_DB(tmp)
                Statistics.eval_stablity(tmp)
                Access.save_wave(tmp, "./HW_Impulse1_Base5k.wav", 1, 2, self.in_fs)
                raise ValueError("**")

            # # 定位
            # located_frames = self._located_by_chirp(
            #     frames, self.noise_lib.get_chirp_with_fs(self.in_fs, 1))

            # # 信道估计 or 噪声消除
            # if self.located_flag1 and not self.located_flag2:
            #     self.located_flag2 = True
            #     continue
            # if self.located_flag2 and not self.simualted_flag:
            #     self._channel_simulation(located_frames,
            #                              self.noise_lib.get_keys())
            #     self.simualted_flag = True
            #     continue
            # if self.located_flag2 and self.simualted_flag:
            #     re = self._eliminate_noise(located_frames,
            #                                self.noise_lib.get_keys())
            #     tmp = np.concatenate((tmp, re))
            #     if global_var.run_time > 25:
            #         # 滤波
            #         lowpass_filter = signal.butter(
            #             8, 0.2, 'lowpass')  # 7.2k/24k=0.3 5/24k=0.2
            #         tmp = signal.filtfilt(lowpass_filter[0],
            #                                          lowpass_filter[1],
            #                                          tmp)
            #         MPlot.plot(tmp)
            #         Access.save_wave(tmp, "./AncSpeaker.wav", 1, 2, self.in_fs)
            #         raise ValueError("**")

    def stop(self):
        global_var.raw_input_pool.release()
        self.exit_flag = True
        self.join()

    def _channel_simulation(self, mixer_frames, keys):
        # 去chirp
        mf_without_chirp = mixer_frames[self.chirp_frames_count:]
        self.chirp_bases = mixer_frames[:self.chirp_frames_count]
        # MPlot.subplot([mixer_frames, mf_without_chirp], self.in_fs)
        # raise ValueError("***")
        # 长度对齐
        mf_without_chirp = self._align_length(mf_without_chirp,
                                              int(self.noise_num *
                                                  self.noise_frames_count),
                                              tp=1)  # 这里为了防止极端情况，补全200条
        # 均等切分
        sw = SlideWindow(mf_without_chirp, self.noise_frames_count,
                         self.noise_frames_count)
        # splited_frames = np.split(mf_without_chirp, self.noise_num)
        # Test
        # chirp_count = self.chirp_frames_count
        # self._align_length(self._align_length(splited_frames[0],chirp_count+len(splited_frames[0],0)),len(mixer_frames))
        # MPlot.plot_together([
        #     self._align_length(
        #         np.concatenate((np.zeros(chirp_count + len(splited_frames[0])),
        #                         splited_frames[1])), len(mixer_frames), 1),
        #     mixer_frames
        # ])
        # raise ValueError("***")
        splited_frames = list(map(lambda x: np.expand_dims(x, 1).T, sw))
        splited_frames = np.concatenate(splited_frames, axis=0)
        # 根据keys值分类
        classifed_frames = []
        n_clusters = 3
        # Test
        for i in range(np.max(keys) + 1):
            classifed_frames.append(splited_frames[self._find_index(keys, i)])
        # 每类选举出最普适的代表
        for cf in classifed_frames:
            if cf.shape[0] == 0:
                self.noise_bases.append(np.zeros(cf.shape[1]))
                continue
            clf = KMeans(n_clusters=n_clusters)
            clf.fit(cf)
            labels = clf.labels_
            classifed_labels = []
            for i in range(n_clusters):
                classifed_labels.append(self._find_index(labels, i))
            max_length_indexes = list(
                filter(
                    lambda x: len(x) == max(
                        list(map(lambda y: len(y), classifed_labels))),
                    classifed_labels))[-1]
            self.noise_bases.append(
                cf[max_length_indexes[len(max_length_indexes) // 2]])
        for i in range(len(self.noise_bases)):
            self.noise_bases[i] = np.abs(np.fft.fft(
                self.noise_bases[i]))  # 只保存基底的幅度谱
        # MPlot.subplot([mf_without_chirp] + self.noise_bases,
        #               self.in_fs)

    def _find_index(self, data, value):
        re = []
        for i, d in enumerate(data):
            if d == value:
                re.append(i)
        return re

    def _eliminate_noise(self, mixer_frames, keys):
        # 去chirp
        mf_without_chirp = mixer_frames[self.chirp_frames_count:]
        # 长度对齐
        mf_without_chirp = self._align_length(mf_without_chirp,
                                              int(self.noise_num *
                                                  self.noise_frames_count),
                                              tp=1)  # 这里为了防止极端情况，补全200条
        # 谱减法
        sw = SlideWindow(mf_without_chirp, self.noise_frames_count,
                         self.noise_frames_count)
        re = np.array([])
        for key, frames in zip(keys, sw):
            sp = np.fft.fft(frames)
            S, P = np.abs(sp), np.exp(1.j * np.angle(sp))
            new_S = S - self.noise_bases[key]
            new_S[new_S < 0] = 0  # 谱减法最简单的方式
            re = np.concatenate((re, np.fft.ifft(new_S * P).real))
        return re

    def _located_by_chirp(self, raw_input_frames, chirp_frames):
        # 1.包络检测，并找到最大值点下标。用于同步
        if not self.located_flag1:
            max_index = self._find_max(
                self._envelope(raw_input_frames, chirp_frames))
            if abs(self.last_max_index - max_index) < 50:
                print("Located success")
                self.located_flag1 = True
            self.last_max_index = max_index
        else:
            self.last_max_index -= 4  # ??? 和硬件设备有关
            max_index = self.last_max_index

        # 2.从last_input池中读取保存的上轮右半数据
        last_input_frames = global_var.last_input_pool.get_all()

        # 3.将本轮数据的右半保存至last_input池
        global_var.last_input_pool.put(raw_input_frames[max_index:])

        # 4.拼接上轮右半数据核本轮左半数据
        joined_input_frames = np.concatenate(
            (last_input_frames, raw_input_frames[:max_index]))

        return joined_input_frames

    def _align_length(self, frames, base_length, tp=0):
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

    def _envelope(self, frames, chirp):
        # 包络检测模块。会将frames分段与chirp信号进行卷积
        # frames：输入原始信号
        # chirp：用于解调原始信号的卷积信号
        # res：返回的卷积结果
        N1 = len(frames)
        N2 = len(chirp)

        res = []
        i = 0
        while i < N1:
            N = min(N2, N1 - i)
            chirp_freq = np.fft.fft(chirp[0:N])
            frames_freq = np.fft.fft(frames[i:i + N])
            tmp = frames_freq * chirp_freq
            tmp = list(np.zeros((math.floor(N / 2) + 1))) + list(
                tmp[math.floor(N / 2) - 1:N + 1] * 2)
            res = res + list(abs(np.fft.ifft(tmp)))[0:N - 1]
            i = i + N2

        return np.array(res)

    def _find_max(self, frames):
        # 若Clip中有一个最大值点，则输出其下标。若有两个值大小差距在阈值(0.3)以内的最大值点，则输出其下标的中位点
        # frames：需要检测最大值点的输入声音片段
        # max_index：返回的最大值点下标
        first_max_index = 0
        for i in range(1, len(frames)):
            if frames[i] > frames[first_max_index]:
                first_max_index = i

        second_max_index = 0
        for i in range(1, len(frames)):
            if frames[i] > frames[second_max_index] and i != first_max_index:
                second_max_index = i

        threshold = 0.3
        if frames[first_max_index] / frames[second_max_index] <= 1 + threshold:
            max_index = math.floor((first_max_index + second_max_index) / 2)
        else:
            max_index = first_max_index
        return max_index
