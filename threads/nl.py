import threading, wave, math
from utils.mplot import MPlot
import numpy as np

import global_var
from utils.codec import Codec
from utils.resampler import Resampler
from utils.access import Access
from utils.mplot import MPlot
import matplotlib.pyplot as plt


class NoiseLib(threading.Thread):
    def __init__(self, out_fs, chirp_length, lowest_f, noise_num):
        threading.Thread.__init__(self)
        # 配置噪声库
        self.daemon = True
        self.exit_flag = False  # 线程退出标志
        self.out_fs = out_fs
        self.chirp_length = chirp_length
        self.f1 = 100
        self.f2 = 10e3

        self.lowest_f = lowest_f
        self.noise_num = noise_num
        self.noise_bases_num = 2
        self.noise_length = 1 / self.lowest_f

        self.chirp = self._generate_chirp()
        self.noise_bases = self._generate_noise_bases()
        self.keys = np.array([])

        # 开始运行线程
        self.start()

    def run(self):
        while not self.exit_flag:
            global_var.noise_pool.put(self._concatenate_chirp_noise())
            # global_var.noise_pool.put(self.test_wave)  # Test
            # global_var.noise_pool.put(self.noise_frames)  # Test

    def stop(self):
        global_var.noise_pool.release()
        self.exit_flag = True
        self.join()

    # 生成噪声基底
    def _generate_noise_bases(self):
        re = []
        t = np.linspace(0, self.noise_length,
                        int(self.out_fs * self.noise_length))

        # # Phase
        # # random_factors = [0.77968736, 2.70990493]
        # # random_factors = []
        # # for _ in range(2):
        # #     tmp = []
        # #     for _ in range(4):
        # #         tmp.append(np.round(np.random.rand()*np.pi,3))
        # #     random_factors.append(tmp)
        # # print(random_factors)
        # # random_factors = [[0.345, 0.559, 0.634, 2.522], [2.108, 1.028, 0.699, 2.015]]
        # random_factors = [[0.345, 0, 0, 0], [2.108, 0, 0, 0]]
        # for rf in random_factors:
        #     # tmp =  np.sin(2 * np.pi * self.lowest_f * t + rf[0]) +\
        #     # np.sin(2 * np.pi * 2 * self.lowest_f * t + rf[1]) +\
        #     # np.sin(2 * np.pi * 4 * self.lowest_f * t + rf[2]) +\
        #     # np.sin(2 * np.pi * 8 * self.lowest_f * t + rf[3])
        #     tmp = np.sin(2 * np.pi * self.lowest_f * t + rf[0])
        #     # tmp = tmp / np.max(np.abs(tmp))  # 加上这个会导致基功率不一致，不会改善振铃效应
        #     re.append(tmp)

        # # Freq and mag
        # random_factors = [[0.61, 0.36, 0.99, 0.35, 0.21], [1, 0, 0, 0, 0]]
        # for rf in random_factors:
        #     tmp = rf[0] * np.sin(2 * np.pi * self.lowest_f * t) -\
        #         rf[1] * np.sin(2 * np.pi * 2 * self.lowest_f * t) +\
        #         rf[2] * np.sin(2 * np.pi * 4 * self.lowest_f * t) +\
        #         rf[3] * np.sin(2 * np.pi * 8 * self.lowest_f * t) -\
        #         rf[4] * np.sin(2 * np.pi * 16 * self.lowest_f * t)
        #     tmp = tmp / np.max(np.abs(tmp))
        #     re.append(tmp)

        # Continuous change phase
        random_factors = []
        w_bases = np.arange(100, 3100, 100)
        for i in range(self.noise_bases_num):
            # random_factors.append(2 * np.random.randint(0, 2, len(w_bases)) -
            #                       1)
            # Test
            if i == 0:
                random_factors.append(
                    np.array([
                        1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1,
                        1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1
                    ]))
            else:
                random_factors.append(
                    np.array([
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0
                    ]))
        for rfs in random_factors:
            tmp = np.zeros(len(t))
            for w, rf in zip(w_bases, rfs):
                tmp += rf * np.sin(2 * np.pi * w * t)
            tmp = tmp / np.max(np.abs(tmp))
            re.append(tmp)

        return re

    def _generate_chirp(self):
        t = np.linspace(0,
                        self.chirp_length,
                        num=int(self.out_fs * self.chirp_length))
        up_chirp = np.cos(2 * np.pi * self.f1 * t +
                          (np.pi *
                           (self.f2 - self.f1) / self.chirp_length) * t**2)
        down_chirp = np.cos(2 * np.pi * self.f2 * t -
                            (np.pi *
                             (self.f2 - self.f1) / self.chirp_length) * t**2)
        return up_chirp, down_chirp

    # 随机拼接噪声
    def _concatenate_chirp_noise(self):
        # 生成随机密钥
        # self.keys = np.random.randint(self.noise_bases_num,
        #                                    size=self.noise_num)
        # self.keys = np.array([
        #     1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        #     0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
        #     1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1,
        #     0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1,
        #     1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
        #     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        #     0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
        #     0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0,
        #     0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        #     1, 1
        # ])
        # self.keys = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
        self.keys = []
        while len(self.keys) < self.noise_num:
            self.keys += [0] * 20 + [0] * 20

        # 根据key生成noise
        re = np.array([])
        for key in self.keys:
            re = np.concatenate((re, self.noise_bases[key]))

        # 拼接chirp和noise
        re = re
        # re = np.concatenate((self.chirp[0], re))
        # re = np.concatenate((np.zeros(len(self.chirp[0])), re))
        # re = np.zeros(len(re))
        # re[0] = 1
        # re = self.chirp[0].copy()

        return re

    def get_chirp_with_fs(self, fs, tp=1):
        return Resampler.resample(self.chirp[tp], self.out_fs, fs)

    def get_keys(self):
        return self.keys
