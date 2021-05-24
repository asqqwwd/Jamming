import threading, wave, math
import numpy as np

import global_var
from utils.codec import Codec
from utils.resampler import Resampler
from utils.access import Access


class NoiseLib(threading.Thread):
    def __init__(self, out_fs, chirp_length, noise_length):
        threading.Thread.__init__(self)
        # 配置噪声库
        self.daemon = True
        self.exit_flag = False  # 线程退出标志
        self.out_fs = out_fs
        self.chirp_length = chirp_length
        self.noise_length = noise_length
        self.f_lower_bound = 100  # 噪声频率下界
        self.f_upper_bound = 1000  # 噪声频率上界
        self.num_of_base = 19  # 噪声基底个数
        self.chirp = self._generate_chirp()
        self.noise = np.array([])
        # tmp = Access.load_wave("./waves/raw/offer.wav")
        # tmp = tmp[len(tmp) // 4:len(tmp) // 4 * 3]
        # self.test_wave = Resampler.resample(tmp, 44100, 96000)
        # 开始运行线程
        self.start()

    def run(self):
        flag = False
        while not self.exit_flag:
            if not flag:
                global_var.noise_pool.put(
                    self._generate_simulation_signal(self.out_fs, 19))
                flag = True
            global_var.noise_pool.put(
                np.concatenate((self.chirp[0], self._generate_noise())))
            # global_var.noise_pool.put(self.test_wave)  # Test
            # global_var.noise_pool.put(self.noise_frames)  # Test

    def stop(self):
        global_var.noise_pool.release()
        self.exit_flag = True
        self.join()

    # 生成初始测试信号
    def _generate_simulation_signal(self, fs, length):
        ss_count = int(fs * length)
        step_length = length / self.num_of_base
        step = ss_count // self.num_of_base
        w_bases = np.linspace(self.f_lower_bound, self.f_upper_bound,
                              self.num_of_base)
        ss = np.zeros(ss_count)
        t = np.linspace(0, step_length, step)
        for i in range(self.num_of_base):
            ss[i * step:(i + 1) * step] = np.sin(2 * np.pi * w_bases[i] * t)
        return ss

    # 每次生成不同噪声库噪声
    def _generate_noise(self):
        noise_frames_count = int(self.out_fs * self.noise_length)
        random_factor = np.random.random(self.num_of_base)
        random_factor2 = np.random.random(self.num_of_base)
        # random_factor = np.ones(self.num_of_base)  # Test

        w_bases = np.linspace(self.f_lower_bound, self.f_upper_bound,
                              self.num_of_base)
        t = np.linspace(0, self.noise_length, num=noise_frames_count)
        random_noise = np.zeros(noise_frames_count)
        for i in range(self.num_of_base):
            random_noise += random_factor[i] * np.sin(
                2 * np.pi * w_bases[i] * (t - np.pi * random_factor2[i] / 2))
        random_noise = random_noise / np.max(np.abs(random_noise))
        return random_noise

    def _generate_chirp(self):
        up_chirp = down_chirp = np.zeros(int(self.out_fs * self.chirp_length))
        return up_chirp, down_chirp

    def get_chirp(self, tp=1):
        return self.chirp[tp]

    def get_chirp_noise(self, dst_fs, tp=1):
        return Resampler.resample(np.concatenate((self.chirp[tp], self.noise)),
                                  self.out_fs, dst_fs)

    def get_w_bases(self):
        return np.linspace(self.f_lower_bound, self.f_upper_bound,
                           self.num_of_base)
