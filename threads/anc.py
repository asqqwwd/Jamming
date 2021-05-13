import math, logging, global_var, threading, os, time
import numpy as np
import librosa
from utils.access import Access
from itertools import groupby
import matplotlib.pyplot as plt


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
        self.H = np.array([])
        self.V = np.array([])

        self.start()

    def run(self):
        # Test
        # start_index = self._find_start_burr(self.raw_input_frames)
        # print(start_index)
        # start_index -= 8000
        # tmp = np.array([])
        while not self.exit_flag:
            # 1.读取输入音频数据。此过程会阻塞，直到有足够多的数据
            frames = global_var.raw_input_pool.get(self.frames_count)

            # 2.噪声信号平均功率计算
            if self.snr_calculate_block_count > 0:
                self.snr_calculate_block_count -= 1
                self.noise_power = self._get_snr(frames)
                print("SNR", self.noise_power)
            # 3.信道估计
            elif self.simulation_block_count > 0:
                self.simulation_block_count -= 1

                located_frames = self._location_by_burr(frames)  # 输入定位
                # located_frames = self._location_by_chirp(
                #     frames, self.noise_lib.get_chirp(1))

                self._channel_simulation_slr(located_frames,
                                             self.noise_lib.get_chirp_noise(
                                                 self.in_fs),
                                             tp=1)
            # 4.噪声消除
            else:
                located_frames = self._location_by_burr(frames)  # 输入定位
                # located_frames = self._location_by_chirp(
                #     frames, self.noise_lib.get_chirp(1))

                processed_input_frames = self._eliminate_noise(
                    located_frames,
                    self.noise_lib.get_chirp_noise(self.in_fs),
                    tp=1)
                global_var.processed_input_pool.put(processed_input_frames)

                self.test_block -= 1
                if self.test_block == 0:
                    plt.figure(1)
                    plt.plot(global_var.processed_input_pool.get_all())
                    plt.show()
                    break

                # Test
                # print("Frame", np.mean(np.abs(located_frames)))
                # print("H", np.mean(self.H))
                # print("Quality", np.mean(np.abs(processed_input_frames)))

    def stop(self):
        global_var.raw_input_pool.release()
        self.exit_flag = True
        self.join()

    def _location_by_chirp(self, raw_input_frames, chirp_frames):
        # 1.包络检测，并找到最大值点下标。用于同步
        max_index = self._find_max(
            self._envelope(raw_input_frames, chirp_frames))

        # 2.从last_input池中读取保存的上轮右半数据
        last_input_frames = global_var.last_input_pool.get_all()

        # 3.将本轮数据的右半保存至last_input池
        global_var.last_input_pool.put(raw_input_frames[max_index:])

        # 4.拼接上轮右半数据核本轮左半数据
        joined_input_frames = np.concatenate(
            (last_input_frames, raw_input_frames[:max_index]))

        return joined_input_frames

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

    def _channel_simulation_slr(self, mixer_frames, pure_frames, tp=1):
        """
        simple linear regression：认为mixer频率点的y只与pure对应频率x点有关
        tp: 0 => 估计mixer_k，1 => 估计pure_k
        Notes: 估计pure_k效果会好很多
        """
        logging.info(
            "System Clock-{:.2f}(s)-Channel simulation [{} {}]".format(
                global_var.run_time, len(mixer_frames), len(pure_frames)))
        # align
        mf, pf = self._align_length(mixer_frames, pure_frames)
        # stft and divide
        mf_spec_mag, _ = self._divide_magphase(self._fft(mf))  # mixer
        pf_spec_mag, _ = self._divide_magphase(self._fft(pf))  # pure
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
        n = 13
        # plt.figure(10 - self.simulation_block_count)
        # plt.plot(X[n], color="g")
        # plt.plot(Y[n], color="b")
        # plt.specgram(pf,
        #              Fs=16000,
        #              scale_by_freq=True,
        #              sides='default')  # 绘制频谱
        tmp1 = X[n]  # pure
        tmp2 = Y[n]  # mixer
        plt.figure(10-self.simulation_block_count)
        plt.scatter(tmp1,tmp2)
        plt.figure(20-self.simulation_block_count)
        plt.plot(tmp2)
        # plt.plot(tmp1, beta1[n] * tmp1 + beta0[n])

    def _channel_simulation_mlr(self, mixer_frames, pure_frames):
        """
        multiple linear regression：认为mixer频率点的y与pure对应频率x点以及附近频率有关
        """
        pass

    def _eliminate_noise(self, mixer_frames, pure_frames, tp=1):
        """
        tp: 1 => k*mixer-pure; 2 => mixer-k*pure
        Notes: 不要在频域直接相减，变换到时域后在相减，在频域相减效果很差
        """
        logging.info("System Clock-{:.2f}(s)-Elininate noise [{} {}]".format(
            global_var.run_time, len(mixer_frames), len(pure_frames)))
        # align
        mf, pf = self._align_length(mixer_frames, pure_frames)
        # stft and divide
        mf_spec_mag, mf_spec_phase = self._divide_magphase(
            self._fft(mf))  # mixer
        pf_spec_mag, pf_spec_phase = self._divide_magphase(
            self._fft(pf))  # pure
        re_spec_mag = np.zeros(shape=mf_spec_mag.shape)
        # filter
        if tp == 0:
            for i in range(re_spec_mag.shape[0]):
                re_spec_mag[i] = mf_spec_mag[i] * self.H + self.V
            # merge and istft
            return self._ifft(self._merge_magphase(re_spec_mag,
                                                   mf_spec_phase)) - pf
        else:
            for i in range(re_spec_mag.shape[0]):
                re_spec_mag[i] = pf_spec_mag[i] * self.H + self.V
            # merge and istft
            return mf - self._ifft(
                self._merge_magphase(re_spec_mag, pf_spec_phase))

    def _fft(self, frames):
        """
        Return: (n_frame,n_fft)
        """
        return librosa.stft(frames,
                            n_fft=2048,
                            hop_length=160,
                            window="hamming").T

    def _ifft(self, frames_spectrum):
        return librosa.istft(frames_spectrum.T,
                             hop_length=160,
                             window="hamming")

    def _divide_magphase(self, D, power=1):
        """Separate a complex-valued stft D into its magnitude (S)
        and phase (P) components, so that `D = S * P`."""
        S = np.abs(D)
        S **= power
        P = np.exp(1.j * np.angle(D))
        return S, P

    def _merge_magphase(self, S, P):
        """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
        return S * P

    def _align_length(self, frames, base_frames):
        base_length = len(base_frames)
        aligned_frames = None
        if len(frames) < base_length:
            aligned_frames = np.concatenate(
                (frames, np.zeros(base_length - len(frames))))
        else:
            aligned_frames = frames[:base_length]
        return aligned_frames, base_frames

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
            frames_freq = np._fft._fft(frames[i:i + N])
            chirp_freq = np._fft._fft(chirp[0:N])
            tmp = frames_freq * chirp_freq
            tmp = list(np.zeros((math.floor(N / 2) + 1))) + list(
                tmp[math.floor(N / 2) - 1:N + 1] * 2)
            res = res + list(abs(np._fft._ifft(tmp)))[0:N - 1]
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

    def _find_burr(self, frames):
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
        # print("New SNR", self._get_snr(frames[tmp[re][0]:tmp[re][1]]))
        return tmp[re][0]  # 这里使用最大值定位效果比min好，但仍然使用最小值

    def _get_snr(self, frames):  # 计算噪声平均功率
        n = len(frames)
        tot = 0
        for i in range(0, n):
            tot += frames[i]**2
        return tot / n
