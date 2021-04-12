import math, logging, global_var, threading
import numpy as np


class ActiveNoiseControl(threading.Thread):
    def __init__(self, nosie_lib, in_fs, in_channel, in_bit_depth,
                 chirp_nosie_length, simulation_length):
        threading.Thread.__init__(self)
        self.daemon = True
        self.exit_flag = False
        self.noise_lib = nosie_lib
        self.in_fs = in_fs
        self.in_channels = in_channel
        self.in_bit_depth = in_bit_depth
        self.chirp_nosie_frames_count = int(chirp_nosie_length * in_fs)
        self.simulation_length = simulation_length
        self.H = np.array([])
        
        self.start()

    def run(self):
        self.c1 = 0
        self.c2 = 0
        while not self.exit_flag:
            # 1.读取输入音频数据。此过程会阻塞，直到有足够多的数据
            frames = global_var.raw_input_pool.get(
                self.chirp_nosie_frames_count)

            # 2.输入定位
            located_frames = self._location(
                frames, self.noise_lib.get_down_chirp(self.in_fs))

            # 3.信道估计 or 噪声消除
            if global_var.run_time < self.simulation_length:
                self._channel_simulation(
                    located_frames, self.noise_lib.get_chirp_noise(self.in_fs))
            else:
                processed_input_frames = self._eliminate_noise(
                    located_frames, self.noise_lib.get_chirp_noise(self.in_fs))
                global_var.processed_input_pool.put(processed_input_frames)

    def stop(self):
        global_var.raw_input_pool.release()
        self.exit_flag = True
        self.join()

    def _location(self, raw_input_frames, chirp_frames):
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

    def _channel_simulation(self, reality_frames, ideal_frames):
        pass
        logging.info("System Clock-{}(s)-Channel simulation".format(
            round(global_var.run_time, 2)))
        print(len(reality_frames), len(ideal_frames))
        # self.tmp = np.concatenate((self.tmp,ideal_frames))
        #self._save_data(reality_frames,
        #                "./tests/saved/train_x{}.npy".format(self.c1))
        #self._save_data(ideal_frames,
        #                "./tests/saved/train_y{}.npy".format(self.c1))
        self.c1 += 1

    def _eliminate_noise(self, reality_frames, ideal_frames):
        logging.info("System Clock-{}(s)-Eliminate noise".format(
            round(global_var.run_time, 2)))
        print(len(reality_frames), len(ideal_frames))
        #self._save_data(reality_frames,
        #                "./tests/saved/test_x{}.npy".format(self.c2))
        #self._save_data(ideal_frames,
        #                "./tests/saved/test_y{}.npy".format(self.c2))
        self.c2 += 1
        return reality_frames

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
            frames_freq = np.fft.fft(frames[i:i + N])
            chirp_freq = np.fft.fft(chirp[0:N])
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

    def _save_data(self, data, save_fillname):
        np.save(save_fillname, data)
