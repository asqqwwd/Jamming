import abc, threading, random, logging, time
import spidev
import numpy as np

import global_var
from utils.codec import Codec
from utils.modulate import Modulate


class SpiOutput(threading.Thread):
    def __init__(self, out_fs, frames_per_buffer):
        threading.Thread.__init__(self)
        # 初始化配置
        self.daemon = True
        self.exit_flag = False
        self.out_fs = out_fs
        self.frames_per_buffer = frames_per_buffer
        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = 976000  # 97600/4/2=122khz
        self.start()

    def run(self):
        self._spi_init()
        while not self.exit_flag:
            # 1.如果keyword池非空，则读取数据。跳3
            if not global_var.keyword_pool.is_empty():
                raw_output_frames = global_var.keyword_pool.get(
                    self.frames_per_buffer)
            # 2.如果keyword池为空，直接读取noise池中数据。跳3
            else:
                raw_output_frames = global_var.noise_pool.get(
                    self.frames_per_buffer)

            # 3.调制
            modulated_output_frames = Modulate.am_modulate(
                raw_output_frames, 2, self.out_fs)

            # 4.将[-1,1]的浮点数一维数组转换为bytes流输出
            modulated_output_frames = (modulated_output_frames[0],
                                       modulated_output_frames[1],
                                       modulated_output_frames[0],
                                       modulated_output_frames[1])
            self.spi.writebytes2(self._encode(modulated_output_frames))

    def stop(self):
        self.exit_flag = True
        self.spi.close()
        self.join()

    def _spi_init(self):
        command = 0b01010001
        data1 = 0b00000000
        data2 = 0b00000000
        self.spi.writebytes([command, data1, data2])

    def _encode(self, frames):
        commandA = 0b00110000.to_bytes(1, byteorder='little', signed=False)
        commandB = 0b00110001.to_bytes(1, byteorder='little', signed=False)
        commandC = 0b00110010.to_bytes(1, byteorder='little', signed=False)
        commandD = 0b00110011.to_bytes(1, byteorder='little', signed=False)
        re = []
        ch1 = (frames[0] * 2**12).clip(-2**12,
                                       2**12 - 1).astype(np.int16).tobytes()
        ch2 = (frames[1] * 2**12).clip(-2**12,
                                       2**12 - 1).astype(np.int16).tobytes()
        ch3 = (frames[2] * 2**12).clip(-2**12,
                                       2**12 - 1).astype(np.int16).tobytes()
        ch4 = (frames[3] * 2**12).clip(-2**12,
                                       2**12 - 1).astype(np.int16).tobytes()
        for i in len(ch1):
            re.append(commandA)
            re.append(self._reverse_bit_in_bytes(ch1[i][0]))
            re.append(self._reverse_bit_in_bytes(ch1[i][1]))
            re.append(commandB)
            re.append(self._reverse_bit_in_bytes(ch2[i][0]))
            re.append(self._reverse_bit_in_bytes(ch2[i][1]))
            re.append(commandC)
            re.append(self._reverse_bit_in_bytes(ch3[i][0]))
            re.append(self._reverse_bit_in_bytes(ch3[i][1]))
            re.append(commandD)
            re.append(self._reverse_bit_in_bytes(ch4[i][0]))
            re.append(self._reverse_bit_in_bytes(ch4[i][1]))
        return re

    def _reverse_bit_in_bytes(self, byte):
        byte = ((byte & 0xF0) >> 4) | ((byte & 0x0F) << 4)
        byte = ((byte & 0xCC) >> 2) | ((byte & 0x33) << 2)
        byte = ((byte & 0xAA) >> 1) | ((byte & 0x55) << 1)
        return byte
