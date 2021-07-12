import abc, threading, random, logging, time
import pyaudio
import numpy as np

import global_var
from utils.codec import Codec
from utils.modulate import Modulate
from utils.access import Access


class PyaudioIO(threading.Thread):
    def __init__(self, in_fs, in_channel, in_bit_depth, in_frames_per_buffer,
                 in_device_keyword, in_host_api, out_fs, out_channel,
                 out_bit_depth, out_frames_per_buffer, out_device_keyword,
                 out_host_api):
        # 初始化配置
        PyaudioIO.in_fs = in_fs
        PyaudioIO.in_channel = in_channel
        PyaudioIO.in_bit_depth = in_bit_depth
        PyaudioIO.in_frames_per_buffer = in_frames_per_buffer

        PyaudioIO.out_fs = out_fs
        PyaudioIO.out_channel = out_channel
        PyaudioIO.out_bit_depth = out_bit_depth
        PyaudioIO.out_frames_per_buffer = out_frames_per_buffer

        self.p = pyaudio.PyAudio()
        params_in = {
            "rate":
            in_fs,
            "channels":
            in_channel,
            "format":
            self.p.get_format_from_width(in_bit_depth // 8),
            "input":
            True,
            "input_device_index":
            self._get_device_index_by_keyword(in_device_keyword, in_host_api,
                                              0),
            "start":
            False,
            "frames_per_buffer":
            in_frames_per_buffer,
            "stream_callback":
            PyaudioIO.callback_in
        }
        params_out = {
            "rate":
            out_fs,
            "channels":
            out_channel,
            "format":
            self.p.get_format_from_width(out_bit_depth // 8),
            "output":
            True,
            "output_device_index":
            self._get_device_index_by_keyword(out_device_keyword, out_host_api,
                                              1),
            "start":
            False,
            "frames_per_buffer":
            out_frames_per_buffer,
            "stream_callback":
            PyaudioIO.callback_out
        }

        # 为输入设备创建输入流
        self.stream_in = self.p.open(**params_in)
        self.stream_out = self.p.open(**params_out)
        logging.info("Open input device [index:{}] [name:{}]".format(
            params_in["input_device_index"], in_device_keyword))
        logging.info("Open output device [index:{}] [name:{}]".format(
            params_out["output_device_index"], out_device_keyword))

        # 开始
        self.start()

    @classmethod
    def callback_in(cls, in_data, frame_count, time_info, status):
        # 1.将bytes流输入转换为[-1,1]的浮点数一维数组
        frames = Codec.decode_bytes_to_audio(in_data, PyaudioIO.in_channel,
                                             PyaudioIO.in_bit_depth)

        # 2.将一维数组保存入缓冲池
        # global_var.raw_input_pool.put(
        #     global_var.test_pool.get(PyaudioIO.in_frames_per_buffer))  # Test
        global_var.raw_input_pool.put(frames)

        # 3.更新系统时间
        true_run_time = time.time() - global_var.start_time
        global_var.run_time += PyaudioIO.in_frames_per_buffer / PyaudioIO.in_fs
        # logging.info("Run time: {:.2f}s, {:.2f}s, {:.2f}s".format(
        #     true_run_time, global_var.run_time,
        #     true_run_time - global_var.run_time))
        return (None, pyaudio.paContinue)

    @classmethod
    def callback_out(cls, in_data, frame_count, time_info, status):
        # 1.如果keyword池非空，则读取数据。跳3
        if not global_var.keyword_pool.is_empty():
            raw_output_frames = global_var.keyword_pool.get(
                PyaudioIO.out_frames_per_buffer)
        # 2.如果keyword池为空，直接读取noise池中数据。跳3
        else:
            raw_output_frames = global_var.noise_pool.get(
                PyaudioIO.out_frames_per_buffer)

        # 3.调制
        modulated_output_frames = Modulate.am_modulate(raw_output_frames,
                                                       PyaudioIO.out_channel,
                                                       PyaudioIO.out_fs)

        # 4.将[-1,1]的浮点数一维数组转换为bytes流输出
        out_data = Codec.encode_audio_to_bytes(modulated_output_frames,
                                               PyaudioIO.out_channel,
                                               PyaudioIO.out_bit_depth)

        return (out_data, pyaudio.paContinue)

    def start(self):
        self.stream_in.start_stream()
        self.stream_out.start_stream()

        # # Test
        # tmp = Access.load_data("./test2_phase_random.npy")
        # global_var.test_pool.put(10 * tmp)

    def stop(self):
        self.stream_out.stop_stream()
        self.stream_out.close()
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.p.terminate()
        self.join()

    def _get_device_index_by_keyword(self, keyword, host_api, tp):
        if keyword is None:
            return None
        p = pyaudio.PyAudio()
        re = None
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if keyword not in dev_info["name"]:
                continue
            if dev_info["maxInputChannels" if tp ==
                        0 else "maxOutputChannels"] > 0 and dev_info[
                            "hostApi"] == host_api:
                re = dev_info["index"]
                break
        p.terminate()
        if not re:
            logging.info(
                "Can't find {} device, use default device".format(keyword))
        return re