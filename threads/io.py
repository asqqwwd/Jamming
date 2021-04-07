import abc, threading, random, logging, time
import pyaudio
import numpy as np

import global_var
from utils.codec import Codec
from utils.modulate import Modulate


class PyaudioInput(threading.Thread):
    def __init__(self, in_fs, in_channel, in_bit_depth, in_frames_per_buffer,
                 in_device_keyword):
        threading.Thread.__init__(self)
        # 初始化配置
        self.daemon = True
        self.exit_flag = False
        self.in_fs = in_fs
        self.in_channels = in_channel
        self.in_bit_depth = in_bit_depth
        self.in_frames_per_buffer = in_frames_per_buffer
        self.p = pyaudio.PyAudio()
        params = {
            "rate": self.in_fs,
            "channels": self.in_channels,
            "format": self.p.get_format_from_width(self.in_bit_depth // 8),
            "input": True,
            "input_device_index": None,
            "start": False
        }

        # 为输入设备创建输入流
        self.stream = None
        params["input_device_index"] = self._get_device_index_by_keyword(
            in_device_keyword)
        logging.info("Open input device [index:{}] [name:{}]".format(
            params["input_device_index"], in_device_keyword))
        self.stream = self.p.open(**params)
        # 开始线程
        self.start()

    def run(self):
        self.stream.start_stream()
        while not self.exit_flag:
            # 1.读取输入音频数据。此过程会阻塞，直到有足够多的数据
            bytes_buffer = self.stream.read(self.in_frames_per_buffer)

            # 2.将bytes流输入转换为[-1,1]的浮点数一维数组
            frames = Codec.decode_bytes_to_audio(bytes_buffer,
                                                 self.in_channels,
                                                 self.in_bit_depth)

            # 3.将一维数组保存入缓冲池
            global_var.raw_input_pool.put(frames)

            # 4.更新系统时间
            true_run_time = time.time() - global_var.start_time
            global_var.run_time += self.in_frames_per_buffer / self.in_fs
            logging.info("Run time: {}s, {}s, {}s".format(
                round(true_run_time, 2), round(global_var.run_time, 2),
                round(true_run_time - global_var.run_time, 2)))

        self.stream.stop_stream()
        self.stream.close()

    def stop(self):
        self.exit_flag = True
        self.p.terminate()
        self.join()

    def _save_data(self, data, save_fillname):
        np.save(save_fillname, data)

    def _get_device_index_by_keyword(self, keyword):
        if keyword is None:
            return None
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if keyword not in dev_info["name"]:
                continue
            if dev_info["maxInputChannels"] > 0:
                return dev_info["index"]
        p.terminate()


class PyaudioOutput(threading.Thread):
    def __init__(self, out_fs, out_channel, out_bit_depth, frames_per_buffer,
                 usb_card_keyword):
        threading.Thread.__init__(self)
        # 初始化配置
        self.daemon = True
        self.exit_flag = False
        self.out_fs = out_fs
        self.out_channels = out_channel
        self.out_bit_depth = out_bit_depth
        self.frames_per_buffer = frames_per_buffer
        self.p = pyaudio.PyAudio()
        params = {
            "rate": out_fs,
            "channels": out_channel,
            "format": self.p.get_format_from_width(out_bit_depth // 8),
            "output": True,
            "output_device_index": None,
            "start": False
        }

        # 为当前所有可用输出设备创建输出流
        self.devices = OutputDeviceIterable(device_keyword=usb_card_keyword)
        self.streams = StreamsIterable()
        for index, info in self.devices:
            logging.info("Open output device [index:{}] [name:{}]".format(
                index, info))
            params["output_device_index"] = index
            self.streams.append(self.p.open(**params))
        self.start()

    def run(self):
        for stream in self.streams:
            stream.start_stream()
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
            out_data = Codec.encode_audio_to_bytes(modulated_output_frames,
                                                   self.out_channels,
                                                   self.out_bit_depth)

            # 5.分声道输出
            for i, stream in enumerate(self.streams):
                # data = bytes(
                #     int(random.random() * 256)
                # for _ in range(len(raw_output_frames)))
                stream.write(out_data)  # 此过程会阻塞，直到填入数据被全部消耗
        for stream in self.streams:
            stream.stop_stream()
            stream.close()

    def stop(self):
        self.exit_flag = True
        self.p.terminate()
        self.join()


class DeviceIterable(abc.ABC):
    def __init__(self):
        self.index_infoss = []

    @abc.abstractmethod
    def _get_devices_by_keyword(self, device_keyword, host_api):
        pass


class InputDeviceIterable(DeviceIterable):
    def __init__(self, device_keyword="Realtek(R) Audio"):
        super(InputDeviceIterable, self).__init__(
        )  # 继承父类构造方法，也可写成DeviceIterable.__init__(self,*args)
        self._get_devices_by_keyword(device_keyword)

    def _get_devices_by_keyword(self, device_keyword):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if device_keyword not in dev_info["name"]:
                continue
            if dev_info["maxInputChannels"] > 0:
                self.index_infoss.append((dev_info["index"], dev_info["name"]))
        p.terminate()

    # __iter__要求必须返回迭代器。带有yield，当作生成器，即迭代器。
    def __iter__(self):
        index = 0
        for device_info in self.index_infoss:
            yield device_info
            index += 1


class OutputDeviceIterable(DeviceIterable):
    def __init__(self, device_keyword="Realtek(R) Audio"):
        super(OutputDeviceIterable, self).__init__()
        self._get_devices_by_keyword(device_keyword)

    def _get_devices_by_keyword(self, device_keyword):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if device_keyword not in dev_info["name"]:
                continue
            if dev_info["maxOutputChannels"] > 0 and dev_info["hostApi"] == 1:
                self.index_infoss.append((dev_info["index"], dev_info["name"]))
        p.terminate()

    # 迭代对象最简单写法，无需迭代器。index自动从0开始递增
    def __getitem__(self, index):
        return self.index_infoss[index]


class StreamsIterable():
    def __init__(self):
        self.streams = []

    def __repr__(self):
        return "Streams count [{}]".format(len(self.streams))

    # 返回迭代器，传统写法
    def __iter__(self):
        return StreamsIterator(self.streams)

    def append(self, stream):
        self.streams.append(stream)


class StreamsIterator():
    def __init__(self, streams):
        self.streams = streams
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            stream = self.streams[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return stream