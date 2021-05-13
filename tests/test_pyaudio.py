import pyaudio, random, abc, threading,wave
import numpy as np
from utils.codec import Codec
from utils.resampler import Resampler
from utils.pool import PoolCycle
from utils.modulate import am_modulate


class PyaudioOutput(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # 配置输出线程
        self.daemon = True
        self.exit_flag = False
        self._load_wave("./waves/raw/id10003-09-2.wav")
        self.p = pyaudio.PyAudio()
        params = {
            "rate": 96000,
            "channels": 2,
            "format": self.p.get_format_from_width(2),
            "output": True,
            "output_device_index": None,
            "start": False
        }

        # 为当前所有可用输出设备创建输出流
        self.devices = OutputDeviceIterable()
        self.streams = StreamsIterable()
        for device_index in self.devices:
            print(device_index)
            params["output_device_index"] = device_index
            self.streams.append(self.p.open(**params))
        self.start()

    def run(self):
        for stream in self.streams:
            stream.start_stream()
        while not self.exit_flag:
            for i, stream in enumerate(self.streams):
                self.pool.get(1024)
                # data = bytes(int(random.random() * 256) for _ in range(1024))
                stream.write(data)  # 此过程会阻塞，直到填入数据被全部消耗
        for stream in self.streams:
            stream.stop_stream()
            stream.close()

    def stop(self):
        self.exit_flag = True
        self.p.terminate()
        self.join()

    def _load_wave(self, filename):
        # 根据文件名读取音频文件
        try:
            wf = wave.open(filename, "rb")
            print(wf.getparams())
            nchannels = wf.getparams().nchannels
            sampwidth = wf.getparams().sampwidth
            framerate = wf.getparams().framerate
            nframes = wf.getparams().nframes
            bytes_buffer = wf.readframes(nframes)  # 一次性读取所有frame

            audio_clip = Codec.decode_bytes_to_audio(bytes_buffer, nchannels,
                                                     sampwidth * 8)

            audio_clip = Resampler.resample(audio_clip, framerate, 96000)
            self.pool = PoolCycle()
            self.pool.put(audio_clip)
            

        except:
            raise TypeError("Can't read wave file!")


class DeviceIterable(abc.ABC):
    def __init__(self):
        self.devices_info = []

    @abc.abstractmethod
    def _get_devices_by_keyword(self, device_keyword, host_api):
        pass


class OutputDeviceIterable(DeviceIterable):
    def __init__(self, device_keyword="Realtek USB2.0 Audio", host_api=0):
        super(OutputDeviceIterable, self).__init__()
        self._get_devices_by_keyword(device_keyword, host_api)
        print(self.devices_info)

    def _get_devices_by_keyword(self, device_keyword, host_api):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if device_keyword not in dev_info["name"]:
                continue
            if dev_info["maxOutputChannels"] > 0 and dev_info[
                    "hostApi"] == host_api:
                self.devices_info.append((dev_info["name"], dev_info["index"]))
        del self.devices_info[1:]
        p.terminate()

    # 迭代对象最简单写法，无需迭代器。index自动从0开始递增
    def __getitem__(self, index):
        return self.devices_info[index][-1]


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


def run():
    pi = PyaudioOutput()
    input("Press any key to exit>>>")
    pi.stop()


if __name__ == "__main__":
    run()
