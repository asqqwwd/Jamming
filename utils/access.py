import wave, os
import numpy as np
from utils.codec import Codec
from utils.resampler import Resampler


class Access():
    @classmethod
    def save_data(self, data, filename):
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename, data)

    @classmethod
    def load_data(self, filename):
        return np.load(filename)

    @classmethod
    def save_wave(self, data, filename, channel, sampwidth, framerate):
        if os.path.exists(filename):
            os.remove(filename)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channel)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(
                Codec.encode_audio_to_bytes(data, channel, sampwidth * 8))

    @classmethod
    def load_wave(self, filename):
        with wave.open(filename, "rb") as wf:
            nchannels = wf.getparams().nchannels
            sampwidth = wf.getparams().sampwidth
            framerate = wf.getparams().framerate
            nframes = wf.getparams().nframes
            bytes_buffer = wf.readframes(nframes)  # 一次性读取所有frame

            audio_clip = Codec.decode_bytes_to_audio(bytes_buffer, nchannels,
                                                     sampwidth * 8)
        return audio_clip

    @classmethod
    def load_wave_with_fs(self, filename, fs):
        with wave.open(filename, "rb") as wf:
            nchannels = wf.getparams().nchannels
            sampwidth = wf.getparams().sampwidth
            framerate = wf.getparams().framerate
            nframes = wf.getparams().nframes
            bytes_buffer = wf.readframes(nframes)  # 一次性读取所有frame

            audio_clip = Codec.decode_bytes_to_audio(bytes_buffer, nchannels,
                                                     sampwidth * 8)
        return Resampler.resample(audio_clip, framerate, fs)
