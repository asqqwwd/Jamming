import wave, os
import numpy as np
from utils.codec import Codec
from utils.resampler import Resampler


class Access():
    @classmethod
    def save_data(cls, data, filename):
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename, data)

    @classmethod
    def load_data(cls, filename):
        return np.load(filename)

    @classmethod
    def save_wave(cls, data, filename, channel, sampwidth, framerate):
        if os.path.exists(filename):
            os.remove(filename)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channel)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(
                Codec.encode_audio_to_bytes(data, channel, sampwidth * 8))

    @classmethod
    def load_wave(cls, filename):
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
    def load_wave_with_fs(cls, filename, fs):
        with wave.open(filename, "rb") as wf:
            nchannels = wf.getparams().nchannels
            sampwidth = wf.getparams().sampwidth
            framerate = wf.getparams().framerate
            nframes = wf.getparams().nframes
            bytes_buffer = wf.readframes(nframes)  # 一次性读取所有frame

            audio_clip = Codec.decode_bytes_to_audio(bytes_buffer, nchannels,
                                                     sampwidth * 8)
        return Resampler.resample(audio_clip, framerate, fs)

    @classmethod
    def save_txt(cls, data, filename):
        # head = [
        #     "RIGOL:DG5:CSV DATA FILE", "TYPE:Arb", "AMP:1.0000 Vpp",
        #     "PERIOD:1.00E-6 S", "DOTS:" + str(len(data)), "MODE:Normal",
        #     "AFG Frequency:1000000.000000", "AWG N:0", "x,y[V]"
        # ]
        head = [
            "RIGOL:DG5:CSV DATA FILE", "TYPE:Arb", "AMP:1.0000 Vpp",
            "PERIOD:1.00E-6 S", "DOTS:" + str(len(data)), "MODE:Normal",
            "AFG Frequency:1000000.0000000", "AWG N:0", "x,y[V]"
        ]

        with open(filename, "w") as wf:
            wf.write("\n".join(head).strip())  # write只能写入字符，writelines只能写入字符列表
            wf.write("\n," +
                     "\n,".join(list(map(lambda x: "{:.4f}".format(x), data))))

    @classmethod
    def save_csv(cls, data, filename):
        Access.save_txt(data, filename)
