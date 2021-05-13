from pydub import AudioSegment
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import groupby

from utils.codec import Codec
from utils.resampler import Resampler


def _load_wave(filename):
    # 根据文件名读取音频文件
    try:
        with wave.open(filename, "rb") as wf:
            print(wf.getparams())
            nchannels = wf.getparams().nchannels
            sampwidth = wf.getparams().sampwidth
            framerate = wf.getparams().framerate
            nframes = wf.getparams().nframes
            bytes_buffer = wf.readframes(nframes)  # 一次性读取所有frame

        audio_clip = Codec.decode_bytes_to_audio(bytes_buffer, nchannels,
                                                 sampwidth * 8)
        # if isinstance(audio_clip, tuple):
        #     audio_clip = Resampler.resample(audio_clip[1], framerate, 16000)
        # else:
        #     audio_clip = Resampler.resample(audio_clip, framerate, 16000)
        return audio_clip
    except:
        raise TypeError("Can't read wave file!")


def _find_burr(frames):
    abs_of_frames = np.abs(frames)
    mean_pow = np.mean(abs_of_frames)
    index_list = []
    for i, value in enumerate(frames):
        if value < -mean_pow * 7:
            index_list.append(i)
    return _group_continous_number(index_list)


def _group_continous_number(data):
    re = []
    fun = lambda x: x[1]-x[0]
    for k, g in groupby(enumerate(data), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 10:
            re.append(min(l1))
    return re


def run():
    # audio = AudioSegment.from_file("./offer_record.m4a", "m4a")
    # audio.export("./offer_record.wav", format="wav")
    wave_record = _load_wave("./tests/noise_record.wav")

    plt.figure(1)
    plt.scatter(np.linspace(0, len(wave_record)/16000, num=len(wave_record)),wave_record, s=0.2)
    plt.title("Record")

    print(_find_burr(wave_record))

    plt.show()



if __name__ == "__main__":
    run()