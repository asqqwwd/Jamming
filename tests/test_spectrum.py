import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import AudioSegment
import scipy.signal as signal

file_dir = "./tests/waves"
save_dir = "./tests/spectrums"

def run():
    wav_filenames = list(filter(lambda f: ".wav" in f, os.listdir(file_dir)))
    m4a_filenames = list(filter(lambda f: ".m4a" in f, os.listdir(file_dir)))
    for filename in m4a_filenames:
        if filename[:-4] not in wav_filenames:
            audio = AudioSegment.from_file(os.path.join(file_dir, filename), "m4a")
            audio.export(os.path.join(file_dir, "{}.wav".format(filename[:-4])),
                        format="wav")
    filenames = list(filter(lambda f: ".wav" in f, os.listdir(file_dir)))

    for i, filename in enumerate(filenames):
        f = wave.open(os.path.join(file_dir, filename),
                      'rb')  # 调用wave模块中的open函数，打开语音文件。
        params = f.getparams()  # 得到语音参数
        nchannels, sampwidth, framerate, nframes = params[:
                                                          4]  # nchannels:音频通道数，sampwidth:每个音频样本的字节数，framerate:采样率，nframes:音频采样点数
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        wavaData = np.fromstring(strData,
                                 dtype=np.int16)  # 得到的数据是字符串，将字符串转为int型
        wavaData = wavaData * 1.0 / max(abs(wavaData))  # wave幅值归一化
        wavaData = np.reshape(wavaData, [nframes, nchannels]).T  # .T 表示转置
        f.close()

        #（1）绘制语谱图
        plt.figure(i)
        plt.subplot(2, 1, 1)
        plt.specgram(wavaData[0],
                     Fs=framerate,
                     scale_by_freq=True,
                     sides='default')  # 绘制频谱
        plt.title("{}".format(filename[:-4]))
        plt.xlabel('Time(s)')
        plt.ylabel('Frequency')

        #（2）绘制时域波形
        plt.subplot(2, 1, 2)
        time = np.arange(0, nframes) * (1.0 / framerate)
        time = np.reshape(time, [nframes, 1]).T
        plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")

        figmanager = plt.get_current_fig_manager()
        figmanager.window.state('zoomed')    #最大化
        plt.savefig(os.path.join(save_dir, "{}.jpg".format(filename[:-4])))
        plt.show()