import math, logging
from itertools import groupby
from re import X
from utils.mplot import MPlot
import numpy as np
import wave
import pylab as pl
import numpy as np
from utils.iters import SlideWindow
from sklearn.cluster import KMeans


class Statistics():
    FRAMESIZE = 256

    # method 1: absSum
    @classmethod
    def calVolume(cls, frames, frameSize, overLap):
        wlen = len(frames)
        step = frameSize - overLap
        frameNum = int(math.ceil(wlen * 1.0 / step))
        volume = np.zeros((frameNum, 1))
        for i in range(frameNum):
            curFrame = frames[np.arange(i * step,
                                        min(i * step + frameSize,
                                            wlen)).astype('int64')]
            curFrame = curFrame - np.median(curFrame)  # zero-justified
            volume[i] = np.sum(np.abs(curFrame)**2) / frameSize  # 平方和
        return volume

    # # method 2: 10 times log10 of square sum
    # def calVolumeDB(waveData, frameSize, overLap):
    #     wlen = len(waveData)
    #     step = frameSize - overLap
    #     frameNum = int(math.ceil(wlen*1.0/step))
    #     volume = np.zeros((frameNum,1))
    #     for i in range(frameNum):
    #         curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
    #         curFrame = curFrame - np.mean(curFrame) # zero-justified
    #         volume[i] = 10*np.log10(np.sum(curFrame*curFrame))
    #     return volume

    @classmethod
    def cal_plain_envelope(cls, frames=None):
        # calculate volume
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!在这里修改帧长！！！！！！！！！！！！！！！！！！！！！！！！
        frameSize = cls.FRAMESIZE
        overLap = frameSize / 2
        volume11 = cls.calVolume(frames, frameSize, overLap)

        max_volume = np.max(volume11)
        min_volume = np.min(volume11)
        avg_volume = np.average(volume11)

        return avg_volume, max_volume, min_volume

        # volume12 = calVolumeDB(waveData,frameSize,overLap)

        # plot the wave
        # 计算时间轴的长度
        # time = np.arange(0, nframes)*(1.0/framerate)
        # time2 = np.arange(0, len(volume11))*(frameSize-overLap)*1.0/framerate
        # pl.subplot(311)
        # pl.plot(time, waveData)
        # pl.ylabel("Amplitude")
        # pl.plot(time2, volume11)
        # pl.ylabel("absSum")
        # pl.subplot(313)
        # pl.plot(time2, volume12, c="g")
        # pl.ylabel("Decibel(dB)")
        # pl.xlabel("time (seconds)")
        # pl.show()

    @classmethod
    def cal_speech_envelope(cls, frames=None):
        avg_volume, max_volume, min_volume = cls.cal_plain_envelope(frames)

        frameSize = cls.FRAMESIZE
        overLap = frameSize / 2
        wlen = len(frames)
        step = frameSize - overLap
        frameNum = int(math.ceil(wlen * 1.0 / step))
        speech = []
        nonspeech = []
        volume = np.zeros((frameNum, 1))
        for i in range(frameNum):
            curFrame = frames[np.arange(i * step,
                                        min(i * step + frameSize,
                                            wlen)).astype('int64')]
            curFrame = curFrame - np.median(curFrame)  # zero-justified
            curVolume = np.sum(np.abs(curFrame)**2)
            if curVolume > max_volume * 0.8:
                speech.append(curVolume)
            elif curVolume < max_volume * 0.2:
                nonspeech.append(curVolume)

        return np.average(speech), np.average(nonspeech)

    @classmethod
    def cal_relative_DB(cls, frames1, frames2):
        mag1 = cls.cal_plain_envelope(frames1)[0]
        mag2 = cls.cal_plain_envelope(frames2)[0]
        if mag1 < mag2:
            return 10 * np.log10(mag2 / mag1)
        else:
            return 10 * np.log10(mag1 / mag2)

    @classmethod
    def eval_stablity(cls, frames):
        sw = SlideWindow(frames, 480, 480)
        splited_frames = list(map(lambda x: np.expand_dims(x, 1).T, sw))
        splited_frames = np.concatenate(splited_frames, axis=0)
        print("Unstable factor: {}".format(np.sum(np.var(splited_frames, axis=0))))

    @classmethod
    def eval_relative_DB(cls, frames, speaker_power=0.00025):
        print("Relative DB: {}".format(10 * np.log10(
            Statistics.cal_plain_envelope(frames)[0] / speaker_power)))

    @classmethod
    def get_max_unsuccessive_fragment(cls,
                                      frames,
                                      n_clusters=2,
                                      noise_frames_count=480):
        sw = SlideWindow(frames, noise_frames_count, noise_frames_count)
        splited_frames = list(map(lambda x: np.expand_dims(x, 1).T, sw))
        splited_frames = np.concatenate(splited_frames, axis=0)
        # 选出信号周期性片段，剔除无声和过渡阶段片段
        clf = KMeans(n_clusters=n_clusters)
        clf.fit(splited_frames)
        labels = clf.labels_
        # 返回最长非连续片段
        return splited_frames[cls._find_max_unsuccessive_fragment_indexes(
            labels)]

    @classmethod
    def get_max_successive_fragment(cls,
                                    frames,
                                    n_clusters=2,
                                    noise_frames_count=480):
        sw = SlideWindow(frames, noise_frames_count, noise_frames_count)
        splited_frames = list(map(lambda x: np.expand_dims(x, 1).T, sw))
        splited_frames = np.concatenate(splited_frames, axis=0)
        # 选出信号周期性片段，剔除无声和过渡阶段片段
        clf = KMeans(n_clusters=n_clusters)
        clf.fit(splited_frames)
        labels = clf.labels_
        # 返回最长连续片段
        return splited_frames[cls._find_max_successive_fragment_indexes(
            labels)]

    @classmethod
    def _find_max_successive_fragment_indexes(cls, ary1d):
        if len(ary1d) == 0:
            return []
        max_reg = []
        successive_reg = [(0, ary1d[0])]
        for i, value in enumerate(ary1d[1:], start=1):
            if successive_reg[-1][-1] == value:
                successive_reg.append((i, value))
            else:
                if len(successive_reg) > len(max_reg):
                    max_reg = successive_reg.copy()
                successive_reg = [(i, value)]
        # Test
        for k, b in groupby(ary1d, key=lambda x: int(x)):
            print("key {}: {}".format(k, len(list(b))))
        return list(map(lambda x: x[0], max_reg))

    @classmethod
    def _find_max_unsuccessive_fragment_indexes(cls, ary1d):
        merge_dict = {}
        for i, value in enumerate(ary1d):
            if value in merge_dict.keys():
                merge_dict[value] += [i]
            else:
                merge_dict[value] = [i]
        # Test
        for k, b in groupby(ary1d, key=lambda x: int(x)):
            print("key {}: {}".format(k, len(list(b))))
        # for v in merge_dict.values():
        #     print(len(v))
        return merge_dict[max(merge_dict.keys(),
                              key=lambda k: len(merge_dict[k]))]