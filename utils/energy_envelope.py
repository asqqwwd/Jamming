import math
import numpy as np
import wave
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

FRAMESIZE = 256

# method 1: absSum
def calVolume(frames, frameSize, overLap):
    wlen = len(frames)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen*1.0/step))
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = frames[np.arange(i*step,min(i*step+frameSize,wlen)).astype('int64')]
        curFrame = curFrame - np.median(curFrame) # zero-justified
        volume[i] = np.sum(np.abs(curFrame)**2)/frameSize
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

def cal_plain_envelope(frames=None):
    # calculate volume
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!在这里修改帧长！！！！！！！！！！！！！！！！！！！！！！！！
    print("帧长为frameSize")
    frameSize = FRAMESIZE
    overLap = frameSize/2
    volume11 = calVolume(frames,frameSize,overLap)

    max_volume = np.max(volume11)
    min_volume = np.min(volume11)
    avg_volume = np.average(volume11)

    return avg_volume, max_volume, min_volume, frames

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

def cal_speech_envelope(frames=None):
    avg_volume, max_volume, min_volume, waveData = cal_plain_envelope(frames)

    frameSize = FRAMESIZE
    overLap = frameSize/2
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen*1.0/step))
    speech = []
    nonspeech = []
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen)).astype('int64')]
        curFrame = curFrame - np.median(curFrame) # zero-justified
        curVolume = np.sum(np.abs(curFrame)**2)
        if curVolume > max_volume*0.8:
            speech.append(curVolume)
        elif curVolume < max_volume*0.2:
            nonspeech.append(curVolume)

    return np.average(speech), np.average(nonspeech)

    

def main():
    a = np.load("../Mi_Base100+Speaker.npy")
    plt.figure(1)
    plt.plot(a)
    #用于计算背景噪声和mask声音的使用smoothWav，其中的avg_volume用于后续计算
    avg_volume, max_volume, min_volume, waveData = cal_plain_envelope(a)
    print(np.sum(np.power(a,2))/len(a),avg_volume)
    #用于计算说话声的使用smoothWav，返回两个值分别为speech和nonespeech用于后续计算
    # speechVolume, nonespeechVolume = cal_speech_envelope(a)
    #两个函数都可以直接传入wav数据，只需要参数中表明即可，文件传入主要用于测试，实例如下：
    # speechWav(wavdata=OutData)
    print("over")
    plt.show()


main()