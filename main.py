import logging, sys, os
from utils.mfft import MFFT

import settings
from threads.anc import ActiveNoiseControl
from threads.io import PyaudioIO
# from threads.io2 import SoundDeviceInput, SoundDeviceOutput
from threads.kws import KeywordSpotting
from threads.nl import NoiseLib
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from utils.access import Access
from utils.iters import SlideWindow
from utils.mplot import MPlot


def run():
    # fs = 48000
    # a = Access.load_data("./test1_chirp_am_phase.npy") * 10
    # b = Access.load_data("./test1_chirp_am_freq.npy") * 10
    # c = Access.load_data("./test1_chirp_am_mag.npy") * 10
    # MPlot.subplot([a, b, c])
    # raise ValueError("**")

    # a = np.random.normal(0, 1, 48000)
    # MPlot.plot_specgram(a, 48000)
    # Access.save_wave(np.tile(a, 20), "./WhiteNoise.wav", 1, 2, 48000)
    # lowpass_filter = signal.butter(8, 0.2, 'lowpass')  # 7.2k/24k=0.3 5/24k=0.2
    # b = signal.filtfilt(lowpass_filter[0], lowpass_filter[1], a)
    # MPlot.plot_specgram(b, 48000)
    # Access.save_wave(np.tile(b, 20), "./WhiteNoise_5KFilter.wav", 1, 2, 48000)
    # raise ValueError("**")

    # base = generate_noise_bases(1 / 100, 48000)
    # MPlot.plot(base)
    # # MPlot.plot(np.tile(base,2))
    # Access.save_wave(np.tile(base, 2000), "./PB_Sin1k.wav", 1, 2, 48000)
    # # # h = [0] * 40 + [1] + [0.8] * 10 + [0.5] * 10 + [0.3] * 10 + [0.1] * 10
    # # # b = np.convolve(a, h, mode="same")
    # # # MPlot.subplot([a, b])
    # # # MPlot.plot_specgram(a, 48000)
    # # # MPlot.plot_specgram(b, 48000)

    # raise ValueError("**")

    # 启动程序
    _config_logging()
    logging.info("Start jamming programmer")

    params1 = {
        "chirp_length": settings.CHIRP_LENGTH,
        "out_fs": settings.OUT_FS,
        "lowest_f": settings.LOWEST_F,
        "noise_num": settings.NOISE_NUM,
    }
    nl_thread = NoiseLib(**params1)

    params2 = {
        "nosie_lib": nl_thread,
        "in_fs": settings.IN_FS,
        "chirp_length": settings.CHIRP_LENGTH,
        "lowest_f": settings.LOWEST_F,
        "noise_num": settings.NOISE_NUM,
    }
    anc_thread = ActiveNoiseControl(**params2)

    params3 = {
        "in_fs": settings.IN_FS,
        "in_channel": settings.IN_CHANNEL,
        "in_bit_depth": settings.IN_BIT_DEPTH,
        "in_frames_per_buffer": settings.IN_FRAMES_PER_BUFFER,
        "in_device_keyword": settings.IN_DEVICE_KEYWORD,
        "in_host_api": settings.IN_HOST_API,
        "out_fs": settings.OUT_FS,
        "out_channel": settings.OUT_CHANNEL,
        "out_bit_depth": settings.OUT_BIT_DEPTH,
        "out_frames_per_buffer": settings.OUT_FRAMES_PER_BUFFER,
        "out_device_keyword": settings.OUT_DEVICE_KEYWORD,
        "out_host_api": settings.OUT_HOST_API
    }
    io_thread = PyaudioIO(**params3)

    params4 = {
        "in_fs": settings.IN_FS,
        "out_fs": settings.OUT_FS,
        "mute_period_length": settings.MUTE_PERIOD_LENGTH,
        "kws_frame_length": settings.KWS_FRAME_LENGTH
    }
    # kws_thread = KeywordSpotting(**params4)

    input("")
    logging.info("Stop jamming programmer")
    # kws_thread.stop()
    # io_thread.stop()
    # anc_thread.stop()
    # nl_thread.stop()


def _config_logging():
    if not os.path.exists("logs"):
        os.mkdir("logs")

    # log_filename = datetime.datetime.now().strftime("%Y-%m-%d-%H%M") + ".log"
    log_filename = "test.log"
    log_filepath = os.path.join(os.path.join(os.getcwd(), "logs"),
                                log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    fh = logging.FileHandler(filename=log_filepath, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)  # 日志输出到终端
    logger.addHandler(fh)  # 日志输出到文件
    logging.getLogger('matplotlib.font_manager').disabled = True  # 禁用字体管理记录器

    logging.info("Current log file {}".format(log_filepath))


def _compress_dim(fs, base_f, n, spec_n):
    """
    fs: sample rate
    base_f: base of frequency
    n: target dims after compress
    spec_n: spectrum at a certain moment
    """
    f_list = np.linspace(0, fs // 2, len(spec_n))
    new_spec_n = spec_n
    new_spec_n[new_spec_n < 1e-4] = 0
    re = []
    for freq in np.arange(base_f, n * base_f + base_f, base_f):
        index_list = []
        for i, f in enumerate(f_list):
            if abs(f - freq) <= 10:
                index_list.append(i)
        re.append(np.sum(spec_n[index_list]))
    re = np.array(re)
    return re


def _decompress_dim(fs, base_f, n, factors):
    """
    fs: sample rate
    base_f: base of frequency
    n: target dims after decompress
    factors: polynomial's factors
    """
    f_list = np.linspace(0, fs // 2, n)
    re = np.zeros(n)
    for freq, factor in zip(
            np.arange(base_f,
                      len(factors) * base_f + base_f, base_f), factors):
        index_list = []
        for i, f in enumerate(f_list):
            if abs(f - freq) <= 5:
                index_list.append(i)
        print(index_list)
        re[index_list[len(index_list) // 2]] = factor
    return re


def align_length(frames, base_length, tp=0):
    '''
    tp=0: left padding zeros
    tp=1: right padding zeros
    '''
    aligned_frames = None
    if len(frames) < base_length and tp == 0:
        aligned_frames = np.concatenate(
            (np.zeros(base_length - len(frames)), frames))
    if len(frames) < base_length and tp == 1:
        aligned_frames = np.concatenate(
            (frames, np.zeros(base_length - len(frames))))
    else:
        aligned_frames = frames[:base_length]
    return aligned_frames


# 生成噪声基底
def generate_noise_bases(length, fs):
    t = np.linspace(0, length, int(fs * length) + 1)
    re = np.zeros(len(t))

    w_bases = np.arange(1000, 1100, 100)
    rfs = 2 * np.random.randint(0, 2, len(w_bases)) - 1
    for w, rf in zip(w_bases, rfs):
        re += rf * np.sin(2 * np.pi * w * t)
    re = re / np.max(np.abs(re))
    return re[:-1]


if __name__ == "__main__":
    run()