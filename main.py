import logging, sys, os

import settings
from threads.anc import ActiveNoiseControl
from threads.io import PyaudioIO
# from threads.io2 import SoundDeviceInput, SoundDeviceOutput
from threads.kws import KeywordSpotting
from threads.nl import NoiseLib
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np


def run():
    # 启动程序
    _config_logging()
    logging.info("Start jamming programmer")

    params1 = {
        "chirp_length": settings.CHIRP_LENGTH,
        "out_fs": settings.OUT_FS,
        "noise_length": settings.NOISE_LENGTH,
    }
    nl_thread = NoiseLib(**params1)

    params2 = {
        "nosie_lib": nl_thread,
        "in_fs": settings.IN_FS,
        "in_channel": settings.IN_CHANNEL,
        "in_bit_depth": settings.IN_BIT_DEPTH,
        "chirp_length": settings.CHIRP_LENGTH,
        "nosie_length": settings.NOISE_LENGTH,
        "snr_calculate_block_count": settings.SNR_CALCULATE_BLOCK_COUNT,
        "simulation_block_count": settings.SIMULATION_BLOCK_COUNT
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
    # io_thread = PyaudioIO(**params3)

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


if __name__ == "__main__":
    run()