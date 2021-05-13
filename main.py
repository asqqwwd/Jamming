import logging, sys, os

import settings
from threads.anc import ActiveNoiseControl
from threads.io import PyaudioIO
# from threads.io2 import SoundDeviceInput, SoundDeviceOutput
from threads.kws import KeywordSpotting
from threads.nl import NoiseLib


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
    kws_thread.stop()
    io_thread.stop()
    anc_thread.stop()
    nl_thread.stop()


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


if __name__ == "__main__":
    run()