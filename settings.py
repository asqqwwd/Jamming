import sys
if "win" in sys.platform:
    IS_ON_RASPI = False  # 运行在windows上
else:
    IS_ON_RASPI = True  # 运行在raspi上

if IS_ON_RASPI:
    import os, sys
    # os.close(sys.stderr.fileno())  # 关闭错误std流

CHIRP_LENGTH = 0.1
KWS_FRAME_LENGTH = 0.2  # 关键词识别长度
MUTE_PERIOD_LENGTH = 5  # 停止干扰时长
LOWEST_F = 100  # 最低频率
NOISE_NUM = 200  # 噪声组合个数

IN_FS = 48000  # 麦克风采样率
IN_CHANNEL = 1  # 麦克风输入通道数，这里先不要改
IN_BIT_DEPTH = 16  # 麦克风采样位深
IN_DEVICE_KEYWORD = "snd_rpi_i2s_card" if IS_ON_RASPI else "WO Mic"  # 声卡名称，None表示使用默认输入设备
IN_HOST_API = 0 if IS_ON_RASPI else 1  # 输入设备的host api
IN_FRAMES_PER_BUFFER = 2048 * 8  # 每次输入声音长度

OUT_FS = 96000  # 扬声器采样率
OUT_CHANNEL = 1  # 扬声器输出通道数
OUT_BIT_DEPTH = 16  # 扬声器采样位深
OUT_DEVICE_KEYWORD = "USB Audio" if IS_ON_RASPI else "USB-C"  # 声卡名称
OUT_HOST_API = 0 if IS_ON_RASPI else 1  # 输出设备的host api
OUT_FRAMES_PER_BUFFER = int((CHIRP_LENGTH +
                         1 / LOWEST_F * NOISE_NUM) * OUT_FS)  # 每次输出声音长度
