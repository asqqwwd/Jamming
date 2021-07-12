import wave
import numpy as np
from utils.resampler import Resampler
from utils.modulate import Modulate
from utils.codec import Codec
from utils.access import Access


def generate_sin(length, fs, w):
    re = np.array([])
    t = np.linspace(0, length, num=fs * length)
    re = np.sin(2 * np.pi * w * t)
    return re


def run():
    # # offer
    # frames = Access.load_wave_with_fs("./waves/raw/offer.wav", 96000)

    # sin
    w = 1000
    length = 1
    frames = generate_sin(length, 96000, w)

    # modulate
    save_frames = Modulate.am_modulate(frames, 2, 96000)

    # Access.save_txt(save_frames[0], "F://offer_primary.csv")
    # Access.save_txt(save_frames[1], "F://offer_secondary.csv")
    Access.save_csv(save_frames[0], "F://1khz_primary.csv")
    # Access.save_csv(save_frames[1], "F://100hz_secondary.csv")
    # Access.save_csv(save_frames,"F://100hz_1channel")
