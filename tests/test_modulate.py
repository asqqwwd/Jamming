import wave
import numpy as np
from utils.resampler import Resampler
from utils.modulate import Modulate
from utils.codec import Codec
from utils.access import Access


def generate_sin(size, fs, w):
    re = np.array([])
    t = np.linspace(0, size / fs, num=size)
    re = np.sin(2 * np.pi * w * t)
    return re

def run():
    # offer
    frames = Access.load_wave_with_fs("./waves/raw/offer.wav", 96000)

    # sin
    # w = 3e3
    # length = 4
    # frames = generate_sin(length * 96000, 96000, w)

    # modulate
    save_frames = Modulate.am_modulate(frames, 2, 96000)

    # Access.save_txt(save_frames[0], "./waves/modulated/offer_primary.txt")
    # Access.save_txt(save_frames[1], "./waves/modulated/offer_secondary.txt")
    Access.save_csv(save_frames[0], "./waves/modulated/offer_primary.csv")
