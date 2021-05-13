import wave
import numpy as np
from utils.resampler import Resampler
from utils.modulate import Modulate
from utils.codec import Codec

def generate_sin(size,fs,w):
    re = np.array([])
    t = np.linspace(0, size / fs, num=size)
    re = np.sin(2 * np.pi * w * t)
    return re

# def run():
#     w = 10e3
#     sin_frames = generate_sin(10*96000,96000,w)
#     save_frames = Modulate.pm_modulate(sin_frames, 2, 96000)

#     with wave.open("./waves/modulated/sin_{}k_pm.wav".format(int(w/1e3)), "wb") as wf:
#         wf.setnchannels(2)
#         wf.setsampwidth(2)
#         wf.setframerate(96000)
#         wf.writeframes(
#             Codec.encode_audio_to_bytes(save_frames, 2, 16))
def run():
    with wave.open("./waves/raw/offer.wav", "rb") as rf:
        print(rf.getparams())
        nchannels = rf.getparams().nchannels
        sampwidth = rf.getparams().sampwidth
        framerate = rf.getparams().framerate
        nframes = rf.getparams().nframes
        frames = Codec.decode_bytes_to_audio(rf.readframes(nframes), nchannels,
                                             sampwidth * 8)
        frames = Resampler.resample(frames, framerate, 96000)

    save_frames = Modulate.am_modulate(frames, 1, 96000)
    # save_frames = (save_frames,save_frames)

    with wave.open("./waves/modulated/offer_40k_c1.wav", "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(sampwidth)
        wf.setframerate(96000)
        wf.writeframes(
            Codec.encode_audio_to_bytes(save_frames, 1, sampwidth * 8))
