# # from tests.test_pyaudio import run
# from tests.test_modulate import run
# # from tests.test_spectrum import run
# # from tests.test_locate import run
# # from tests.test_io import run
# # from tests.test_anc2 import run
# # from tests.test_anc3 import run
# # from tests.test_anc4 import run
# # from tests.test_fft import run
# from tests.test_no_linear import run
# # from tests.test_align import run
# # from tests.test_pool import run
# from tests.test_mutation import run
# from tests.test_ring_effect import run
# from tests.test_spec_sub import run
from tests.test_LMS import run

run()

# import numpy as np
# from utils.mplot import MPlot
# from utils.mfft import MFFT

# fs = 8000
# x = np.arange(0, fs // 2, 1)
# x = np.concatenate((np.zeros(fs // 2), x))
# y = x

# t = np.linspace(0, 1, fs)
# # a = np.sin(2 * np.pi * 50 * t + 2 * np.pi * 10 * t**2)
# a = np.sin(2 * np.pi * 50 * t)
# S = np.fft.fft(a)
# S = np.abs(S[:len(S) // 2])
# S = S / np.max(S)
# b = np.sin(2 * np.pi * 100 * t)
# R = np.fft.fft(b)
# R = np.abs(R[:len(R) // 2])
# R = R / np.max(R)
# R = np.concatenate((np.zeros(4000), R, np.zeros(3999)))

# tmp = []
# for i in range(fs // 2):
#     y = x - i
#     tmp.append(np.sum(np.convolve(S, y, mode="full") * R))
# print(np.argmax(tmp))

# # MPlot.subplot(
# #     [S,y, np.convolve(S, y, mode="full")])

