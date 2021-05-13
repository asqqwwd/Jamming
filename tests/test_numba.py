from numba import int32, float32, boolean,jit  # import the types
from numba.experimental import jitclass

import threading, wave, math, time
import numpy as np

import global_var, settings
from utils.codec import Codec
from utils.resampler import Resampler

# spec = [
#     # ('daemon', boolean),               # a simple scalar field
#     ('exit_flag', boolean),          # an array field
#     ('out_fs', int32),          # an array field
#     ('noise_length', float32),
#     ('chirp_length', float32),
#     ('f_lower_bound', int32),
#     ('f_upper_bound', int32),
#     ('num_of_base', int32),
#     ('f1', int32),
#     ('f2', int32),
#     ('up_chirp_frames', float32[:]),
#     ('down_chirp_frames', float32[:]),
#     ('noise_frames', float32[:]),
# ]

spec1 = [
    ('value', int32),  # a simple scalar field
    ('array', float32[:]),  # an array field
]


@jitclass(spec1)
class Test(object):
    def __init__(self, value):
        self.value = value
        # self.array = np.linspace(0, 1, 10**6)
        self.array = np.zeros(value, dtype=np.float32)

    def get(self):
        print("**")
        for s in self.array:
            kk = s
            yield s

# @jit(nopython=True)
def AA():
    print("**")
    array = np.zeros(10**8, dtype=np.float32)
    for s in array:
        kk = s
        yield s


def run():
    # nl_thread = NoiseLib(settings.OUT_FS, settings.NOISE_LENGTH,
    #                      settings.CHIRP_LENGTH)
    # input("")
    # nl_thread.stop()

    t1 = Test(10**9)

    # print(t1.get())
    st = time.time()
    for s in AA():
        pass
    print("Cost: {}".format(time.time()-st))
