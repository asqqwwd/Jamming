from scipy import signal
# import resampy
import numpy as np


class Resampler():
    @classmethod
    def resample(cls, frames, org_fs, dst_fs):
        if org_fs == dst_fs:
            return frames
        else:
            return signal.resample(frames, (int(
                (frames.size * dst_fs) / org_fs)))
            # return np.zeros(int((frames.size * dst_fs) / org_fs))

    # @classmethod
    # def resample(cls, frames, org_fs, dst_fs):
    #     # print(type(frames),len(frames),org_fs,dst_fs)
    #     # input("***")
    #     return resampy.resample(frames, org_fs, dst_fs)
