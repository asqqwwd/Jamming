import numpy as np
import scipy.signal as signal


class MFilter():
    @classmethod
    def ideal_filter(cls, m, fs, cut_off_freq=5e3):
        # 理想滤波
        m_sp = np.fft.fft(m)
        S, P = np.abs(m_sp), np.exp(1.j * np.angle(m_sp))
        S[int(cut_off_freq / fs * len(m)):int((fs - cut_off_freq) / fs *
                                              len(m))] = 0
        return np.fft.ifft(S * P).real

    @classmethod
    def non_ideal_filter(cls, m, fs, cut_off_freq=5e3):
        # 非理想滤波滤波
        cut_off_freq = 5e3
        lowpass_filter = signal.butter(8, 2 * cut_off_freq / fs, 'lowpass')
        return signal.filtfilt(lowpass_filter[0], lowpass_filter[1], m)
