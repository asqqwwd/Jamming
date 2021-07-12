import scipy.signal as signal
import numpy as np


class MFFT():
    @classmethod
    def stft(cls, frames, fs):
        """
        Return: (n_frame,n_fft)
        """
        # nperseg窗口大小等于nfft，若nperseg>nfft，则会自动填0，但这会导致无法还原
        # 窗口小，提升时间分辨率，降低频率分辨率
        return signal.stft(frames, fs=fs, nperseg=fs // 10)

    @classmethod
    def istft(cls, spectrogram, fs):
        return signal.istft(spectrogram, fs=fs)[-1]

    @classmethod
    def divide_magphase(cls, D):
        """Separate a complex-valued stft D into its magnitude (S)
        and phase (P) components, so that `D = S * P`."""
        S = np.abs(D)
        P = np.exp(1.j * np.angle(D))
        return S, P

    @classmethod
    def merge_magphase(cls, S, P):
        """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
        return S * P
