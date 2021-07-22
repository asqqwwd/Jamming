from numpy.fft import ifft
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

    @classmethod
    def convolve(cls, a, h):
        final_length = len(a) + len(h) - 1
        A = np.fft.fft(cls.align_length(a, final_length, tp=1))
        H = np.fft.fft(cls.align_length(h, final_length, tp=1))
        return np.fft.ifft(A * H).real

    @classmethod
    def deconvolve(cls, y, h):
        if len(h) > len(y):
            raise ValueError("Length of y must be greater than h")
        h_padding = cls.align_length(h, len(y), tp=1)
        Y = np.fft.fft(y)
        H = np.fft.fft(h_padding)
        a = Y.real
        b = Y.imag
        c = H.real
        d = H.imag
        denominator = np.power(c, 2) + np.power(d, 2)
        denominator[denominator < 1e-8] = 1e-8
        if len(denominator[denominator < 1e-8]) > 0:
            print("H zero complex warning: " +
                  str(len(denominator[denominator < 1e-8])))
        return np.fft.ifft((a * c + b * d) / denominator +
                           (b * c - a * d) / denominator * 1.j).real[:len(y) -
                                                                     len(h) +
                                                                     1]

    @classmethod
    def wiener_deconvolve(cls, y, h, x, n):
        if len(h) > len(y):
            raise ValueError("Length of y must be greater than h")
        Y = np.fft.fft(y)
        H = np.fft.fft(cls.align_length(h, len(y), tp=1))
        SNR = np.mean(np.abs(2 * np.fft.fft(x) / len(x))**2) / np.mean(
            np.abs(2 * np.fft.fft(n) / len(n))**2)
        SNR = 10 * np.log10(SNR)  # 不进行幅度调整将无效果
        H[H==0] = 1+1j
        return np.fft.ifft(
            Y / (H * (1 + (1 / (SNR * np.abs(H)**2))))).real[:len(y) - len(h) +
                                                             1]

    # @classmethod
    # def wiener_deconvolve2(cls, y, h, x, n):
    #     if len(h) > len(y):
    #         raise ValueError("Length of y must be greater than h")
    #     Y = np.fft.fft(y)
    #     H = np.fft.fft(cls.align_length(h, len(y), tp=1))
    #     S = np.mean(np.abs(2 * np.fft.fft(x) / len(x))**2)
    #     N = np.mean(np.abs(2 * np.fft.fft(n) / len(n))**2)
    #     H_conjugate_transpose = H.real-H.imag*1j
    #     G = H_conjugate_transpose*S/(np.abs(H)**2*S+N)
    #     return np.fft.ifft(Y*G).real[:len(y)-len(h)+1]

    @classmethod
    def align_length(cls, frames, base_length, tp=0):
        '''
        tp=0: left padding zeros
        tp=1: right padding zeros
        '''
        aligned_frames = None
        if len(frames) < base_length and tp == 0:
            aligned_frames = np.concatenate(
                (np.zeros(base_length - len(frames)), frames))
        if len(frames) < base_length and tp == 1:
            aligned_frames = np.concatenate(
                (frames, np.zeros(base_length - len(frames))))
        else:
            aligned_frames = frames[:base_length]
        return aligned_frames

    @classmethod
    def uniform_align_length(cls, ary1d, base_length):
        if base_length < len(ary1d):
            raise ValueError("Base length must greater than input length")
        diff_length = base_length - len(ary1d)
        if diff_length % 2 == 0:
            return np.concatenate((np.zeros(diff_length // 2), ary1d,
                                   np.zeros(diff_length // 2)))
        else:
            return np.concatenate((np.zeros(diff_length // 2), ary1d,
                                   np.zeros(diff_length // 2 + 1)))

    @classmethod
    def repeat_align_length(cls, frames, base_length):
        re = np.tile(frames, base_length // len(frames) + 1)
        re = cls.align_length(re, base_length)
        return re
