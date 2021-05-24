import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt


def _divide_magphase(D, power=1):
    """Separate a complex-valued stft D into its magnitude (S)
    and phase (P) components, so that `D = S * P`."""
    S = np.abs(D)
    S **= power
    P = np.exp(1.j * np.angle(D))
    return S, P


def _merge_magphase(S, P):
    """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
    return S * P


# if __name__ == "__main__":
#     t = np.linspace(0, 1, 16000)
#     a = np.sin(2 * np.pi * 1000 * t)
#     b_S, b_P = _divide_magphase(
#         signal.stft(a, fs=16000, nperseg=512, return_onesided=False)[-1])
#     # c_S, c_P = _divide_magphase(librosa.stft(a, n_fft=512, hop_length=128).T)
#     # print(b_S.shape, c_S.shape)
#     b = 2 * b_S[:257]
#     # c = c_S

#     fig = plt.figure(1)
#     ax = fig.add_subplot(111, projection='3d')
#     X = np.linspace(0, 8000, b.shape[1])
#     Y = np.linspace(0, 1, b.shape[0])
#     X, Y = np.meshgrid(X, Y)
#     ax.plot_surface(X, Y, b, cmap='rainbow')

#     # fig = plt.figure(2)
#     # ax = fig.add_subplot(111, projection='3d')
#     # X = np.linspace(0, 8000, c.shape[1])
#     # Y = np.linspace(0, 1, c.shape[0])
#     # X, Y = np.meshgrid(X, Y)
#     # ax.plot_surface(X, Y, c, cmap='rainbow')

#     plt.figure(3)
#     # plt.subplot(2, 1, 1)
#     plt.plot(signal.istft(_merge_magphase(b_S, b_P), fs=16000, nperseg=512))

#     plt.show()

if __name__ == "__main__":
    amp = 2 * np.sqrt(2)
    tt = np.linspace(0, 3, 16000)
    a = np.sin(2 * np.pi * 1000 * tt)
    f, t, Zxx = signal.stft(a, fs=16000, nperseg=512)
    print(f)
    plt.figure(1)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.ylim([f[1], f[-1]])

    _, xrec = signal.istft(Zxx, 16000,nperseg=512)
    plt.figure(2)
    plt.plot(xrec)
    plt.show()
