import numpy as np
import matplotlib.pyplot as plt


def run():
    t = np.linspace(0, 2, num=16000 * 2)
    a= np.sin(2 * np.pi * 100 * t)
    b = np.sin(2 * np.pi * 1000 * t)
    c = np.sin(2 * np.pi * 10 * t)

    plt.figure(1)

    d = a+b + c
    plt.subplot(3, 1, 1)
    plt.plot(d)
    # plt.show()

    e = np.fft.fft(d, 16000*2)
    plt.subplot(3, 1, 2)
    plt.plot(e)
    # plt.show()

    f = np.fft.ifft(e, 16000)
    plt.subplot(3, 1, 3)
    plt.plot(f)
    plt.show()


if __name__ == "__main__":
    run()