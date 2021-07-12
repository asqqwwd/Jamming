import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class MPlot():
    figure_count = 1

    @classmethod
    def plot_specgram(cls, data, fs, tp="db"):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        f, t, Zxx = signal.stft(data, fs=fs, nperseg=fs // 10)  # 窗口越小，频率泄露越严重
        if tp == "db":
            plt.pcolormesh(t,
                           f,
                           10 * np.log10(np.abs(Zxx)),
                           shading='gouraud',
                           cmap="binary")
        elif tp == "mag":
            plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap="binary")
        else:
            raise TypeError("Type error!")
        plt.colorbar()
        plt.ylim([f[1], f[-1]])
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def plot_specgram_plus(cls, data, fs):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        plt.specgram(data,
                     Fs=fs,
                     sides="onesided",
                     scale="dB",
                     mode="magnitude",
                     NFFT=fs // 10,
                     noverlap=fs // 20)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def subplot_specgram(cls, datas, fs):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        subplot_num = len(datas)
        counter = 1
        for data in datas:
            plt.subplot(subplot_num, 1, counter)
            counter += 1
            f, t, Zxx = signal.stft(data, fs=fs,
                                    nperseg=fs // 10)  # 窗口越小，频率泄露越严重
            plt.pcolormesh(t,
                           f,
                           10 * np.log10(np.abs(Zxx)),
                           shading='gouraud',
                           cmap="binary")
            plt.colorbar()
            plt.ylim([f[1], f[-1]])
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def subplot_specgram_plus(cls, datas, fs):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        subplot_num = len(datas)
        counter = 1
        for data in datas:
            plt.subplot(subplot_num, 1, counter)
            counter += 1
            plt.specgram(data,
                         Fs=fs,
                         sides="onesided",
                         scale="dB",
                         mode="magnitude",
                         NFFT=fs // 10,
                         noverlap=fs // 20)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def plot(cls, data):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        plt.plot(data)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def subplot(cls, datas):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        subplot_num = len(datas)
        counter = 1
        for data in datas:
            plt.subplot(subplot_num, 1, counter)
            counter += 1
            plt.plot(data)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def plot_together(cls, datas):
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        color_card = ["r", "g", "b", "y"] * 100
        linestyle_card = ["dashed", "solid"] * 100
        for data, cc, ls in zip(datas, color_card, linestyle_card):
            plt.plot(data, color=cc, linestyle=ls)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

    @classmethod
    def plot_complex(cls, x, y):
        '''
        x: x in real axis and be regularized
        y: y in imag axis and be regularized
        '''
        plt.figure(MPlot.figure_count)
        MPlot.figure_count += 1
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('axes', 0.5))

        circle = plt.Circle((0, 0), 1, color='b', fill=False)
        ax.add_artist(circle)

        for p1, p2 in zip(x, y):
            if p1**2 + p2**2 > 1e-1:
                plt.plot([0, p1], [0, p2], color="r")

        plt.scatter(x, y, s=20, color="r")
        plt.show()
