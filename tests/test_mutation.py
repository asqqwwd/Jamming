from utils.access import Access
from utils.mplot import MPlot
import numpy as np


def run():
    fs = 48000
    lowest_f = 100
    t = np.linspace(0, 1 / lowest_f, int(1 / lowest_f * fs))

    # 1.Generate noise bases
    noise_bases = []
    
    # Phase
    # random_factors = [[0, np.pi / 4, 0, np.pi / 4],
    #                   [np.pi / 4, 0, np.pi / 4, 0]]
    # random_factors = []
    # for _ in range(2):
    #     tmp = []
    #     for _ in range(4):
    #         tmp.append(np.round(np.random.rand()*np.pi,3))
    #     random_factors.append(tmp)
    random_factors = [[0.126, 0.848, 0.238, 2.254],
                          [0.683, 0.36, 1.12, 0.052]]
    for rf in random_factors:
        tmp =  np.sin(2 * np.pi * lowest_f * t + rf[0]) +\
        np.sin(2 * np.pi * 2 * lowest_f * t + rf[1]) +\
        np.sin(2 * np.pi * 4 * lowest_f * t + rf[2]) +\
        np.sin(2 * np.pi * 8 * lowest_f * t + rf[3])
        tmp = tmp / np.max(np.abs(tmp))
        noise_bases.append(tmp)

    for nb in noise_bases:
        print(np.sum(np.power(nb,2)))

    # # Freq
    # random_factors = [[1, 0, 0, 0], [0, 0, 1, 0]]
    # for rf in random_factors:
    #     tmp =  rf[0] * np.sin(2 * np.pi * lowest_f * t) +\
    #         rf[1] * np.sin(2 * np.pi * 2 * lowest_f * t) +\
    #         rf[2] * np.sin(2 * np.pi * 4 * lowest_f * t) +\
    #         rf[3] * np.sin(2 * np.pi * 8 * lowest_f * t)
    #     tmp = tmp / np.max(np.abs(tmp))
    #     noise_bases.append(tmp)

    # # Mag
    # random_factors = [[0, 0, 0.5, 0], [0, 0, 1, 0]]
    # for rf in random_factors:
    #     tmp = rf[0] * np.sin(2 * np.pi * lowest_f * t) +\
    #         rf[1] * np.sin(2 * np.pi * 2 * lowest_f * t) +\
    #         rf[2] * np.sin(2 * np.pi * 4 * lowest_f * t) +\
    #         rf[3] * np.sin(2 * np.pi * 8 * lowest_f * t)
    #     noise_bases.append(tmp)

    # 2.Concatenate
    keys = []
    while len(keys) < 200:
        keys += [0] * 10 + [1] * 10
    out = np.array([])
    for i in range(200):
        out = np.concatenate((out, noise_bases[keys[i]]))

    MPlot.plot(out)


    # Access.save_wave(np.array(out), "./test.wav", 1, 2, fs)
