from sklearn.neural_network import MLPRegressor
import numpy as np
import time, os
import matplotlib.pyplot as plt
import scipy.signal as signal

from sklearn.metrics import mean_squared_error
from tests.test_anc2 import Anc
from joblib import dump, load
from tqdm import tqdm


def run():
    st = time.time()

    # 读取数据集
    anc = Anc()
    (x, y) = anc.load("input4.npy")
    print(x.shape, y.shape)
    # raise ValueError()
    ii = 1000
    x_train, y_train, x_test, y_test = x[:ii], y[:ii], x[ii:], y[ii:]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # raise ValueError()

    # 加载模型
    models = []
    filename_list = os.listdir("./models")
    filename_list.sort(key=lambda x: int(x[:-7]))
    for filename in filename_list:
        print(filename)
        models.append(load(os.path.join("./models", filename)))

    # # 训练模型
    # models = []
    # for y_train_w in tqdm(y_train.T):
    #     models.append(
    #         MLPRegressor(hidden_layer_sizes=(100,50),
    #                      activation='relu',
    #                      solver='adam',
    #                      alpha=0.01,
    #                      max_iter=400))
    #     models[-1].fit(x_train, y_train_w)

    # 保存模型
    if not os.listdir("./models"):
        for filename in os.listdir("./models"):
            os.remove(os.path.join("./models", filename))
    for i, model in enumerate(models):
        dump(model, "./models/{}.joblib".format(i))

    # 验证训练数据集
    predicts_w = []
    for model in models:
        predicts_w.append(model.predict(x_train))
    predicts = np.array(predicts_w).T
    print(predicts.shape)
    mse1 = mean_squared_error(predicts, y_train)
    print("Train ERROR:", mse1)

    # 验证测试数据集
    predicts_w = []
    for model in models:
        predicts_w.append(model.predict(x_test))
    predicts = np.array(predicts_w).T
    print(predicts.shape)
    mse2 = mean_squared_error(predicts, y_test)
    print("Test ERROR:", mse2)

    # Test
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(0, 100, predicts.shape[1])
    Y = np.linspace(0, 100, predicts.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, predicts, cmap='rainbow')

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(0, 100, predicts.shape[1])
    Y = np.linspace(0, 100, predicts.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, y_test, cmap='rainbow')
    plt.show()

    # plt.figure(2)
    # plt.specgram(predicts, Fs=16000, scale_by_freq=True,
    #                 sides='default')  # 绘制频谱

    print("Totally cost:", time.time() - st)