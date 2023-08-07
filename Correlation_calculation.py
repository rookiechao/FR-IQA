import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_PLCC(y_pred, y_val):
    return stats.pearsonr(y_pred, y_val)[0]


def get_SROCC(y_pred, y_val):
    return stats.spearmanr(y_pred, y_val)[0]


def get_KROCC(y_pred, y_val):
    return stats.stats.kendalltau(y_pred, y_val)[0]


def get_RMSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.sqrt(np.mean((y_p - y_v) ** 2))


def get_MSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.mean((y_p - y_v) ** 2)


def mos_scatter(pred, mos, show_fig=False):
    fig = plt.figure()
    plt.scatter(mos, pred, s=5, c='g', alpha=0.5)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    if show_fig:
        plt.show()
    return fig