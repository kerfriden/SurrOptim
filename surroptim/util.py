import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Minimal R2 implementation: 1 - SSE/SST.

    Kept intentionally simple: returns 1 - sum((y-y_pred)^2)/sum((y-mean(y))^2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1.0 - ss_res / ss_tot

# Backwards compatibility alias
r2_score_simple = r2_score


def prediction_plot(X=None, y=None, function=None, x_plot=None, y_plot=None, show=True, marker_color=True, clabel=None, xlabel=None, ylabel=None, xlim=None, ylim=None):

    if X is not None:
        dim = X.shape[1]
    elif y_plot is None and x_plot is not None:
        dim = 1
    elif y_plot is not None and x_plot is not None:
        dim = 2
    else:
        'cannot plot if data is not 1D or 2D'

    if dim > 2:
        'cannot plot if data is not 1D or 2D'

    if x_plot is None:
        x_min = np.min(X[:, 0])
        x_max = np.max(X[:, 0])
        x_plot = np.linspace(x_min, x_max, 40)
    if dim == 2 and (y_plot is None):
        y_min = np.min(X[:, 1])
        y_max = np.max(X[:, 1])
        y_plot = np.linspace(y_min, y_max, 40)

    if dim == 2:

        X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

        if function is not None:
            X_grid = np.concatenate((X_plot.reshape((-1, 1)), Y_plot.reshape((-1, 1))), axis=1)
            Z = function(X_grid)
            Z = Z.reshape((len(y_plot), len(x_plot)))
            plt.contourf(X_plot, Y_plot, Z, 20, alpha=0.2)
            if clabel is not None:
                plt.colorbar(label=clabel)
            else:
                plt.colorbar()
            plt.contour(X_plot, Y_plot, Z, 20, colors='black')

        if X is not None:
            if (y is not None) and marker_color == True:
                plt.scatter(X[:, 0], X[:, 1], c=y)
                if clabel is not None:
                    plt.colorbar(label=clabel)
                else:
                    plt.colorbar()
            else:
                plt.scatter(X[:, 0], X[:, 1], c='k')

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

    elif dim == 1:

        if function is not None:
            X_plot = x_plot.reshape((-1, 1))
            Z = function(X_plot)

            plt.plot(x_plot, Z)

        if (y is not None):
            plt.scatter(X, y, c='k')

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None and dim == 2:
        plt.ylim(ylim)

    plt.grid()
    if show == True:
        plt.show()


def surf_plot(function=None, x_plot=None, y_plot=None, show=True, xlim=None, ylim=None, xlabel=None, ylabel=None, zlabel=None, X=None, Y=None):

    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    X_grid = np.concatenate((X_plot.reshape((-1, 1)), Y_plot.reshape((-1, 1))), axis=1)
    Z = function(X_grid)
    Z = Z.reshape((len(y_plot), len(x_plot)))

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_plot, Y_plot, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)

    if X is not None:
        ax.scatter(X[:, 0], X[:, 1], Y.squeeze())

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if show == True:
        plt.show()
