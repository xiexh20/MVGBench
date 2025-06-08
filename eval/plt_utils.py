from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib
matplotlib.use('Agg') # without using server


class PltVisualizer:
    "helper class"
    def __init__(self):
        pass

    @staticmethod
    def plot_and_save(x, y, xlabel, ylable, title, outfile,
                      figsize=(10, 10), legend=None, ylim=None,mute=False,
                      xlim=None):
        """

        Args:
            x:
            y:(N, D), with D different curves, each with length N
            xlabel:
            ylable:
            title:
            outfile:
            figsize:
            legend:
            ylim:

        Returns:

        """
        fig = plt.figure(figsize=figsize)
        # if legend is not None:
        #     for i, label in enumerate(legend):
        #         plt.plot(x, y[:, i], label=label)
        # else:
        # plt.plot(x, y, 'o', color='black') # plot dot
        plt.plot(x, y)
        if legend is not None:
            plt.legend(legend, fontsize=24)
        plt.xlabel(xlabel, fontsize=40)
        plt.ylabel(ylable, fontsize=40)
        if title is not None:
            plt.title(title)
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)

        # ax = plt.gca() # get current axis
        # handles, labels = ax.get_legend_handles_labels()
        # # sort both labels and handles by labels
        # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        # ax.legend(handles, labels)

        plt.tick_params(axis='x', labelsize=20)  # Set x-axis tick font size to 12
        plt.tick_params(axis='y', labelsize=20)  # Set y-axis tick font size to 10

        fig.savefig(outfile, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        if not mute:
            print("Plot saved to {}".format(outfile))
        plt.close(fig)

    @staticmethod
    def plot_and_save_robustness(x, y, xlabel, ylable, title, outfile,
                      figsize=(10, 10), legend=None, ylim=None, mute=False,
                      xlim=None, arrow='\u2191'):
        """
        configs tuned for robustness figure
        Args:
            x:
            y:(N, D), with D different curves, each with length N
            xlabel:
            ylable:
            title:
            outfile:
            figsize:
            legend:
            ylim:

        Returns:

        """
        fig = plt.figure(figsize=figsize)
        # if legend is not None:
        #     for i, label in enumerate(legend):
        #         plt.plot(x, y[:, i], label=label)
        # else:
        # plt.plot(x, y, 'o', color='black') # plot dot
        # plt.plot(x, y, linewidth=2.5, linestyle='-', marker='s', markersize=12, )
        plt.plot(x, y, linewidth=2.5, linestyle='-', marker='s', markersize=8, )
        # plt.plot(x, y, linewidth=2.5, linestyle='-', marker='^', markersize=16, )
        if legend is not None:
            plt.legend(legend, fontsize=22)
        plt.xlabel(xlabel, fontsize=32)
        plt.ylabel(ylable+arrow, fontsize=32)
        plt.title("    ")
        # if title is not None:
        #     plt.title(title)
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)

        # ax = plt.gca() # get current axis
        # handles, labels = ax.get_legend_handles_labels()
        # # sort both labels and handles by labels
        # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        # ax.legend(handles, labels)

        plt.tick_params(axis='x', labelsize=18)  # Set x-axis tick font size to 12
        plt.tick_params(axis='y', labelsize=18)  # Set y-axis tick font size to 10
        # Get the current axis and update all spines
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.6)  # set border thickness
        # Optionally, increase the tick line thickness
        plt.tick_params(width=1.5, length=8)

        fig.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
        if not mute:
            print("Plot saved to {}".format(outfile))
        plt.close(fig)

    @staticmethod
    def plot_hist2d(x, y, title, xlabel, ylabel, outfile, figsize=(10, 8), bins=100,
                    cmap="RdYlGn_r",
                    norm=colors.LogNorm(),
                    # norm=colors.PowerNorm(1.0),
                    ylim=None):
        """
        plot 2d histogram
        Args:
            x: (N, ), np array
            y: (N, ), np array
            title: figure title
            xlabel: xlabel
            ylabel: ylabel
            figsize:
            color is normalized by log

        Returns:

        """
        fig = plt.figure(figsize=figsize)
        plt.hist2d(x, y, bins=bins, cmap=cmap, norm=norm)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(ylim)
        plt.colorbar()
        plt.savefig(outfile)
        print("Plot saved to {}".format(outfile))
        plt.close(fig)

    @staticmethod
    def plot_hist(x, title, xlabel, ylabel, outfile, figsize=(10, 8), bins=100):
        """
        plot 1d histogram
        Args:
            x:
            title:
            xlabel:
            ylabel:
            outfile:
            figsize:

        Returns:

        """
        counts, bins = np.histogram(x)
        fig = plt.figure(figsize=figsize)
        plt.hist(x, density=False, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(outfile)
        print("Plot saved to {}".format(outfile))
        plt.close(fig)


if __name__ == '__main__':
    import numpy as np
    from numpy.random import multivariate_normal

    # test plot_hist2d
    result = np.vstack([
        multivariate_normal([10, 10],
                            [[3, 2], [2, 3]], size=1000000),
        multivariate_normal([30, 20],
                            [[2, 3], [1, 3]], size=100000)
    ])

    PltVisualizer.plot_hist2d(result[:, 0],
           result[:, 1], 'test', "xlabel", 'ylabel', 'debug/histogram/test.png',
                              ylim=(0, 30))
