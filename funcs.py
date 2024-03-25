import matplotlib
import matplotlib.pyplot as pt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def add_graph(x_values, y_values, title, x_label, y_label, axes: Axes):
    pt.figure()  # .tight_layout()
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    # start, end = pt.axis()[0], pt.axis()[1]
    # pt.xticks(np.arange(start, end, 100))
    axes.plot(x_values, y_values, linewidth=0.7)
    axes.grid()


def save_graphs(pages: PdfPages):
    fig_nums = pt.get_fignums()
    figs = [pt.figure(n) for n in fig_nums]
    for fig in figs:
        pages.savefig(fig)
        print(fig.axes.__len__())


def save_graphs_on_same_page(pages: PdfPages, fig: Figure):
    pages.savefig(fig)
