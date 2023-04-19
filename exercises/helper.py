from typing import List
from typing import Set
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netin import Graph


def update_name_homophily(data: Union[Graph, pd.DataFrame]) -> str:
    """
    Renames the object name to include the homophily values of the graph.
    :param data:
    :return:
    """

    # valid = [Graph, pd.DataFrame]
    # test = [c for c in valid if isinstance(data, c)]
    # if len(test) == 0:
    #     raise TypeError(f"The data must be one of these types: {valid}")

    if pd.DataFrame == type(data):
        current = data.name
        h_MM = data.h_MM.unique()[0]
        h_mm = data.h_mm.unique()[0]
    else:
        current = data.get_model_name()
        h_MM = data.get_homophily_majority()
        h_mm = data.get_homophily_minority()

    to_add = r"h$_{MM}$=<hMM>, h$_{mm}$=<hmm>".replace("<hMM>", f"{h_MM}").replace("<hmm>", f"{h_mm}")
    return f"{current}\n{to_add}"


def plot_edge_type_counts(data: Union[Graph, list[Graph], set[Graph]], fn=None, **kwargs):
    """
    Plots the edge type counts of a single or multiple graphs
    :param data: a single graph or a list of graphs
    :param kwargs: width_bar, figsize, loc, nc_legend
    :return:
    """

    if type(data) not in [list, set, List, Set]:
        data = [data]

    w, h = len(data) * 3.2, 3.2  # default figure size (width, height)
    width = kwargs.pop('width_bar', 0.25)  # the width of the bars
    figsize = kwargs.pop('figsize', (w, h))  # figure size (width, height)
    loc = kwargs.pop('loc', 'upper right')  # position of legend
    ncols = kwargs.pop('nc_legend', 1)  # number of columns in legend

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
    multiplier = 0

    x = None
    groups = None
    maxy = 0
    for g in data:
        etc = g.count_edges_types()
        name = g.get_model_name()

        groups = list(etc.keys()) if groups is None else groups
        x = np.arange(len(groups)) if x is None else x
        y = [etc[i] for i in groups]
        maxy = max(max(y), maxy)

        offset = width * multiplier
        rects = ax.bar(x + offset, y, width, label=name)
        ax.bar_label(rects, padding=3)
        multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Counts')
        ax.set_title("Counts of edge types");
        ax.set_xticks(x + width, groups)
        ax.legend(loc=loc, ncols=ncols)

    # set limits
    ax.set_ylim(0, maxy * 1.1)

    # save plot
    if fn is not None:
        validate_path(fn)
        fig.savefig(fname=fn, bbox_inches='tight', dpi=300)
        print(f'{fn} saved.')

    # Final
    plt.show()
    plt.close()


def validate_path(fn):
    import os
    path = os.path.dirname(fn)
    if path != '':
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
