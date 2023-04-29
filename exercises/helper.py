import os
from collections import Counter
from typing import List, Tuple
from typing import Set
from typing import Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import netin
import networkx as nx
import numpy as np
import pandas as pd
from netin import Graph
from netin import UnDiGraph
from netin import viz
from netin.stats import networks as net
from netin.utils import constants as const


def update_name_homophily(data: Union[Graph, pd.DataFrame]) -> str:
    """
    Renames the object name to include the homophily values of the graph.

    Parameters
    ----------
    data: Graph or pd.DataFrame
        The graph or dataframe containing the name of the graph (model or dataset) and the homophily values.

    Returns
    -------
    str
        The new name
    """
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

    Parameters
    ----------
    data: netin.Graph or List[netin.Graph] or Set[netin.Graph]
        a single graph or a list of graphs

    kwargs: dict
        width_bar, figsize, loc, nc_legend
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
        etc = g.calculate_edge_type_counts()
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
    """
    If the path does not exist, it creates it.
    :param fn:  file name
    """
    import os
    path = os.path.dirname(fn)
    if path != '':
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)


def _get_edge_color(s, t, g):
    if g.get_class_value(s) == g.get_class_value(t):
        if g.get_class_value(s) == const.MINORITY_VALUE:
            return viz.COLOR_MINORITY
        else:
            return viz.COLOR_MAJORITY
    return viz.COLOR_MIXED


def _plot_graph(g, ax, pos, **kwargs):
    node_shape = kwargs.get('node_shape', 'o')
    node_size = kwargs.get('node_size', 1)
    edge_width = kwargs.get('edge_width', 0.02)
    edge_style = kwargs.get('edge_style', 'solid')
    edge_arrows = kwargs.get('edge_arrows', True)
    arrow_style = kwargs.get('arrow_style', '-|>')
    arrow_size = kwargs.get('arrow_size', 2)

    # nodes
    maj = g.graph['class_values'][g.graph['class_labels'].index("M")]
    nodes, node_colors = zip(
        *[(node, viz.COLOR_MAJORITY if data[g.graph['class_attribute']] == maj else viz.COLOR_MINORITY)
          for node, data in g.nodes(data=True)])
    nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_size=node_size, node_color=node_colors,
                           node_shape=node_shape, ax=ax)

    # edges
    edges = g.edges()
    edges, edge_colors = zip(*[((s, t), _get_edge_color(s, t, g)) for s, t in edges])
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=edges, edge_color=edge_colors,
                           width=edge_width, style=edge_style, arrows=edge_arrows, arrowstyle=arrow_style,
                           arrowsize=arrow_size)


def _add_class_legend(fig: matplotlib.figure.Figure, **kwargs):
    maj_patch = mpatches.Patch(color=viz.COLOR_MAJORITY, label='majority')
    min_patch = mpatches.Patch(color=viz.COLOR_MINORITY, label='minority')
    bbox = kwargs.pop('bbox', (1.04, 1))
    loc = kwargs.pop('loc', "upper left")
    fig.legend(handles=[maj_patch, min_patch], bbox_to_anchor=bbox, loc=loc)


def _save_plot(fig: matplotlib.figure.Figure, fn=None, **kwargs):
    dpi = kwargs.pop('dpi', 300)
    bbox_inches = kwargs.pop('bbox_inches', 'tight')
    wspace = kwargs.pop('wspace', None)
    hspace = kwargs.pop('hspace', None)
    left = kwargs.pop('left', None)
    right = kwargs.pop('right', None)
    bottom = kwargs.pop('bottom', None)
    top = kwargs.pop('top', None)

    fig.tight_layout()
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    if fn is not None and fig is not None:
        validate_path(fn)
        fig.savefig(fn, dpi=dpi, bbox_inches=bbox_inches)
        print("%s saved" % fn)
    plt.show()
    plt.close()


def plot_samples(originals: List[netin.Graph], samples: List[List[netin.Graph]], fn: str = None, seed=None, **kwargs):
    """
    Plots the original graphs (row=0) and the samples (row=1,2,...) in a grid.
    Columns represent a different configuration of network structure, and rows (row>=1) represent a different sample.

    :param originals: list of original graphs
    :param samples: list of lists, where each sub-list contains the sample for each original graph
    :param fn: filename to save the plot
    :param kwargs: keyword arguments to pass to _plot_graph
    :return: None
    """
    # validate all sub-lists have the same size
    ngs = len(originals)
    if len(samples) == 0 or sum([len(ss) != ngs for ss in samples]) > 1:
        raise ValueError("The number of graphs in each sample must be the same, and there must be at least one sample.")

    # plot
    nr = 1 + len(samples)
    nc = len(originals)
    figsize = kwargs.pop('figsize', (10, 10))  # figure size (width, height)

    fig, axes = plt.subplots(nr, nc, figsize=figsize, sharex=False, sharey=False)

    # plot original graph
    for col, g in enumerate(originals):
        ax = axes[0, col]
        pos = nx.spring_layout(g, seed=seed)
        ax.set_title(g.get_model_name())
        title = get_title_graph(g)
        ax.set_title(title)
        _plot_graph(g, ax, pos, **kwargs)
        ax.set_axis_off()

        # plot samples
        for row, sample_set in enumerate(samples):
            ax = axes[row + 1, col]
            s = sample_set[col]
            # title = s.get_model_name()
            title = get_title_graph(s)
            ax.set_title(title)
            _plot_graph(s, ax, pos, **kwargs)
            ax.set_axis_off()

    _add_class_legend(fig, **kwargs)  # bbox, loc
    _save_plot(fig, fn, **kwargs)  # dpi, bbox_inches, wspace, hspace, left, right, bottom, top


def get_title_graph(g: netin.Graph):
    """
    Build the title of the graph by using the graph attributes.
    :param g:
    :return:
    """
    if 'method' in g.graph:
        title = f"{g.graph['method']}\n" \
                f"n={g.graph['n']} | " \
                f"e={g.graph['e']} | " \
                f"f$_m$={g.graph['f_m']:.2f} | " \
                f"sim={g.graph['similarity']:.2f}"
    else:
        title = f"{g.get_model_name()}\n" \
                f"n={g.graph['n']} | " \
                f"e={g.number_of_edges()} | " \
                f"f$_m$={g.graph['f_m']:.2f} | " \
                f"sim={net.get_similitude(g):.2f}"
    return title


def prepare_graph(g: Union[nx.Graph, nx.DiGraph], name: str, class_attribute: str) -> Union[
    netin.DiGraph, netin.UnDiGraph]:
    """
    Given a networkx graph, it creates a netin graph with the same structure and attributes.
    :param g:  networkx graph
    :param name:  name of the dataset
    :param class_attribute:  name of the attribute that contains the class label
    :return: netin graph
    """
    n = g.number_of_nodes()
    k = net.get_min_degree(g)
    f_m = net.get_minority_fraction(g, class_attribute)
    seed = None

    gn = UnDiGraph(n=n, k=k, f_m=f_m, seed=seed)
    gn.set_model_name(name)
    gn.add_edges_from(g.edges(data=True))
    gn.add_nodes_from(g.nodes(data=True))

    counter = Counter([obj[class_attribute] for n, obj in g.nodes(data=True)])
    class_values, class_counts = zip(*counter.most_common()) # from M to m
    class_labels = ['female' if c == 1 else 'male' if c == 0 else 'unknown' for c in class_values]
    gn._initialize(class_attribute=class_attribute, class_values=class_values, class_labels=class_labels)

    obj = {'model': gn.get_model_name(),
           'e': g.number_of_edges()}

    gn.graph.update(obj)
    return gn


def load_fb_data_as_networkx(path: str = 'data/fb_friends') -> nx.Graph:
    """
    Loads the Facebook data from the given path. It returns the graph as a netin.DiGraph or netin.UnDiGraph.

    Parameters
    ----------
    path: str
        path to the data

    Returns
    -------
        g
            netin.DiGraph or netin.UnDiGraph
    """
    def read_edges(fn: str) -> List[Tuple[int, int]]:
        edges = []
        with open(fn, 'r') as f:
            for line in f.readlines():
                if not line.startswith('#'):
                    s, t = line.split(',')
                    edges.append((int(s), int(t.strip())))
        return edges

    def read_gender(fn: str) -> dict:
        gender = {}
        with open(fn, 'r') as f:
            for line in f.readlines():
                if not line.startswith('#'):
                    n, a = line.split(',')
                    gender.update({int(n): int(a.strip())})
        return gender

    edges_fn = os.path.join(path, 'fb_friends.csv')
    edges = read_edges(edges_fn)
    node_fn = os.path.join(path, 'genders.csv')
    node_attr = read_gender(node_fn)

    g = nx.Graph()
    g.add_edges_from(edges)
    nx.set_node_attributes(g, node_attr, 'gender')

    unk = {n: -1 for n, obj in g.nodes(data=True) if 'gender' not in obj}
    nx.set_node_attributes(g, unk, 'gender')

    return g
