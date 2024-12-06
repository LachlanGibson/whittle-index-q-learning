import matplotlib.pyplot as plt
import numpy as np
from .utilities import whittle_dependence


def plot_whittle_dependence(
        P, R, vkwarg,
        vals = None,
        xlab = None,
        legend = False,
        WIkwargs = {},
        **plotkwargs):
    
    if vals is None:
        vals = list(vkwarg.values())[0]
    if xlab is None:
        xlab = list(vkwarg.keys())[0]
    wi = whittle_dependence(P,R, vkwarg, **WIkwargs)
    for i in range(wi.shape[1]):
        if legend:
            plt.plot(vals, wi[:,i], **plotkwargs, label = "State " + str(i+1))
        else:
            plt.plot(vals, wi[:,i], **plotkwargs)
    plt.ylabel("Modified Whittle Index")
    plt.xlabel(xlab)
    if legend:
        plt.legend()


def plot_whittle_exploration_dependence(
        P, R,
        grid_size = [11, 5],
        WIkwargs = {},
        **plotkwargs):
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    er = np.linspace(0,1,grid_size[0]).tolist()
    ap = np.linspace(0,1,grid_size[1]).tolist()
    for i, prop in enumerate(ap):
        vkwarg = {"explore" : [[e, prop] for e in er]}
        plot_whittle_dependence(
            P,R,vkwarg,
            vals = er,
            xlab = "Exploration Rate",
            color = colours[i%len(colours)],
            WIkwargs = WIkwargs,
            **plotkwargs,
            label = "Activation probability = " + str(prop)[:4]
        )
    #plt.legend()


