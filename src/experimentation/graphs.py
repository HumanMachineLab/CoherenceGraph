import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def display_pk_wd_proximity(pred_thresholds, pk, wd, proximity, file=None):
    
    plt.style.use("ggplot")
    # plot lines
    plt.plot(pred_thresholds, pk, label = "pk")
    plt.plot(pred_thresholds, wd, label = "wd")
    plt.plot(pred_thresholds, proximity, label = "prox")
    plt.legend()
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    # plt.show()

    if file is not None:
        # plt.grid(True)
        tikzplotlib.save(file)
        plt.cla()
    else:
        plt.show()
        plt.cla()