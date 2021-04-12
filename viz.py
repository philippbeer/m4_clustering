"""
This module organize the visualization functions
"""
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from sklearn.metrics import silhouette_samples
from tqdm import tqdm


def create_sil_diagram(kmeans_per_k, X,
                       top_scores_indexes,
                       file_name,
                       silhouette_scores) -> None:
    """
    create silhoutte diagram for passed models
    (adapted from HOML book)
    """
    plt.figure(figsize=(11, 9))
    
    
    # no. of subplots
    no_sils = len(top_scores_indexes)
    n_rows = 1
    n_cols = 1
    if no_sils < 2:
        n_cols = 1
    else:
        n_cols = 3
        n_rows = math.ceil(no_sils/3)
    j = 1 # counter for subplot position
    for idx in top_scores_indexes:
        plt.subplot(n_rows, n_cols, j)
        j += 1
        k = kmeans_per_k[idx].cluster_centers_.shape[0]
        y_pred = kmeans_per_k[idx].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)
        
        padding = X.shape[0]//30
        pos = padding
        ticks = []

        print('#### kMeans silhouette coefficient ####')
        for i in tqdm(range(k)):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        plt.ylabel("Cluster")
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
        plt.axvline(x=silhouette_scores[idx], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    save_fig(file_name)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    if not os.path.exists('img'):
        os.makedirs('img')
    path = os.path.join('./img/', fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


    
