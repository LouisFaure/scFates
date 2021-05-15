import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData


def linearity_deviation(adata: AnnData, start_milestone, end_milestone, ntop_genes=30):

    name = start_milestone + "->" + end_milestone

    topgenes = adata.var[name + "_rss"].sort_values(ascending=False)[:ntop_genes]
    ymin = np.min(topgenes)
    ymax = np.max(topgenes)
    ymax += 0.3 * (ymax - ymin)

    fig, ax = plt.subplots()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.9, len(topgenes) - 0.1)
    for ig, gene_name in enumerate(topgenes.index):
        ax.text(
            ig,
            topgenes[gene_name],
            gene_name,
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
        )

    ax.set_xlabel("ranking")
    ax.set_ylabel("deviance from linearity")
