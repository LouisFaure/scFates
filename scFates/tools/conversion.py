import numpy as np
import cellrank as cr
import matplotlib.pyplot as plt

from .graph_tools import *
from .pseudotime import *
from .. import logging as logg
from .. import settings


def cellrank_to_tree(
    adata,
    time,
    Nodes,
    method="ppt",
    ppt_lambda=20,
    auto_root=True,
    reassign_pseudotime=True,
    plot_circular=False,
    copy=False,
    **kwargs
):

    logg.info(
        "Converting CellRank results to a principal tree",
        end=" " if settings.verbosity > 2 else "\n",
    )

    adata = adata.copy() if copy else adata

    n_states = adata.obsm["to_terminal_states"].shape[1]

    if n_states == 2:
        adata.obsm["X_fates"] = np.vstack(
            [
                np.array(adata.obsm["to_terminal_states"][:, 0].flatten()),
                adata.obs[time],
            ]
        ).T
        logg.hint(
            "with .obsm['X_fates'], created by combining:\n"
            "    .obsm['to_terminal_states'][:,0] and adata.obs['" + time + "']\n"
        )
    else:
        logg.hint(
            "with .obsm['X_fates'], created by combining:\n"
            "    .obsm['X_fate_simplex_fwd'] (from cr.pl.circular_projection) and adata.obs['"
            + time
            + "']\n"
        )
        cr.pl.circular_projection(adata, keys=["kl_divergence"])
        plt.close()

        adata.obsm["X_fates"] = np.concatenate(
            [adata.obsm["X_fate_simplex_fwd"], adata.obs[time].values.reshape(-1, 1)],
            axis=1,
        )

    tree(
        adata,
        Nodes=Nodes,
        use_rep="X_fates",
        method=method,
        ppt_lambda=ppt_lambda,
        **kwargs
    )

    if auto_root:
        logg.info("\nauto selecting a root using " + time + ".\n")
        tips = adata.uns["graph"]["tips"]
        R = adata.uns["graph"]["R"]
        root_sel = tips[adata.obs[time].iloc[R[:, tips].argmax(axis=0)].values.argmin()]
        root(adata, int(root_sel))
        pseudotime(adata)

    logg.info("\nfinished", time=True, end="\n" if reassign_pseudotime else " ")

    if reassign_pseudotime:
        adata.obs["t"] = adata.obs[time]
        logg.info("    .obs['t'] has been replaced by .obs['" + time + "']")

    return adata if copy else None
