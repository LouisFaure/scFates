from anndata import AnnData
from typing import Optional, Union
import pandas as pd
import numpy as np
import scanpy as sc
from ..tools.utils import get_X
from ..tools.covariate import group_test
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scanpy.plotting._utils import savefig_or_show


def trend_covariate(
    adata: AnnData,
    gene: str,
    group_key: str,
    show_null: bool = False,
    ax: Optional = None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
    **kwargs
):

    df = pd.DataFrame(
        dict(
            t=adata.obs.t,
            exp=get_X(adata, adata.obs_names, gene, None).ravel(),
            groups=adata.obs[group_key],
        )
    )

    df = df.sort_values("t")

    if ax is None:
        ax = sc.pl.scatter(
            adata, x="t", y=gene, title=gene, color=group_key, show=False, **kwargs
        )
    else:
        sc.pl.scatter(
            adata,
            x="t",
            y=gene,
            title=gene,
            color=group_key,
            show=False,
            ax=ax,
            **kwargs
        )
    ax.set_ylabel("expression")

    cate = adata.obs[group_key].cat.categories

    m1, m0 = group_test(df, "groups", return_pred=True, trend_test=True)
    dct = dict(zip(adata.obs[group_key].cat.categories, (m1, m1)))
    dct_null = dict(zip(adata.obs[group_key].cat.categories, (m0, m0)))

    for g in cate:
        ax.plot(
            df.loc[df.groups == g, "t"],
            dct[g][df.groups == g],
            c=np.array(adata.uns[group_key + "_colors"])[cate == g][0],
            linewidth=2,
        )
        if show_null:
            ax.plot(
                df.loc[df.groups == g, "t"],
                dct_null[g][df.groups == g],
                c=np.array(adata.uns[group_key + "_colors"])[cate == g][0],
                linewidth=2,
                linestyle="--",
            )
            leg = True
            if "legend_loc" in kwargs:
                if kwargs["legend_loc"] == "none":
                    leg = False

            if leg:
                legend1 = ax.get_legend()

            legend2 = plt.legend(
                [
                    Line2D([0], [0], linewidth=1, color="k"),
                    Line2D([0], [0], linewidth=1, color="k", linestyle="--"),
                ],
                ["model", "null hypothesis"],
                loc="best",
                handlelength=2,
            )

            ax.add_artist(legend2)
            if leg:
                ax.add_artist(legend1)

    savefig_or_show("trend_covariate", show=show, save=save)

    if show == False:
        return ax
