from anndata import AnnData
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from joblib import delayed
from tqdm import tqdm
import sys
import igraph
from typing import Optional, Union

from .utils import ProgressParallel

from .. import logging as logg
from .. import settings


def pseudotime(
    adata: AnnData,
    n_jobs: int = 1,
    n_map: int = 1,
    seed: Optional[int] = None,
    copy: bool = False,
):
    """\
    Compute pseudotime.

    Projects cells onto the tree, and uses distance from the root as a pseudotime value.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_jobs
        Number of cpu processes to use in case of performing multiple mapping.
    n_map
        number of probabilistic mapping of cells onto the tree to use. If n_map=1 then likelihood cell mapping is used.
    seed
        A numpy random seed for reproducibility for muliple mappings
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.obs['edge']`
            assigned edge.
        `.obs['t']`
            assigned pseudotime value.
        `.obs['seg']`
            assigned segment of the tree.
        `.obs['milestone']`
            assigned region surrounding forks and tips.
        `.uns['pseudotime_list']`
            list of cell projection from all mappings.
    """

    if "root" not in adata.uns["graph"]:
        raise ValueError(
            "You need to run `tl.root` or `tl.roots` before projecting cells."
        )

    adata = adata.copy() if copy else adata

    graph = adata.uns["graph"]
    pp_seg = adata.uns["graph"]["pp_seg"]
    pp_info = adata.uns["graph"]["pp_info"]

    reassign, recolor = False, False
    if "milestones" in adata.obs:
        if adata.obs.milestones.dtype.name == "category":
            tmp_mil = adata.obs.milestones.cat.categories.copy()
            reassign = True
        if "milestones_colors" in adata.uns:
            tmp_mil_col = adata.uns["milestones_colors"].copy()
            recolor = True

    logg.info("projecting cells onto the principal graph", reset=True)

    from sklearn.metrics import pairwise_distances

    P = pairwise_distances(graph["F"].T, metric=graph["metrics"])

    if n_map == 1:
        df_l = [map_cells(graph, R=adata.obsm["X_R"], P=P, multi=False)]
    else:
        if seed is not None:
            np.random.seed(seed)
            map_seeds = np.random.randint(999999999, size=n_map)
        else:
            map_seeds = [None for i in range(n_map)]
        df_l = ProgressParallel(
            n_jobs=n_jobs, total=n_map, file=sys.stdout, desc="    mappings"
        )(
            delayed(map_cells)(
                graph=graph, R=adata.obsm["X_R"], P=P, multi=True, map_seed=map_seeds[m]
            )
            for m in range(n_map)
        )

    # formatting cell projection data
    for i in range(len(df_l)):
        df_l[i].index = adata.obs_names
    df_summary = df_l[0]

    df_summary["seg"] = df_summary["seg"].astype("category")
    df_summary["edge"] = df_summary["edge"].astype("category")

    # remove pre-existing palette to avoid errors with plotting
    if "seg_colors" in adata.uns:
        del adata.uns["seg_colors"]

    if set(df_summary.columns.tolist()).issubset(adata.obs.columns):
        adata.obs[df_summary.columns] = df_summary
    else:
        adata.obs = pd.concat([adata.obs, df_summary], axis=1)

    # list(map(lambda x: x.column))

    # todict=list(map(lambda x: dict(zip(["cells"]+["_"+s for s in x.columns.tolist()],
    #                                   [x.index.tolist()]+x.to_numpy().T.tolist())),df_l))
    names = np.arange(len(df_l)).astype(str).tolist()
    # vals = todict
    dictionary = dict(zip(names, df_l))
    adata.uns["pseudotime_list"] = dictionary

    if n_map > 1:
        adata.obs["t_sd"] = (
            pd.concat(
                list(
                    map(
                        lambda x: pd.Series(x["t"]),
                        list(adata.uns["pseudotime_list"].values()),
                    )
                ),
                axis=1,
            )
            .apply(np.std, axis=1)
            .values
        )

        # reassign cells to their closest segment
        root = adata.uns["graph"]["root"]
        tips = adata.uns["graph"]["tips"]
        endpoints = tips[tips != root]

        allsegs = pd.concat(
            [df.seg for df in adata.uns["pseudotime_list"].values()], axis=1
        )
        allsegs = allsegs.apply(lambda x: x.value_counts(), axis=1)
        adata.obs.seg = allsegs.idxmax(axis=1)
        adata.obs.t = pd.concat(
            [df.t for df in adata.uns["pseudotime_list"].values()], axis=1
        ).mean(axis=1)

        for s in pp_seg.n:
            df_seg = adata.obs.loc[adata.obs.seg == s, "t"]

            # reassign cells below minimum pseudotime of their assigned seg
            if any(int(s) == pp_seg.index[pp_seg["from"] != root]):
                start_t = pp_info.loc[pp_seg["from"], "time"].iloc[int(s) - 1]
                cells_back = allsegs.loc[df_seg[df_seg < start_t].index]
                ncells = cells_back.shape[0]
                if ncells != 0:
                    filter_from = pd.concat(
                        [pp_info.loc[pp_seg["from"], "time"] for i in range(ncells)],
                        axis=1,
                    ).T.values
                    filter_to = pd.concat(
                        [pp_info.loc[pp_seg["to"], "time"] for i in range(ncells)],
                        axis=1,
                    ).T.values
                    t_cells = adata.obs.loc[cells_back.index, "t"]

                    boo = (filter_from < t_cells.values.reshape((-1, 1))) & (
                        filter_to > t_cells.values.reshape((-1, 1))
                    )

                    cells_back = (cells_back.fillna(0) * boo).apply(
                        lambda x: x.index[np.argsort(x)][::-1], axis=1
                    )
                    cells_back = cells_back.apply(lambda x: x[x != s][0])
                    adata.obs.loc[cells_back.index, "seg"] = cells_back.values

            # reassign cells over maximum pseudotime of their assigned seg
            if any(int(s) == pp_seg.index[~pp_seg.to.isin(endpoints)]):
                end_t = pp_info.loc[pp_seg["to"], "time"].iloc[int(s) - 1]
                cells_front = allsegs.loc[df_seg[df_seg > end_t].index]
                ncells = cells_front.shape[0]
                if ncells != 0:
                    filter_from = pd.concat(
                        [pp_info.loc[pp_seg["from"], "time"] for i in range(ncells)],
                        axis=1,
                    ).T.values
                    filter_to = pd.concat(
                        [pp_info.loc[pp_seg["to"], "time"] for i in range(ncells)],
                        axis=1,
                    ).T.values

                    t_cells = adata.obs.loc[cells_front.index, "t"]

                    boo = (filter_to > t_cells.values.reshape((-1, 1))) & (
                        filter_from < t_cells.values.reshape((-1, 1))
                    )

                    cells_front = (cells_front.fillna(0) * boo).apply(
                        lambda x: x.index[np.argsort(x)][::-1], axis=1
                    )
                    cells_front = cells_front.apply(lambda x: x[x != s][0])
                    adata.obs.loc[cells_front.index, "seg"] = cells_front.values

    milestones = pd.Series(index=adata.obs_names, dtype=str)
    for seg in pp_seg.n:
        cell_seg = adata.obs.loc[adata.obs["seg"] == seg, "t"]
        if len(cell_seg) > 0:
            milestones[
                cell_seg.index[
                    (cell_seg - min(cell_seg) - (max(cell_seg - min(cell_seg)) / 2) < 0)
                ]
            ] = pp_seg.loc[int(seg), "from"]
            milestones[
                cell_seg.index[
                    (cell_seg - min(cell_seg) - (max(cell_seg - min(cell_seg)) / 2) > 0)
                ]
            ] = pp_seg.loc[int(seg), "to"]
    adata.obs["milestones"] = milestones
    adata.obs.milestones = (
        adata.obs.milestones.astype(int).astype("str").astype("category")
    )

    adata.uns["graph"]["milestones"] = dict(
        zip(
            adata.obs.milestones.cat.categories,
            adata.obs.milestones.cat.categories.astype(int),
        )
    )

    # setting consistent color palettes
    from ..plot import palette_tools

    palette_tools._set_default_colors_for_categorical_obs(adata, "milestones")
    while reassign:
        if "tmp_mil_col" not in locals():
            break
        if len(tmp_mil_col) != len(adata.obs.milestones.cat.categories):
            break
        rename_milestones(adata, tmp_mil)
        if recolor:
            adata.uns["milestones_colors"] = tmp_mil_col
        reassign = False
    adata.uns["seg_colors"] = [
        np.array(adata.uns["milestones_colors"])[
            pd.Series(adata.uns["graph"]["milestones"]) == t
        ][0]
        for t in adata.uns["graph"]["pp_seg"].to
    ]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n"
        "    .obs['edge'] assigned edge.\n"
        "    .obs['t'] pseudotime value.\n"
        "    .obs['seg'] segment of the tree assigned.\n"
        "    .obs['milestones'] milestone assigned.\n"
        "    .uns['pseudotime_list'] list of cell projection from all mappings."
    )

    return adata if copy else None


def map_cells(graph, R, P, multi=False, map_seed=None):
    import igraph

    g = igraph.Graph.Adjacency((graph["B"] > 0).tolist(), mode="undirected")
    # Add edge weights and node labels.
    g.es["weight"] = np.array(P[graph["B"].nonzero()].ravel()).tolist()[0]

    if multi:
        np.random.seed(map_seed)
        rrm = (
            np.apply_along_axis(
                lambda x: np.random.choice(np.arange(len(x)), size=1, p=x),
                axis=1,
                arr=R,
            )
        ).T.flatten()
    else:
        rrm = np.apply_along_axis(np.argmax, axis=1, arr=R)

    def map_on_edges(v):
        vcells = np.argwhere(rrm == v)

        if vcells.shape[0] > 0:
            nv = np.array(g.neighborhood(v, order=1))[1:]
            nvd = g.shortest_paths(v, nv, weights=g.es["weight"])

            ndf = pd.DataFrame(
                {
                    "cell": vcells.flatten(),
                    "v0": v,
                    "v1": nv[np.argmin(nvd)],
                    "d": np.min(nvd),
                }
            )

            p0 = R[vcells, v].flatten()
            p1 = np.array(
                list(map(lambda x: R[vcells[x], ndf.v1[x]], range(len(vcells))))
            ).flatten()

            ndf["t"] = [
                np.average(
                    graph["pp_info"].time[ndf.iloc[i, [1, 2]].astype(int)],
                    weights=[p0[i], p1[i]],
                )
                for i in range(ndf.shape[0])
            ]

            ndf["seg"] = 0
            isinfork = (graph["pp_info"].loc[ndf.v0, "PP"].isin(graph["forks"])).values
            ndf.loc[isinfork, "seg"] = (
                graph["pp_info"].loc[ndf.loc[isinfork, "v1"], "seg"].values
            )
            ndf.loc[~isinfork, "seg"] = (
                graph["pp_info"].loc[ndf.loc[~isinfork, "v0"], "seg"].values
            )

            return ndf
        else:
            return None

    df = list(map(map_on_edges, range(graph["B"].shape[1])))
    df = pd.concat(df)
    df.sort_values("cell", inplace=True)
    # df.index = graph["cells_fitted"]

    df["edge"] = df.apply(lambda x: str(int(x[1])) + "|" + str(int(x[2])), axis=1)

    df.drop(["cell", "v0", "v1", "d"], axis=1, inplace=True)

    return df


def rename_milestones(adata, new: Union[list, dict], copy: bool = False):

    adata = adata.copy() if copy else adata

    if isinstance(new, dict):
        old = pd.Series(
            adata.uns["graph"]["milestones"].keys(),
            index=adata.uns["graph"]["milestones"].values(),
        )
        dct = pd.Series(new)
        dct.index = dct.index.astype(int)
        adata.uns["graph"]["milestones"] = dict(
            zip(dct.loc[old.index].values, adata.uns["graph"]["milestones"].values())
        )
        new = dct.loc[old.index].values
    else:
        adata.uns["graph"]["milestones"] = dict(
            zip(new, list(adata.uns["graph"]["milestones"].values()))
        )

    adata.obs.milestones = adata.obs.milestones.cat.rename_categories(new)

    return adata if copy else None
