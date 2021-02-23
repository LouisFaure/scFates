from anndata import AnnData
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from joblib import delayed, Parallel
from tqdm import tqdm
import sys
import igraph

from .. import logging as logg
from .. import settings


def pseudotime(adata: AnnData, n_jobs: int = 1, n_map: int = 1, copy: bool = False):
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

    logg.info("projecting cells onto the principal graph", reset=True)

    if n_map == 1:
        df_l = [map_cells(graph, multi=False)]
    else:
        df_l = Parallel(n_jobs=n_jobs)(
            delayed(map_cells)(graph=graph, multi=True)
            for m in tqdm(range(n_map), file=sys.stdout, desc="    mappings")
        )

    # formatting cell projection data

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

    milestones = pd.Series(index=adata.obs_names)
    for seg in graph["pp_seg"].n:
        cell_seg = adata.obs.loc[adata.obs["seg"] == seg, "t"]
        if len(cell_seg) > 0:
            milestones[
                cell_seg.index[
                    (cell_seg - min(cell_seg) - (max(cell_seg - min(cell_seg)) / 2) < 0)
                ]
            ] = graph["pp_seg"].loc[int(seg), "from"]
            milestones[
                cell_seg.index[
                    (cell_seg - min(cell_seg) - (max(cell_seg - min(cell_seg)) / 2) > 0)
                ]
            ] = graph["pp_seg"].loc[int(seg), "to"]
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


def map_cells(graph, multi=False):
    import igraph

    g = igraph.Graph.Adjacency((graph["B"] > 0).tolist(), mode="undirected")
    # Add edge weights and node labels.
    g.es["weight"] = graph["B"][graph["B"].nonzero()]
    if multi:
        rrm = (
            np.apply_along_axis(
                lambda x: np.random.choice(np.arange(len(x)), size=1, p=x),
                axis=1,
                arr=graph["R"],
            )
        ).T.flatten()
    else:
        rrm = np.apply_along_axis(np.argmax, axis=1, arr=graph["R"])

    def map_on_edges(v):
        vcells = np.argwhere(rrm == v)

        if vcells.shape[0] > 0:
            nv = np.array(g.neighborhood(v, order=1))
            nvd = np.array(g.shortest_paths(v, nv)[0])

            spi = np.apply_along_axis(np.argmax, axis=1, arr=graph["R"][vcells, nv[1:]])
            ndf = pd.DataFrame(
                {
                    "cell": vcells.flatten(),
                    "v0": v,
                    "v1": nv[1:][spi],
                    "d": nvd[1:][spi],
                }
            )

            p0 = graph["R"][vcells, v].flatten()
            p1 = np.array(
                list(
                    map(lambda x: graph["R"][vcells[x], ndf.v1[x]], range(len(vcells)))
                )
            ).flatten()

            alpha = np.random.uniform(size=len(vcells))
            f = np.abs(
                (np.sqrt(alpha * p1 ** 2 + (1 - alpha) * p0 ** 2) - p0) / (p1 - p0)
            )
            ndf["t"] = (
                graph["pp_info"].loc[ndf.v0, "time"].values
                + (
                    graph["pp_info"].loc[ndf.v1, "time"].values
                    - graph["pp_info"].loc[ndf.v0, "time"].values
                )
                * alpha
            )
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
    df.index = graph["cells_fitted"]

    df["edge"] = df.apply(lambda x: str(int(x[1])) + "|" + str(int(x[2])), axis=1)

    df.drop(["cell", "v0", "v1", "d"], axis=1, inplace=True)

    return df


def refine_pseudotime(
    adata: AnnData, n_jobs: int = 1, ms_data=None, use_rep=None, copy: bool = False
):

    """\
    Refine computed pseudotime.

    Projection using principal graph can lead to compressed pseudotimes for the cells localised
    near the tips. To counteract this, diffusion based pseudotime is performed using Palantir [Setty19]_ on each
    segment separately.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_jobs
        Number of cpu processes (max is the number of segments).
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.obs['t']`
            updated assigned pseudotimes value.
        `.obs['t_old']`
            previously assigned pseudotime.
    """
    import palantir
    from .utils import palantir_on_seg

    adata = adata.copy() if copy else adata

    adata.obs["t_old"] = adata.obs.t.copy()

    logg.info(
        "refining pseudotime using palantir on each segment of the tree", reset=True
    )

    if use_rep is not None:
        dm_res = palantir.utils.run_diffusion_maps(
            pd.DataFrame(adata.obsm["X_" + use_rep], index=adata.obs_names)
        )
        ms_data = palantir.utils.determine_multiscale_space(dm_res)
    elif ms_data is not None:
        ms_data = pd.DataFrame(adata.obsm["X_" + ms_data], index=adata.obs_names)
    else:
        dm_res = palantir.utils.run_diffusion_maps(
            pd.DataFrame(adata.X, index=adata.obs_names)
        )
        ms_data = palantir.utils.determine_multiscale_space(dm_res)

    pseudotimes = Parallel(n_jobs=n_jobs)(
        delayed(palantir_on_seg)(adata, seg=s, ms_data=ms_data)
        for s in tqdm(
            adata.uns["graph"]["pp_seg"].n.values.astype(str), file=sys.stdout
        )
    )

    g = igraph.Graph(directed=True)
    g.add_vertices(
        np.unique(
            adata.uns["graph"]["pp_seg"]
            .loc[:, ["from", "to"]]
            .values.flatten()
            .astype(str)
        )
    )
    g.add_edges(adata.uns["graph"]["pp_seg"].loc[:, ["from", "to"]].values.astype(str))

    roots = [adata.uns["graph"]["root"]] + [
        adata.uns["graph"]["root2"] if "root2" in adata.uns["graph"] else None
    ]
    root = adata.uns["graph"]["tips"][np.isin(adata.uns["graph"]["tips"], roots)]
    tips = adata.uns["graph"]["tips"][~np.isin(adata.uns["graph"]["tips"], roots)]

    dt = 0
    if "meeting" in adata.uns["graph"]:
        path_to_meet = list(
            map(
                lambda r: g.get_shortest_paths(
                    str(r), str(adata.uns["graph"]["meeting"])
                )[0],
                root,
            )
        )
        for p in path_to_meet:
            pth = np.array(g.vs["name"], dtype=int)[p]
            for i in range(len(pth) - 1):
                sel = (
                    adata.uns["graph"]["pp_seg"]
                    .loc[:, ["from", "to"]]
                    .apply(lambda x: np.all(x == pth[i : i + 2]), axis=1)
                    .values
                )
                adata.obs.loc[
                    adata.obs.seg.isin(adata.obs.seg.cat.categories[sel]), "t"
                ] = (
                    pseudotimes[np.argwhere(sel)[0][0]]
                    * adata.uns["graph"]["pp_seg"].loc[sel, "d"].values[0]
                ).values
            dt_temp = adata.uns["graph"]["pp_seg"].loc[sel, "d"].values[0]
            dt = dt_temp if dt_temp > dt else dt

        allpth = g.get_shortest_paths(
            str(adata.uns["graph"]["meeting"]), [str(t) for t in tips]
        )
        dt_prev = dt
    else:
        allpth = g.get_shortest_paths(str(root[0]), [str(t) for t in tips])
        dt_prev = 0

    for p in allpth:
        pth = np.array(g.vs["name"], dtype=int)[p]
        dt = dt_prev
        for i in range(len(pth) - 1):
            sel = (
                adata.uns["graph"]["pp_seg"]
                .loc[:, ["from", "to"]]
                .apply(lambda x: np.all(x == pth[i : i + 2]), axis=1)
                .values
            )
            adata.obs.loc[
                adata.obs.seg == adata.uns["graph"]["pp_seg"].loc[sel, "n"].values[0],
                "t",
            ] = (
                pseudotimes[np.argwhere(sel)[0][0]]
                * adata.uns["graph"]["pp_seg"].loc[sel, "d"].values[0]
            ).values + dt
            dt = dt + adata.uns["graph"]["pp_seg"].loc[sel, "d"].values[0]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "updated\n" + "    .obs['t'] palantir refined pseudotime values.\n"
        "added\n" + "    .obs['t_old'] previous pseudotime data."
    )

    return adata if copy else None


def rename_milestones(adata, new, copy: bool = False):

    adata = adata.copy() if copy else adata

    adata.uns["graph"]["milestones"] = dict(
        zip(new, list(adata.uns["graph"]["milestones"].values()))
    )

    adata.obs.milestones = adata.obs.milestones.cat.rename_categories(new)

    return adata if copy else None
