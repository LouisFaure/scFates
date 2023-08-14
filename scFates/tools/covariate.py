from typing import Optional, Iterable
from .utils import ProgressParallel, importeR, get_X
from .test_association import test_association
from .. import settings
from .. import logging as logg
from joblib import delayed
import sys
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import itertools

a, b, rstat, rmgcv, Formula = importeR("covariate testing")
import importlib

if importlib.util.find_spec("rpy2") is not None:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri


def test_covariate(
    adata: AnnData,
    group_key: str,
    features: Optional[Iterable] = None,
    seg: Optional[str] = None,
    layer: Optional[str] = None,
    trend_test: bool = False,
    fdr_cut: float = 0.01,
    n_jobs: int = 1,
    n_map: int = 1,
    copy: bool = False,
):
    """\
    Test for branch differential gene expression between covariates on the same trajectory path.

    **Test of amplitude difference**

    The same is used as in :func:`scFates.tl.test_fork`.
    This uses the following model :

    :math:`g_{i} \\sim\ s(pseudotime)+s(pseudotime):Covariate+Covariate`

    Where :math:`s(.)` denotes the penalized regression spline function and
    :math:`s(pseudotime):Covariate` denotes interaction between the smoothed pseudotime and covariate terms.
    From this interaction term, the p-value is extracted.


    **Test of trend difference**

    Inspired from a preprint [Ji22]_, this test compares the following full model:

    :math:`g_{i} \\sim\ s(pseudotime)+s(pseudotime):Covariate+Covariate`

    to the following reduced one:

    :math:`g_{i} \\sim\ s(pseudotime)+s(pseudotime)+Covariate`

    Comparison is done using ANOVA



    Parameters
    ----------
    adata
        Annotated data matrix.
    group_key
        key in `.obs` for the covariates to test.
    features
        Which features to test (all significants by default).
    seg
        In the case of a tree, which segment to use for such test.
    layer
        layer to use for the test
    trend_test
        Whether to perform the trend test instead of amplitude test.
    n_jobs
        number of cpu processes used to perform the test.
    n_map
        number of cell mappings from which to do the test (not implemented yet).
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.var['cov_pval' or 'covtrend_pval']`
            pvalues extracted from tests.
        `.var['cov_fdr' or 'covtrend_fdr']`
            FDR extracted from the pvalues.
        `.var['cov_signi' or 'covtrend_signi']`
            is the feature significant.
        `.var['A->B_lfc']`
            logfoldchange in expression between covariate A and B.

    """

    teststr = "trend" if trend_test else "amplitude"
    logg.info(f"testing covariates ({teststr})", reset=True)

    adata = adata.copy() if copy else adata

    if adata.obs[group_key].dtype.name != "category":
        adata.obs[group_key] = adata.obs[group_key].astype("category")
    segcells = (
        adata.obs_names[adata.obs.seg == seg] if seg is not None else adata.obs_names
    )
    features = adata.var_names if features is None else features
    Xgenes = get_X(adata, segcells, features, layer, togenelist=True)

    if "log1p" in adata.uns_keys() and adata.uns["log1p"]["base"] is not None:
        logbase = np.log(adata.uns["log1p"]["base"])
    else:
        logbase = 1
    dfs = [
        pd.DataFrame(
            dict(t=adata.obs.t, exp=X, groups=adata.obs[group_key], logbase=logbase)
        )
        for X in Xgenes
    ]

    stat = ProgressParallel(
        n_jobs=n_jobs,
        use_tqdm=n_map == 1,
        total=len(dfs),
        file=sys.stdout,
        desc="    single mapping ",
    )(delayed(group_test)(df, "groups", trend_test, logbase) for df in dfs)

    res = "covtrend" if trend_test else "cov"
    adata.var[f"{res}_pval"] = np.nan
    adata.var[f"{res}_fdr"] = np.nan
    adata.var[f"{res}_signi"] = np.nan

    adata.var.loc[features, f"{res}_pval"] = [s[0] for s in stat]
    adata.var.loc[features, f"{res}_fdr"] = multipletests(
        adata.var.loc[features, f"{res}_pval"], method="bonferroni"
    )[1]
    adata.var.loc[features, f"{res}_signi"] = (
        adata.var.loc[features, f"{res}_fdr"] < fdr_cut
    ) * 1

    adata.var["->".join(adata.obs[group_key].cat.categories) + "_lfc"] = np.nan
    adata.var.loc[features, "->".join(adata.obs[group_key].cat.categories) + "_lfc"] = [
        s[1] for s in stat
    ]
    covstr = "covtrend" if trend_test else "cov"
    logg.info(
        "    found " + str(sum(adata.var[f"{res}_signi"])) + " significant features",
        time=True,
        end="\n",
    )
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        "    .var['" + covstr + "_pval'], pvalues extracted from tests.\n"
        "    .var['" + covstr + "_FDR'], FDR extracted from the pvalues.\n"
        "    .var['" + covstr + "_signi'], is the feature significant.\n"
        "    .var['" + "->".join(adata.obs[group_key].cat.categories) + "_lfc'],"
        " logfoldchange in expression between covariate "
        + " and ".join(adata.obs[group_key].cat.categories)
        + "."
    )


def group_test(df, group, trend_test=False, logbase=None, return_pred=False):

    global rmgcv

    if not return_pred:
        mean_A, mean_B = [
            df.exp[df[group] == g].mean() for g in df[group].cat.categories
        ]
        foldchange = (np.expm1(mean_A * logbase) + 1e-9) / (
            np.expm1(mean_B * logbase) + 1e-9
        )
        lfc = np.log2(foldchange)

    if trend_test:
        m1 = rmgcv.gam(
            Formula(
                f"exp ~ s(t,k=5)+s(t,by=as.factor({group}),k=5)+as.factor({group})"
            ),
            data=df,
        )
        m0 = rmgcv.gam(Formula(f"exp ~ s(t,k=5)+as.factor({group})"), data=df)
        if return_pred:
            return (rstat.predict(m1), rstat.predict(m0))
        else:
            test = rmgcv.anova_gam(m1, m0, test="F")
            with (ro.default_converter + pandas2ri.converter).context():
                test_df = ro.conversion.get_conversion().rpy2py(test)
            pval = test_df.loc["2", ["Pr(>F)"]].values[0]
            return (pval, lfc)
    else:
        m = rmgcv.gam(
            Formula(
                f"exp ~ s(t,k=5)+s(t,by=as.factor({group}),k=5)+as.factor({group})"
            ),
            data=df,
        )
        if return_pred:
            return (rstat.predict(m), rstat.predict(m))
        else:
            pval = rmgcv.summary_gam(m)[3][1]
            return (pval, lfc)


def test_association_covariate(
    adata: AnnData, group_key: str, copy: bool = False, **kwargs
):
    """\
    Separately test for associated features for each covariates on the same trajectory path.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_key
        key in `.obs` for the covariates to test.
    copy
        Return a copy instead of writing to adata.
    **kwargs
        Arguments passed to :func:`scFates.tl.test_association`

    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.var['cov_pval' or 'covtrend_pval']`
            pvalues extracted from tests.
        `.var['cov_fdr' or 'covtrend_fdr']`
            FDR extracted from the pvalues.
        `.var['cov_signi' or 'covtrend_signi']`
            is the feature significant.
        `.var['A->B_lfc']`
            logfoldchange in expression between covariate A and B.

    """

    logg.info("test association covariates", reset=True)

    adata = adata.copy() if copy else adata

    if adata.obs[group_key].dtype.name != "category":
        adata.obs[group_key] = adata.obs[group_key].astype("category")

    def gather_stats(sdata, group):
        test_association(sdata, **kwargs)
        sdata.var = sdata.var[["p_val", "A", "fdr", "signi"]]
        sdata.var.columns = group + "_" + sdata.var.columns
        return sdata.var

    reset = False
    if settings.verbosity > 2:
        temp_verb = settings.verbosity
        settings.verbosity = 2
        reset = True
    var = [
        gather_stats(adata[adata.obs[group_key] == g], g)
        for g in adata.obs[group_key].cat.categories
    ]

    for v in var:
        for c in v.columns:
            adata.var[c] = v[c]

    adata.var["signi"] = pd.concat(
        [adata.var[g + "_signi"] for g in adata.obs[group_key].cat.categories], axis=1
    ).apply(any, axis=1)
    if reset:
        settings.verbosity = temp_verb

    strtoshow = [
        [
            "    .var['" + group + "_pval'] values from statistical test.\n",
            "    .var['" + group + "_fdr'] corrected values from multiple testing.\n",
            "    .var['" + group + "_A'] amplitue of change of tested feature.\n",
            "    .var['"
            + group
            + "_signi'] feature is significantly changing along pseudotime.\n",
        ]
        for group in adata.obs[group_key].cat.categories
    ]

    strtoshow = itertools.chain(*strtoshow)

    logg.info("    finished", time=False, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        + "".join(strtoshow)
        + "    .var['signi'], intersection of both significant list of genes."
    )
