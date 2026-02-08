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
    nested: bool = False,
    fdr_cut: float = 0.01,
    n_jobs: int = 1,
    n_map: int = 1,
    copy: bool = False,
):
    r"""
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

    :math:`g_{i} \\sim\ s(pseudotime)+Covariate`

    Comparison is done using ANOVA


    **Nested test**

    This performs two tests:
    1. Shared trend: :math:`g_{i} \\sim\ s(pseudotime)+Covariate` vs :math:`g_{i} \\sim\ Covariate`
    2. Specific trend: :math:`g_{i} \\sim\ s(pseudotime)+s(pseudotime):Covariate+Covariate` vs :math:`g_{i} \\sim\ s(pseudotime)+Covariate`


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
    nested
        Whether to perform the nested suite of tests (shared and specific trends).
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
        `.var['{covariate}_lfc']`
            logfoldchange in expression between covariate and the rest of the cells.
        If `nested=True`:
        `.var['shared_pval']` and `.var['spec_pval']`
            pvalues for shared and specific trends.

    """

    teststr = "nested" if nested else ("trend" if trend_test else "amplitude")
    logg.info(f"testing covariates ({teststr})", reset=True)

    adata = adata.copy() if copy else adata

    if adata.obs[group_key].dtype.name != "category":
        adata.obs[group_key] = adata.obs[group_key].astype("category")
    segcells = (
        adata.obs_names[adata.obs.seg == seg] if seg is not None else adata.obs_names
    )
    features = adata.var_names if features is None else features
    Xgenes = get_X(adata, segcells, features, layer, togenelist=True)

    if "log1p" in adata.uns and adata.uns["log1p"]["base"] is not None:
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
    )(delayed(group_test)(df, "groups", trend_test, logbase, nested=nested) for df in dfs)

    if nested:
        adata.var["shared_pval"] = np.nan
        adata.var["shared_fdr"] = np.nan
        adata.var["shared_signi"] = np.nan
        adata.var["shared_A"] = np.nan
        adata.var["spec_pval"] = np.nan
        adata.var["spec_fdr"] = np.nan
        adata.var["spec_signi"] = np.nan

        shared_pvals = np.array([s[0] for s in stat])
        spec_pvals = np.array([s[1] for s in stat])
        
        shared_pvals[np.isnan(shared_pvals)] = 1.0
        spec_pvals[np.isnan(spec_pvals)] = 1.0

        adata.var.loc[features, "shared_pval"] = shared_pvals
        adata.var.loc[features, "spec_pval"] = spec_pvals
        adata.var.loc[features, "shared_A"] = [s[2] for s in stat]

        adata.var.loc[features, "shared_fdr"] = multipletests(
            adata.var.loc[features, "shared_pval"], method="fdr_bh"
        )[1]
        adata.var.loc[features, "spec_fdr"] = multipletests(
            adata.var.loc[features, "spec_pval"], method="fdr_bh"
        )[1]

        adata.var.loc[features, "shared_signi"] = (
            adata.var.loc[features, "shared_fdr"] < fdr_cut
        ) * 1
        adata.var.loc[features, "spec_signi"] = (
            adata.var.loc[features, "spec_fdr"] < fdr_cut
        ) * 1

        logg.info(
            f"    found {sum(adata.var['shared_signi'])} shared and {sum(adata.var['spec_signi'])} specific features",
            time=True,
            end="\n",
        )
    else:
        res = "covtrend" if trend_test else "cov"
        adata.var[f"{res}_pval"] = np.nan
        adata.var[f"{res}_fdr"] = np.nan
        adata.var[f"{res}_signi"] = np.nan
        adata.var[f"{res}_A"] = np.nan

        pvals = np.array([s[0] for s in stat])
        pvals[np.isnan(pvals)] = 1.0

        adata.var.loc[features, f"{res}_pval"] = pvals
        adata.var.loc[features, f"{res}_A"] = [s[1] for s in stat]
        adata.var.loc[features, f"{res}_fdr"] = multipletests(
            adata.var.loc[features, f"{res}_pval"], method="bonferroni"
        )[1]
        adata.var.loc[features, f"{res}_signi"] = (
            adata.var.loc[features, f"{res}_fdr"] < fdr_cut
        ) * 1

        logg.info(
            "    found " + str(sum(adata.var[f"{res}_signi"])) + " significant features",
            time=True,
            end="\n",
        )

    adata.var["->".join(adata.obs[group_key].cat.categories) + "_lfc"] = np.nan
    adata.var.loc[features, "->".join(adata.obs[group_key].cat.categories) + "_lfc"] = [
        s[-2] for s in stat
    ]

    for i, g in enumerate(adata.obs[group_key].cat.categories):
        adata.var[f"{g}_lfc"] = np.nan
        adata.var.loc[features, f"{g}_lfc"] = [s[-1][i] for s in stat]

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    if nested:
        logg.hint(
            "added \n"
            "    .var['shared_pval'], pvalues for shared trend.\n"
            "    .var['shared_fdr'], FDR for shared trend.\n"
            "    .var['shared_signi'], is the shared trend significant.\n"
            "    .var['shared_A'], amplitude of the shared trend.\n"
            "    .var['spec_pval'], pvalues for specific trend.\n"
            "    .var['spec_fdr'], FDR for specific trend.\n"
            "    .var['spec_signi'], is the specific trend significant.\n"
            "    .var['{covariate}_lfc'], logfoldchange in expression between covariate and the rest of the cells."
        )
    else:
        covstr = "covtrend" if trend_test else "cov"
        logg.hint(
            "added \n"
            "    .var['" + covstr + "_pval'], pvalues extracted from tests.\n"
            "    .var['" + covstr + "_fdr'], FDR extracted from the pvalues.\n"
            "    .var['" + covstr + "_signi'], is the feature significant.\n"
            "    .var['" + covstr + "_A'], amplitude of the trend.\n"
            "    .var['" + "->".join(adata.obs[group_key].cat.categories) + "_lfc'],"
            " logfoldchange in expression between covariate "
            + " and ".join(adata.obs[group_key].cat.categories)
            + ".\n"
            "    .var['{covariate}_lfc'], logfoldchange in expression between covariate and the rest of the cells."
        )


def group_test(
    df, group, trend_test=False, logbase=None, return_pred=False, nested=False
):

    global rmgcv

    if not return_pred:
        categories = df[group].cat.categories
        means = [df.exp[df[group] == g].mean() for g in categories]
        
        # One-vs-rest LFCs
        ovr_lfcs = []
        for i, g in enumerate(categories):
            mean_g = means[i]
            mean_rest = df.exp[df[group] != g].mean()
            foldchange = (np.expm1(mean_g * logbase) + 1e-9) / (
                np.expm1(mean_rest * logbase) + 1e-9
            )
            ovr_lfcs.append(np.log2(foldchange))

        if len(means) == 2:
            foldchange = (np.expm1(means[0] * logbase) + 1e-9) / (
                np.expm1(means[1] * logbase) + 1e-9
            )
            lfc = np.log2(foldchange)
        else:
            lfc = np.nan

    # Keep a copy of groups for amplitude calculation
    groups_py = df[group].values
    first_group = df[group].cat.categories[0]
    mask = groups_py == first_group

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri as p2ri
    from rpy2.robjects.conversion import localconverter
    context = localconverter(ro.default_converter + p2ri.converter)


    with context as cv:
        df = cv.py2rpy(df)

    if nested:
        m_full = rmgcv.gam(
            Formula(
                f"exp ~ s(t,k=5)+s(t,by=as.factor({group}),k=5)+as.factor({group})"
            ),
            data=df,
        )
        m_shared = rmgcv.gam(Formula(f"exp ~ s(t,k=5)+as.factor({group})"), data=df)
        m_null = rmgcv.gam(Formula(f"exp ~ as.factor({group})"), data=df)

        test_spec = rmgcv.anova_gam(m_shared, m_full, test="F")
        test_shared = rmgcv.anova_gam(m_null, m_shared, test="F")

        with context as cv:
            test_spec_df = cv.rpy2py(test_spec)
            test_shared_df = cv.rpy2py(test_shared)

        try:
            pval_spec = test_spec_df.loc["2", "Pr(>F)"]
        except:
            pval_spec = np.nan
            
        try:
            pval_shared = test_shared_df.loc["2", "Pr(>F)"]
        except:
            pval_shared = np.nan

        if pd.isna(pval_spec):
            pval_spec = 1.0
        if pd.isna(pval_shared):
            pval_shared = 1.0

        pr_shared = rmgcv.predict_gam(m_shared)
        with context as cv:
            pr_shared = cv.rpy2py(pr_shared)
        amp_shared = np.max(pr_shared[mask]) - np.min(pr_shared[mask])

        return (pval_shared, pval_spec, amp_shared, lfc, ovr_lfcs)

    if trend_test:
        m1 = rmgcv.gam(
            Formula(
                f"exp ~ s(t,k=5)+s(t,by=as.factor({group}),k=5)+as.factor({group})"
            ),
            data=df,
        )
        m0 = rmgcv.gam(Formula(f"exp ~ s(t,k=5)+as.factor({group})"), data=df)
        if return_pred:
            pr1, pr2 = rmgcv.predict_gam(m1), rmgcv.predict_gam(m0)
            with context as cv:
                return cv.rpy2py(pr1),cv.rpy2py(pr2)
        else:
            test = rmgcv.anova_gam(m0, m1, test="F")
            with context as cv:
                test_df = cv.rpy2py(test)
            try:
                pval = test_df.loc["2", "Pr(>F)"]
            except:
                pval = np.nan
            if pd.isna(pval):
                pval = 1.0
            
            pr_shared = rmgcv.predict_gam(m0)
            with context as cv:
                pr_shared = cv.rpy2py(pr_shared)
            amp_shared = np.max(pr_shared[mask]) - np.min(pr_shared[mask])

            return (pval, amp_shared, lfc, ovr_lfcs)
    else:
        m = rmgcv.gam(
            Formula(
                f"exp ~ s(t,k=5)+s(t,by=as.factor({group}),k=5)+as.factor({group})"
            ),
            data=df,
        )
        if return_pred:
            pr1, pr2 = rmgcv.predict_gam(m), rmgcv.predict_gam(m)
            return cv.rpy2py(pr1),cv.rpy2py(pr2)
        else:
            try:
                pval = rmgcv.summary_gam(m)[3][1]
            except:
                pval = np.nan
            if pd.isna(pval):
                pval = 1.0
            
            pr_full = rmgcv.predict_gam(m)
            with context as cv:
                pr_full = cv.rpy2py(pr_full)
            # For amplitude test, we take the max amplitude among groups
            # Use categories saved before R conversion
            amps = [np.max(pr_full[groups_py == g]) - np.min(pr_full[groups_py == g]) for g in categories]
            amp = np.max(amps)

            return (pval, amp, lfc, ovr_lfcs)


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
