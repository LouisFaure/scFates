from anndata import AnnData


def test_association(adata: AnnData, log_A: bool = False):
    """\
    Plot a set of fitted features over pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix.
    log_A
        change the xaxis scale to log.

    Returns
    -------
    just the plot.

    """

    stats = adata.var.copy(deep=True)
    # correct for zeros but setting them to the lowest value
    stats.loc[stats.fdr == 0, "fdr"] = stats.fdr[stats.fdr != 0].min()
    colors = ["red" if signi else "grey" for signi in stats.signi.values]
    stats.plot.scatter(x="A", y="fdr", color=colors, logx=log_A, logy=True)
