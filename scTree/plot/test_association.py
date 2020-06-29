import pandas as pd
from anndata import AnnData

def test_association(
    adata: AnnData,
    log_A: bool = False):
    
    stats=adata.var.copy(deep=True)
    # correct for zeros but setting them to the lowest value
    stats.loc[stats.fdr==0,"fdr"] = stats.fdr[stats.fdr!=0].min()
    colors = ['red' if signi else 'grey' for signi in stats.signi.values]
    stats.plot.scatter(x="A",y="fdr",color=colors,logx=log_A,logy=True)
