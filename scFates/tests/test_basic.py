import scFates as scf
import scanpy as sc
import numpy as np
from mock import Mock

def fake_palantir(adata,seg,ms_data):
    mini=adata.obs.t.loc[adata.obs.seg==seg].min()
    maxi=adata.obs.t.loc[adata.obs.seg==seg].max()
    return (adata.obs.t.loc[adata.obs.seg==seg]-mini)/(maxi-mini)

def test_pipeline():
    adata=scf.datasets.test_adata()
    adata.layers["scaled"]=sc.pp.scale(adata.X,copy=True)

    scf.tl.curve(adata,Nodes=10,use_rep="pca",device="cpu",seed=1)
    F_PC1_epgc_cpu = adata.uns['graph']['F'][0,:5]

    #scf.tl.curve(adata,Nodes=10,use_rep="pca",device="gpu",seed=1)
    #F_PC1_epgc_gpu = adata.uns['graph']['F'][0,:5]

    scf.tl.tree(adata,Nodes=10,use_rep="pca",method="epg",device="cpu",seed=1)
    F_PC1_epgt_cpu = adata.uns['graph']['F'][0,:5]

    #scf.tl.tree(adata,Nodes=10,use_rep="pca",method="epg",device="gpu",seed=1)
    #F_PC1_epgt_gpu = adata.uns['graph']['F'][0,:5]


    scf.tl.tree(adata,Nodes=100,use_rep="pca",method="ppt",device="cpu",ppt_sigma=1,ppt_lambda=1,seed=1)
    scf.tl.cleanup(adata,leaves=[adata.uns["graph"]['tips'][i] for i in range(1,4)],minbranchlength=10)

    scf.tl.tree(adata,Nodes=100,use_rep="pca",method="ppt",device="cpu",ppt_sigma=1,ppt_lambda=10000,seed=1)
    F_PC1_ppt_cpu = adata.uns['graph']['F'][0,:5]

    #scf.tl.tree(adata,Nodes=100,use_rep="pca",method="ppt",device="gpu",ppt_sigma=1,ppt_lambda=10000,seed=1)
    #F_PC1_ppt_gpu = adata.uns['graph']['F'][0,:5]

    scf.pl.trajectory(adata,emb_back=adata.obsm["X_umap"])

    adata_2=scf.tl.roots(adata,roots=[80,25],meeting=adata.uns['graph']["forks"][0],copy=True)
    scf.tl.root(adata,80)
    pp_info_time=adata.uns["graph"]["pp_info"]["time"][:5].values

    scf.tl.pseudotime(adata_2,n_map=2)
    scf.tl.pseudotime(adata_2)
    scf.tl.pseudotime(adata)
    
    scf.tools.utils.palantir_on_seg=Mock(side_effect=fake_palantir)
    scf.tl.refine_pseudotime(adata_2,ms_data="diffusion")
    adata_temp=scf.tl.refine_pseudotime(adata,ms_data="diffusion",copy=True)
    
    obs_t=adata.obs.t[:5].values

    scf.pl.trajectory_pseudotime(adata,emb_back=adata.obsm["X_umap"],arrows=True)
    scf.pl.milestones(adata)
    
    df = scf.tl.getpath(adata,root_milestone='80',milestones=['19'])
    
    adata.obsm["X_umap3d"]=np.concatenate([adata.obsm["X_umap"],adata.obsm["X_umap"][:,0].reshape(-1,1)],axis=1)
    scf.pl.trajectory_3d(adata)
    scf.pl.trajectory_3d(adata,color="seg")

    scf.tl.test_association(adata,n_jobs=2)
    scf.tl.test_association(adata,A_cut=.3,reapply_filters=True)
    scf.tl.test_association(adata_2,layer="scaled",A_cut=.3)
    nsigni=adata.var.signi.sum()
    A=adata.var.A[:5].values


    scf.pl.test_association(adata)

    scf.tl.fit(adata_2,layer="scaled")
    scf.tl.fit(adata)
    fitted=adata.layers["fitted"][0,:5]
    
    scf.tl.test_association(adata_2,root='80',leaves=['19'])
    scf.tl.fit(adata_2,root='80',leaves=['19'])
    
    scf.tl.cluster(adata,knn=3)
    
    scf.pl.single_trend(adata,feature=adata.var_names[0])
    scf.pl.trends(adata,features=adata.var_names)
    scf.pl.cluster(adata,features=adata.var_names)
    
    scf.pl.cluster(adata,features=adata.var_names,root_milestone='80',milestones=['19'])
    scf.pl.trends(adata,features=adata.var_names,root_milestone='80',milestones=['19'])

    scf.tl.test_fork(adata,layer="scaled",root_milestone='80',milestones=['25','19'],n_jobs=2)
    scf.tl.test_fork(adata,root_milestone='80',milestones=['25','19'],n_jobs=2)
    signi_fdr_nonscaled = adata.uns['80->25<>19']['fork'].signi_fdr.sum()

    scf.tl.test_fork(adata,root_milestone='80',milestones=['25','19'],n_jobs=2,rescale=True)
    signi_fdr_rescaled = adata.uns['80->25<>19']['fork'].signi_fdr.sum()

    scf.tl.branch_specific(adata,root_milestone='80',milestones=['25','19'],effect=0.6)
    branch_spe=adata.uns['80->25<>19']['fork'].branch.values

    scf.tl.activation(adata,root_milestone='80',milestones=['25','19'],n_jobs=1)
    activation = adata.uns['80->25<>19']['fork'].activation.values
    
    scf.pl.modules(adata,root_milestone='80',milestones=['25','19'], highlight = True)
    scf.pl.modules(adata,root_milestone='80',milestones=['25','19'], mode="3d")

    scf.tl.slide_cells(adata,root_milestone='80',milestones=['25','19'],win=200)
    cell_freq_sum = adata.uns['80->25<>19']["cell_freq"][0].sum()
    
    scf.tl.slide_cells(adata,root_milestone='80',milestones=['25'],win=200)

    scf.tl.slide_cors(adata,root_milestone='80',milestones=['25','19'])
    corAB = adata.uns['80->25<>19']['corAB'].loc["19"].iloc[0,:5].values
    
    scf.tl.slide_cors(adata,root_milestone='80',milestones=['25'],genesetA=adata.var_names[[0,1]],genesetB=adata.var_names[[2,3]])
    
    scf.pl.slide_cors(adata,root_milestone='80',milestones=['25','19'])
    scf.pl.slide_cors(adata,root_milestone='80',milestones=['25'],genesetA=adata.var_names[[0,1]],genesetB=adata.var_names[[2,3]])
    
    scf.tl.synchro_path(adata,root_milestone='80',milestones=['25','19'],w=500,step=30,loess_span=0.5)
    scf.pl.synchro_path(adata,root_milestone='80',milestones=['25','19'],loess_span=0.5)
    syncAB=adata.uns["80->25<>19"]["synchro"]["corAB"].values[:5]

    assert np.allclose(F_PC1_epgc_cpu,[7.58077519, -23.63881596, -13.71322111, -5.53347147, -13.44127461])
    #assert np.allclose(F_PC1_epgt_cpu, F_PC1_epgt_gpu)
    assert np.allclose(F_PC1_epgt_cpu,[-2.52311629, -22.52998568, 9.00639977, -14.63753648, -15.91888575])
    #assert np.allclose(F_PC1_ppt_cpu, F_PC1_ppt_gpu, rtol=1e-2)
    assert np.allclose(F_PC1_ppt_cpu, [-14.80722458, -11.65633904,  -3.38224082,  -9.81132852, -10.80313121], rtol=1e-2)
    assert np.allclose(pp_info_time,[18.80301214, 12.79578319,  0.64553512,  9.62211211, 12.26296244], rtol=1e-2)
    assert np.allclose(obs_t,[18.80780351, 18.83173969, 18.81651117, 18.83505737, 18.82784781], rtol=1e-2)
    assert df.shape[0]==877
    assert np.allclose(A,[0.01329398, 0.17759706, 0.1007363 , 0.14490752, 0.07136825], rtol=1e-2)
    assert nsigni == 5
    assert np.allclose(fitted, [0.44351696, 0.68493632, 0.24545846, 0.16027536, 0.79252075], rtol=1e-2)
    assert signi_fdr_nonscaled == 0
    assert signi_fdr_rescaled == 5
    assert np.all(branch_spe == ['19','19','25'])
    assert np.allclose(activation, [3.95669504, 3.95669504, 0.6351621 ], rtol=1e-2)
    assert np.allclose(cell_freq_sum, 187.5316, rtol=1e-2)
    assert np.allclose(corAB, [-0.25072212, -0.2963759, -0.46956663, -0.15842558, -0.01394084], rtol=1e-2)
    assert np.allclose(syncAB,[-0.33520381, -0.32288241, -0.33093774, -0.33303708, -0.35310308], rtol=1e-2)