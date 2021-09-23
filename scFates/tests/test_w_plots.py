import scFates as scf
import scanpy as sc
import numpy as np


def test_pipeline():
    adata = scf.datasets.test_adata()
    adata.layers["scaled"] = sc.pp.scale(adata.X, copy=True)

    scf.tl.curve(adata, Nodes=10, use_rep="pca", device="cpu", seed=1)
    F_PC1_epgc_cpu = adata.uns["graph"]["F"][0, :5]

    # scf.tl.curve(adata,Nodes=10,use_rep="pca",device="gpu",seed=1)
    # F_PC1_epgc_gpu = adata.uns['graph']['F'][0,:5]

    scf.tl.tree(adata, Nodes=10, use_rep="pca", method="epg", device="cpu", seed=1)
    F_PC1_epgt_cpu = adata.uns["graph"]["F"][0, :5]

    scf.tl.circle(adata, Nodes=10, use_rep="pca", device="cpu", seed=1)

    # scf.tl.tree(adata,Nodes=10,use_rep="pca",method="epg",device="gpu",seed=1)
    # F_PC1_epgt_gpu = adata.uns['graph']['F'][0,:5]

    scf.tl.tree(
        adata,
        Nodes=100,
        use_rep="pca",
        method="ppt",
        device="cpu",
        ppt_sigma=1,
        ppt_lambda=1,
        seed=1,
    )
    scf.tl.cleanup(
        adata,
        leaves=[adata.uns["graph"]["tips"][i] for i in range(1, 4)],
        minbranchlength=10,
    )

    scf.tl.tree(
        adata,
        Nodes=100,
        use_rep="pca",
        method="ppt",
        device="cpu",
        ppt_sigma=1,
        ppt_lambda=10000,
        seed=1,
    )
    F_PC1_ppt_cpu = adata.uns["graph"]["F"][0, :5]

    scf.pl.graph(adata)

    adata_2 = scf.tl.roots(
        adata, roots=[80, 25], meeting=adata.uns["graph"]["forks"][0], copy=True
    )
    scf.tl.root(adata, "n_counts")
    scf.tl.root(adata, "Phox2a", tips_only=True, min_val=True)
    scf.tl.root(adata, 80)
    pp_info_time = adata.uns["graph"]["pp_info"]["time"][:5].values

    scf.tl.pseudotime(adata_2, n_map=2)
    scf.tl.pseudotime(adata_2)
    scf.tl.pseudotime(adata)
    scf.tl.dendrogram(adata)

    scf.pl.binned_pseudotime_meta(adata, "leiden", show_colorbar=True)

    adata_ext = scf.tl.extend_tips(adata, copy=True)

    obs_t = adata.obs.t[:5].values

    scf.pl.trajectory(adata, arrows=True)
    scf.pl.milestones(adata)
    scf.pl.milestones(adata, color="t")

    df = scf.tl.getpath(adata, root_milestone="80", milestones=["19"])

    adata.obsm["X_umap3d"] = np.concatenate(
        [adata.obsm["X_umap"], adata.obsm["X_umap"][:, 0].reshape(-1, 1)], axis=1
    )
    scf.pl.trajectory_3d(adata)
    scf.pl.trajectory_3d(adata, color="seg")

    adata_s1 = scf.tl.subset_tree(
        adata, root_milestone="29", milestones=["19"], mode="substract", copy=True
    )
    adata_s2 = scf.tl.subset_tree(
        adata, root_milestone="29", milestones=["19"], mode="extract", copy=True
    )
    adata_at = scf.tl.attach_tree(adata_s1, adata_s2)
    adata_at = scf.tl.attach_tree(adata_s1, adata_s2, linkage=("25", "19"))

    scf.tl.test_association(adata, n_jobs=2)
    scf.tl.test_association(adata, A_cut=0.3, reapply_filters=True)
    scf.tl.test_association(adata_2, layer="scaled", A_cut=0.3)
    nsigni = adata.var.signi.sum()
    A = adata.var.A[:5].values

    scf.pl.test_association(adata)

    scf.tl.linearity_deviation(adata, start_milestone="80", end_milestone="29")

    lindev = adata.var["80->29_rss"].values[:5]

    scf.pl.linearity_deviation(adata, start_milestone="80", end_milestone="29")

    scf.tl.fit(adata_2, layer="scaled")
    scf.tl.fit(adata)
    fitted = adata.layers["fitted"][0, :5]

    scf.tl.test_association(adata_2, root="80", leaves=["19"])
    scf.tl.fit(adata_2, root="80", leaves=["19"])

    scf.tl.rename_milestones(adata_2, ["A", "B", "C", "D"])
    scf.pl.trajectory(adata_2, root_milestone="A", milestones=["B"])

    scf.tl.cluster(adata, knn=3)

    scf.pl.trends(adata, features=adata.var_names, save_genes="genes.tsv")
    scf.pl.trends(
        adata,
        features=adata.var_names,
        annot="milestones",
        plot_emb=False,
        ordering="quantile",
    )
    del adata.uns["milestones_colors"]
    scf.pl.trends(
        adata,
        features=adata.var_names,
        annot="seg",
        root_milestone="80",
        milestones=["19"],
        ordering="pearson",
        show=False,
    )

    scf.pl.single_trend(adata, feature=adata.var_names[0], color_exp="k")
    del adata.uns["seg_colors"]
    scf.pl.single_trend(
        adata,
        layer="scaled",
        feature=adata.var_names[0],
        show=False,
    )

    scf.pl.dendrogram(
        adata, show_info=False, color="t", root_milestone="80", milestones=["25", "19"]
    )

    scf.tl.test_fork(
        adata, layer="scaled", root_milestone="80", milestones=["25", "19"], n_jobs=2
    )
    scf.tl.test_fork(adata, root_milestone="80", milestones=["25", "19"], n_jobs=2)
    signi_fdr_nonscaled = adata.uns["80->25<>19"]["fork"].signi_fdr.sum()
    adata.uns["80->25<>19"]["fork"]["fdr"] = 0.02
    scf.pl.test_fork(adata, root_milestone="80", milestones=["25", "19"])

    scf.tl.test_fork(
        adata, root_milestone="80", milestones=["25", "19"], n_jobs=2, rescale=True
    )
    signi_fdr_rescaled = adata.uns["80->25<>19"]["fork"].signi_fdr.sum()

    scf.tl.branch_specific(
        adata, root_milestone="80", milestones=["25", "19"], effect=0.6
    )
    branch_spe = adata.uns["80->25<>19"]["fork"].branch.values

    scf.tl.module_inclusion(adata, root_milestone="80", milestones=["25", "19"])
    mod_inc = adata.uns["80->25<>19"]["module_inclusion"]["19"]["0"].values
    scf.pl.module_inclusion(
        adata, root_milestone="80", milestones=["25", "19"], bins=12, branch="19"
    )

    scf.tl.activation(adata, root_milestone="80", milestones=["25", "19"], n_jobs=1)
    activation = adata.uns["80->25<>19"]["fork"].activation.values

    scf.tl.activation_lm(adata, root_milestone="80", milestones=["25", "19"], n_jobs=1)
    activation_lm = adata.uns["80->25<>19"]["fork"].slope.values

    scf.pl.modules(adata, root_milestone="80", milestones=["25", "19"])
    scf.pl.modules(adata, root_milestone="80", milestones=["25", "19"], show_traj=True)

    scf.pl.single_trend(
        adata, root_milestone="80", milestones=["25", "19"], module="early", branch="25"
    )

    scf.tl.slide_cells(adata, root_milestone="80", milestones=["25", "19"], win=200)
    cell_freq_sum = adata.uns["80->25<>19"]["cell_freq"][0].sum()

    scf.tl.slide_cells(adata, root_milestone="80", milestones=["25"], win=200)
    adata.uns["80->25<>19"]["fork"].loc["Etv1", "module"] = "early"
    scf.tl.slide_cors(adata, root_milestone="80", milestones=["25", "19"])
    corAB = adata.uns["80->25<>19"]["corAB"]["19"]["genesetA"].iloc[0, :5].values

    scf.tl.slide_cors(
        adata,
        root_milestone="80",
        milestones=["25"],
        genesetA=adata.var_names[[0, 1]],
        genesetB=adata.var_names[[2, 3]],
    )

    scf.pl.slide_cors(adata, root_milestone="80", milestones=["25", "19"])
    scf.pl.slide_cors(adata, root_milestone="80", milestones=["25"])

    scf.tl.synchro_path(
        adata,
        root_milestone="80",
        milestones=["25", "19"],
        w=500,
        step=30,
        loess_span=0.5,
    )
    scf.pl.synchro_path(
        adata, root_milestone="80", milestones=["25", "19"], loess_span=0.5
    )
    syncAB = adata.uns["80->25<>19"]["synchro"]["real"]["25"]["corAB"].values[:5]

    scf.tl.critical_transition(
        adata,
        root_milestone="80",
        milestones=["25", "19"],
        w=50,
        step=30,
        loess_span=0.5,
    )

    CI_lowess = adata.uns["80->25<>19"]["critical transition"]["LOESS"]["25"]["lowess"][
        :5
    ].values

    scf.pl.critical_transition(adata, root_milestone="80", milestones=["25", "19"])

    scf.tl.criticality_drivers(adata, root_milestone="80", milestones=["25", "19"])

    CI_corr = adata.uns["80->25<>19"]["criticality drivers"]["corr"].values

    assert np.allclose(
        F_PC1_epgc_cpu,
        [7.58077519, -23.63881596, -13.71322111, -5.53347147, -13.44127461],
    )
    # assert np.allclose(F_PC1_epgt_cpu, F_PC1_epgt_gpu)
    assert np.allclose(
        F_PC1_epgt_cpu,
        [-2.52311629, -22.52998568, 9.00639977, -14.63753648, -15.91888575],
    )
    # assert np.allclose(F_PC1_ppt_cpu, F_PC1_ppt_gpu, rtol=1e-2)
    assert np.allclose(
        F_PC1_ppt_cpu,
        [-14.80722458, -11.65633904, -3.38224082, -9.81132852, -10.80313121],
        rtol=1e-2,
    )
    assert np.allclose(
        pp_info_time,
        [18.80301214, 12.79578319, 0.64553512, 9.62211211, 12.26296244],
        rtol=1e-2,
    )
    assert np.allclose(
        obs_t,
        [18.80780351, 18.83173969, 18.81651117, 18.83505737, 18.82784781],
        rtol=1e-2,
    )
    assert df.shape[0] == 877
    assert np.allclose(
        A, [0.01329398, 0.17759706, 0.1007363, 0.14490752, 0.07136825], rtol=1e-2
    )
    assert np.allclose(
        lindev,
        [0.31511654, -0.94609691, 0.16721359, -0.53114662, 0.28442143],
        rtol=1e-2,
    )
    assert nsigni == 5
    assert np.allclose(
        fitted, [0.44351696, 0.68493632, 0.24545846, 0.16027536, 0.79252075], rtol=1e-2
    )
    assert signi_fdr_nonscaled == 0
    assert signi_fdr_rescaled == 5
    assert np.all(branch_spe == ["19", "19", "25"])
    assert np.allclose(mod_inc, [0.01487196, 0.01487196], rtol=1e-2)
    assert np.allclose(activation, [4.42785624, 4.61626781, 0.31774091], rtol=1e-2)
    assert np.allclose(activation_lm, [0.03350972, 0.01856113, 0.05883641], rtol=1e-2)
    assert np.allclose(cell_freq_sum, 187.5316, rtol=1e-2)
    assert np.allclose(
        corAB,
        [-0.19718103, -0.25194172, -0.47820323, -0.24967024, -0.06222028],
        rtol=1e-2,
    )
    assert np.allclose(
        syncAB,
        [-0.31113123, -0.28931709, -0.30131992, -0.31664843, -0.34292551],
        rtol=1e-2,
    )

    assert np.allclose(
        CI_lowess,
        [0.67226054, 0.67321935, 0.67406923, 0.67481306, 0.67568304],
        rtol=1e-2,
    )

    assert np.allclose(
        CI_corr,
        [-0.01154921, -0.03707902, -0.09560204, -0.20573824, -0.43346539],
        rtol=1e-2,
    )
