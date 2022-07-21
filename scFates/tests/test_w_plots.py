import scFates as scf
import scanpy as sc
import numpy as np
import pandas as pd


def test_pipeline():
    adata = scf.datasets.test_adata()
    adata.layers["scaled"] = sc.pp.scale(adata.X, copy=True)

    scf.tl.curve(adata, Nodes=10, use_rep="pca", device="cpu", seed=1)
    F_PC1_epgc_cpu = adata.uns["graph"]["F"][0, :5]

    scf.tl.tree(adata, Nodes=10, use_rep="pca", method="epg", device="cpu", seed=1)
    F_PC1_epgt_cpu = adata.uns["graph"]["F"][0, :5]

    adata_circle = scf.tl.circle(
        adata, Nodes=10, use_rep="pca", device="cpu", seed=1, copy=True
    )
    scf.tl.convert_to_soft(adata_circle, 1, 100)
    scf.tl.root(adata_circle, 2)
    scf.tl.pseudotime(adata_circle, n_map=3)
    scf.tl.unroll_circle(adata_circle)

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
        ppt_lambda=200,
        seed=1,
    )

    scf.tl.cleanup(adata)

    F_PC1_ppt_cpu = adata.uns["graph"]["F"][0, :5]

    scf.pl.graph(adata)

    adata_2 = scf.tl.roots(
        adata, roots=[43, 18], meeting=adata.uns["graph"]["forks"][0], copy=True
    )
    scf.tl.root(adata, "n_counts")
    scf.tl.root(adata, "Phox2a", tips_only=True, min_val=True)
    scf.tl.root(adata, 43)
    pp_info_time = adata.uns["graph"]["pp_info"]["time"][:5].values

    scf.tl.pseudotime(adata_2)
    scf.tl.pseudotime(adata)
    scf.tl.dendrogram(adata)
    scf.pl.dendrogram(adata, color_milestones=True)

    adata_s = scf.tl.simplify(adata, copy=True)

    scf.pl.binned_pseudotime_meta(adata, "leiden", show_colorbar=True)

    obs_t = adata.obs.t[:5].values

    scf.pl.trajectory(adata, arrows=True)
    scf.pl.milestones(adata, annotate=True)
    scf.pl.graph(adata_2)
    scf.pl.trajectory(adata_2, color_seg="milestones", arrows=True)
    scf.pl.trajectory(adata_2, color_seg="seg", arrows=True)

    df = scf.tl.getpath(adata, root_milestone="43", milestones=["18"])

    adata.obsm["X_umap3d"] = np.concatenate(
        [adata.obsm["X_umap"], adata.obsm["X_umap"][:, 0].reshape(-1, 1)], axis=1
    )

    scf.pl.trajectory_3d(adata)
    scf.pl.trajectory_3d(adata, color="seg")

    adata_s1 = scf.tl.subset_tree(
        adata, root_milestone="88", milestones=["18"], mode="substract", copy=True
    )
    adata_s2 = scf.tl.subset_tree(
        adata, root_milestone="88", milestones=["18"], mode="extract", copy=True
    )
    adata_at = scf.tl.attach_tree(adata_s1, adata_s2)
    adata_at = scf.tl.attach_tree(adata_s1, adata_s2, linkage=("24", "88"))

    adata_lim = scf.tl.subset_tree(adata, t_max=adata.obs.t.max() / 4 * 3, copy=True)

    scf.tl.test_association(adata, n_jobs=2)
    scf.tl.test_association(adata, A_cut=0.3, reapply_filters=True)
    scf.tl.test_association(adata_2, layer="scaled", A_cut=0.3)
    nsigni = adata.var.signi.sum()
    A = adata.var.A[:5].values

    scf.pl.test_association(adata)

    scf.tl.linearity_deviation(adata, start_milestone="43", end_milestone="88")

    lindev = adata.var["43->88_rss"].values[:5]

    scf.pl.linearity_deviation(adata, start_milestone="43", end_milestone="88")

    scf.tl.fit(adata_2, layer="scaled")
    scf.tl.fit(adata)
    fitted = adata.layers["fitted"][0, :5]

    adata.obs["covariate"] = "A"
    cells = np.random.choice(
        adata.obs_names, size=int(adata.shape[0] / 2), replace=False
    )
    adata.obs.loc[cells, "covariate"] = "B"
    scf.tl.test_association_covariate(adata, "covariate")
    adata.var["signi"] = True
    scf.tl.test_covariate(adata, "covariate")
    scf.tl.test_covariate(adata, "covariate", trend_test=True)
    scf.pl.trend_covariate(
        adata, adata.var_names[0], group_key="covariate", show_null=True
    )

    scf.pl.matrix(adata, adata.var_names, annot_var=True)
    scf.pl.matrix(adata, adata.var_names, norm="minmax", return_data=True)
    scf.pl.matrix(adata, adata.var_names, root_milestone="88", milestones=["18"])

    scf.tl.rename_milestones(adata_2, ["A", "B", "C", "D"])
    scf.pl.trajectory(adata_2, root_milestone="A", milestones=["B"])

    scf.tl.cluster(adata, knn=3)

    g = scf.pl.trends(adata, features=adata.var_names, return_genes=True)
    scf.pl.trends(
        adata,
        features=adata.var_names,
        annot="milestones",
        plot_emb=False,
        ordering="quantile",
    )

    scf.pl.trends(
        adata,
        features=adata.var_names,
        annot="seg",
        root_milestone="43",
        milestones=["88"],
        ordering="pearson",
        show=False,
    )

    scf.pl.single_trend(adata, feature=adata.var_names[0], color_exp="k")

    scf.pl.single_trend(
        adata,
        layer="scaled",
        feature=adata.var_names[0],
        show=False,
    )

    scf.pl.dendrogram(
        adata, show_info=False, color="t", root_milestone="43", milestones=["24", "18"]
    )

    scf.tl.test_fork(
        adata, layer="scaled", root_milestone="43", milestones=["24", "18"], n_jobs=2
    )
    scf.tl.test_fork(adata, root_milestone="43", milestones=["24", "18"], n_jobs=2)
    signi_fdr_nonscaled = adata.uns["43->24<>18"]["fork"].signi_fdr.sum()
    adata.uns["43->24<>18"]["fork"]["fdr"] = 0.02
    scf.pl.test_fork(adata, root_milestone="43", milestones=["24", "18"])

    scf.tl.test_fork(
        adata, root_milestone="43", milestones=["24", "18"], n_jobs=2, rescale=True
    )
    signi_fdr_rescaled = adata.uns["43->24<>18"]["fork"].signi_fdr.sum()

    scf.tl.branch_specific(
        adata, root_milestone="43", milestones=["24", "18"], effect=0.6
    )

    branch_spe = scf.get.fork_stats(
        adata, root_milestone="43", milestones=["24", "18"]
    ).branch.values

    scf.tl.activation(adata, root_milestone="43", milestones=["24", "18"], n_jobs=1)
    activation = adata.uns["43->24<>18"]["fork"].activation.values

    scf.pl.modules(adata, root_milestone="43", milestones=["24", "18"])
    scf.pl.modules(adata, root_milestone="43", milestones=["24", "18"], show_traj=True)
    scf.pl.modules(adata, root_milestone="43", milestones=["24", "18"], module="early")
    scf.pl.modules(
        adata, root_milestone="43", milestones=["24", "18"], module="late", show=False
    )

    scf.tl.activation_lm(adata, root_milestone="43", milestones=["24", "18"], n_jobs=1)
    activation_lm = adata.uns["43->24<>18"]["fork"].slope.values
    adata.uns["43->24<>18"]["fork"]["module"] = "early"
    scf.tl.module_inclusion(adata, root_milestone="43", milestones=["24", "18"])
    mod_inc = adata.uns["43->24<>18"]["module_inclusion"]["18"]["0"].values
    scf.pl.module_inclusion(
        adata, root_milestone="43", milestones=["24", "18"], bins=12, branch="18"
    )

    scf.pl.single_trend(
        adata, root_milestone="43", milestones=["24", "18"], module="early", branch="24"
    )

    scf.tl.slide_cells(adata, root_milestone="43", milestones=["24", "18"], win=200)
    cell_freq_sum = adata.uns["43->24<>18"]["cell_freq"][0].sum()

    scf.tl.slide_cells(adata, root_milestone="43", milestones=["24"], win=200)
    adata.uns["43->24<>18"]["fork"].loc["Etv1", "module"] = "early"
    scf.tl.slide_cors(adata, root_milestone="43", milestones=["24", "18"])
    corAB = adata.uns["43->24<>18"]["corAB"]["18"]["genesetA"].iloc[0, :5].values

    scf.tl.slide_cors(
        adata,
        root_milestone="43",
        milestones=["24"],
        genesetA=adata.var_names[[0, 1]],
        genesetB=adata.var_names[[2, 3]],
    )

    scf.pl.slide_cors(adata, root_milestone="43", milestones=["24", "18"])

    adata.uns["43->24<>18"]["corAB"]["24"]["genesetA"] = pd.DataFrame(
        0.2,
        index=np.array(["a", "b", "c", "d"]),
        columns=adata.uns["43->24<>18"]["corAB"]["24"]["genesetA"].columns,
    )
    adata.uns["43->24<>18"]["corAB"]["24"]["genesetB"] = pd.DataFrame(
        -0.2,
        index=np.array(["a", "b", "c", "d"]),
        columns=adata.uns["43->24<>18"]["corAB"]["24"]["genesetA"].columns,
    )
    adata.uns["43->24<>18"]["corAB"]["18"]["genesetA"] = pd.DataFrame(
        -0.2,
        index=np.array(["a", "b", "c", "d"]),
        columns=adata.uns["43->24<>18"]["corAB"]["24"]["genesetA"].columns,
    )
    adata.uns["43->24<>18"]["corAB"]["18"]["genesetB"] = pd.DataFrame(
        0.2,
        index=np.array(["a", "b", "c", "d"]),
        columns=adata.uns["43->24<>18"]["corAB"]["24"]["genesetA"].columns,
    )

    scf.pl.slide_cors(
        adata, root_milestone="43", milestones=["24", "18"], win_keep=range(3), focus=1
    )

    scf.pl.slide_cors(adata, root_milestone="43", milestones=["24"])

    scf.tl.synchro_path(
        adata,
        root_milestone="43",
        milestones=["24", "18"],
        w=500,
        step=30,
    )

    syncAB = adata.uns["43->24<>18"]["synchro"]["real"]["24"]["corAB"].values[:5]

    adata = scf.datasets.test_adata(plot=True)

    scf.get.slide_cors(
        adata,
        root_milestone="Root",
        milestones=["DC", "Mono"],
        branch="DC",
        geneset_branch="DC",
    )

    scf.pl.trajectory(adata, color_seg="milestones")

    scf.pl.synchro_path(adata, root_milestone="Root", milestones=["DC", "Mono"])

    scf.pl.module_inclusion(
        adata, root_milestone="Root", milestones=["DC", "Mono"], bins=10, branch="Mono"
    )

    assert np.allclose(
        F_PC1_epgc_cpu,
        [9.27506488, -23.41654899, -14.26729458, -5.30612656, -20.13940938],
    )
    # assert np.allclose(F_PC1_epgt_cpu, F_PC1_epgt_gpu)
    assert np.allclose(
        F_PC1_epgt_cpu,
        [-10.81716492, 9.02788641, -14.80570284, -23.39084922, -20.12368624],
    )
    # assert np.allclose(F_PC1_ppt_cpu, F_PC1_ppt_gpu, rtol=1e-2)
    assert np.allclose(
        F_PC1_ppt_cpu,
        [-23.27292273, -12.22871375, 7.1627, -8.76348289, -14.22426111],
        rtol=1e-2,
    )
    assert np.allclose(
        pp_info_time,
        [63.58330042, 40.27148842, 3.17146368, 28.9625195, 38.6481921],
        rtol=1e-2,
    )
    assert np.allclose(
        obs_t,
        [63.8508307, 63.84462296, 63.84324671, 55.24862655, 58.64053484],
        rtol=1e-2,
    )
    assert df.shape[0] == 857
    assert np.allclose(
        A, [0.01253371, 0.23076978, 0.06073513, 0.11296414, 0.05083225], rtol=1e-2
    )
    assert np.allclose(
        lindev,
        [0.30906181, -2.71939561, 0.61090579, -1.1489122, 0.12358796],
        rtol=1e-2,
    )
    assert nsigni == 6
    assert np.allclose(
        fitted,
        [
            2.94194510e-04,
            4.63294288e-01,
            6.39626984e-01,
            5.60286879e-01,
            9.50514575e-01,
        ],
        rtol=1e-2,
    )
    assert signi_fdr_nonscaled == 5
    assert signi_fdr_rescaled == 5
    assert np.all(branch_spe == ["18", "18", "24"])
    assert np.allclose(mod_inc, [7.6982633, 7.6982633], rtol=1e-2)
    assert np.allclose(activation, [24.64678374, 24.64678374, 42.29117514], rtol=1e-2)
    assert np.allclose(activation_lm, [-0.01767213, -0.00753814, 0.02148092], rtol=1e-2)
    assert np.allclose(cell_freq_sum, 197.02803472363118, rtol=1e-2)
    assert np.allclose(
        corAB,
        [-0.08104039, -0.2667385, -0.33143075, -0.05329868, -0.01551609],
        rtol=1e-2,
    )
    assert np.allclose(
        syncAB,
        [-0.31524637, -0.31332134, -0.3392098, -0.33832952, -0.31277207],
        rtol=1e-2,
    )
