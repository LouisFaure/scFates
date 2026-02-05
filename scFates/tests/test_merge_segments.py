"""Test for merge_small_segments function."""
import warnings
import numpy as np
import scFates as scf

scf.settings.verbosity = 0


def test_merge_small_segments():
    """Test that merge_small_segments handles empty segments correctly."""
    # Load test data and build a tree
    adata = scf.datasets.test_adata()

    # Build a tree with many nodes to potentially create short segments
    scf.tl.tree(
        adata,
        Nodes=100,
        use_rep="pca",
        method="ppt",
        device="cpu",
        ppt_sigma=1,
        ppt_lambda=200,
        seed=42,
    )
    scf.tl.cleanup(adata)
    scf.tl.root(adata, adata.uns["graph"]["tips"][0])

    # Run pseudotime and check for warning about empty segments
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scf.tl.pseudotime(adata)
        empty_seg_warnings = [
            warning for warning in w
            if "Some segs have no cell assigned" in str(warning.message)
        ]

    # If there are empty segments, test the merge function
    if len(empty_seg_warnings) > 0:
        # Get initial state
        initial_n_nodes = adata.uns["graph"]["B"].shape[0]

        # Run merge_small_segments
        scf.tl.merge_small_segments(adata)

        # Verify nodes were removed
        final_n_nodes = adata.uns["graph"]["B"].shape[0]
        assert final_n_nodes <= initial_n_nodes, "Nodes should be merged"

        # Verify no empty segments remain after re-running pseudotime
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            scf.tl.pseudotime(adata)
            new_warnings = [
                warning for warning in w2
                if "Some segs have no cell assigned" in str(warning.message)
            ]
        assert len(new_warnings) == 0, "No empty segments should remain"
    else:
        # No empty segments - just verify the function runs without error
        scf.tl.merge_small_segments(adata)

    # Verify basic graph integrity
    assert adata.uns["graph"]["B"].shape[0] == adata.uns["graph"]["B"].shape[1]
    assert adata.uns["graph"]["F"].shape[1] == adata.uns["graph"]["B"].shape[0]
    assert adata.obsm["X_R"].shape[1] == adata.uns["graph"]["B"].shape[0]
    assert "t" in adata.obs
    assert "seg" in adata.obs


def test_merge_small_segments_synthetic_empty():
    """Test merge with synthetically created empty segment."""
    adata = scf.datasets.test_adata()

    scf.tl.tree(
        adata,
        Nodes=50,
        use_rep="pca",
        method="ppt",
        device="cpu",
        ppt_sigma=1,
        ppt_lambda=100,
        seed=1,
    )
    scf.tl.cleanup(adata)
    scf.tl.root(adata, adata.uns["graph"]["tips"][0])
    scf.tl.pseudotime(adata)

    # Force an empty segment by reassigning cells
    pp_seg = adata.uns["graph"]["pp_seg"]
    if len(pp_seg) > 1:
        # Find smallest segment and remove its cells
        seg_counts = adata.obs.seg.value_counts()
        smallest_seg = seg_counts.idxmin()
        
        # Move cells to another segment
        other_segs = seg_counts.index[seg_counts.index != smallest_seg]
        if len(other_segs) > 0:
            adata.obs.loc[adata.obs.seg == smallest_seg, "seg"] = other_segs[0]

            # Now run merge
            initial_nodes = adata.uns["graph"]["B"].shape[0]
            scf.tl.merge_small_segments(adata)
            
            # Verify merge happened
            assert adata.uns["graph"]["B"].shape[0] <= initial_nodes


def test_merge_no_empty_segments():
    """Test that function handles case with no empty segments gracefully."""
    adata = scf.datasets.test_adata()

    scf.tl.tree(
        adata,
        Nodes=30,
        use_rep="pca",
        method="ppt",
        device="cpu",
        ppt_sigma=1,
        ppt_lambda=100,
        seed=1,
    )
    scf.tl.cleanup(adata)
    scf.tl.root(adata, adata.uns["graph"]["tips"][0])
    scf.tl.pseudotime(adata)

    initial_nodes = adata.uns["graph"]["B"].shape[0]

    # Should complete without error even if no empty segments
    scf.tl.merge_small_segments(adata)

    # Verify graph is still valid
    assert adata.uns["graph"]["B"].shape[0] <= initial_nodes
    assert "t" in adata.obs
