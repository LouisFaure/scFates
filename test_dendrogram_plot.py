import scFates as scf
import scanpy as sc
import matplotlib.pyplot as plt
import os

# 1. Load test data
print("Loading test dataset...")
adata = scf.datasets.test_adata()

# 2. Basic processing (following test_w_plots.py pipeline)
print("Processing data (tree inference, root, pseudotime)...")
scf.tl.tree(adata, Nodes=100, use_rep="pca", method="ppt", device="cpu", ppt_sigma=1, ppt_lambda=1, seed=1)
scf.tl.root(adata, 0) # Set a root
scf.tl.pseudotime(adata)

# 3. Generate dendrogram data
print("Generating dendrogram embedding...")
scf.tl.dendrogram(adata)

# 4. Plot and save
print("Plotting dendrogram...")
# Ensure figures directory exists or use a local path
save_path = "dendrogram_test.png"

scf.pl.dendrogram(adata, color_milestones=True, show=False, save="_test.png")
scf.tl.merge_empty_segments(adata)
scf.tl.dendrogram(adata)
scf.pl.dendrogram(adata, color_milestones=True, show=False, save="_test_merged.png")

