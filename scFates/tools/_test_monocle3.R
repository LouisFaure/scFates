suppressMessages(library(monocle3))
test_monocle3 <- function(X,genes,cells,UMAP,pr_graph_cell_proj_closest_vertex,F,B,n_jobs,...){
    rownames(X)=cells
    colnames(X)=genes
    cds = new_cell_data_set(t(X))
    cds@int_colData@listData$reducedDims@listData$UMAP = UMAP

    cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex=as.matrix(pr_graph_cell_proj_closest_vertex)
    cds@principal_graph_aux$UMAP$dp_mst=as.matrix(F)
    cds@principal_graph_aux$UMAP$stree=B

    colnames(B)=colnames(cds@principal_graph_aux$UMAP$dp_mst)
    cds@principal_graph[["UMAP"]]=igraph::graph_from_adjacency_matrix(B)
    pr_graph_test_res <- graph_test(cds, cores=n_jobs,...)
    return(pr_graph_test_res)
}
