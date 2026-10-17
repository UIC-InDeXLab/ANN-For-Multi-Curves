# Third-party resources

The NBA SportVU tracking and event input is obtained from
https://github.com/sealneaward/nba-movement-data at the revision pinned in
`configs/nba-full-20261012.json`. The repository does not include the raw corpus.
Consult the source and relevant rights holders before downloading or reusing it.

The basketball GIF consists of generated court drawings and tracked paths.
No broadcaster video frames, still photographs, or audio are distributed here.
The NBA name, team names, and underlying data remain associated with their
respective rights holders; inclusion is not an endorsement.

Dependencies are installed from their own distributions and retain their own
licenses. In particular, the per-curve graph baseline uses `hnswlib`, while the
numerical kernels use NumPy and Numba and plotting uses Matplotlib and Pillow.
No third-party library source or pretrained model is bundled.

The grid-hashing construction follows the method of Anne Driemel and Francesco
Silvestri, "Locality-Sensitive Hashing of Curves" (2017). The per-curve HNSW
baseline follows the graph indexing implementation distributed as `hnswlib`.
