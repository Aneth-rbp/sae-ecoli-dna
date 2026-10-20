# Decisions

Analysis decisions, mirrored from issue #14. Each entry states what was decided and why.

## Settled

- **Main analysis uses the top-N genes per unit; activation > 0 is secondary.** A zero threshold on the baseline marks each dimension active in about half the genes, which makes the comparison with a sparse SAE unfair.
- **The SAE has 1,024 features and is trained on genomic windows.** About 3,000 genes are too few rows to train a wider dictionary; 1 kb windows with a 500 nt step add roughly 9,000 rows per strand.
- **Evo 2 7B in bf16 on the DGX Spark replaces Nucleotide Transformer v2 50M.** The Spark's GB10 GPU is not Hopper, so the FP8 checkpoints (40B, 20B, 1B) are out and the 7B runs without Transformer Engine; the Windows dev machine only downloads data and runs tests. Known risk: Evo 2 recommends torch 2.6 or 2.7, while the NGC 25.10 image that supports GB10 ships torch 2.9.

## Open (see #14)

- Evo 2 layer (decided in #3, which still targets Nucleotide Transformer and has to be re-scoped).
- N for top-N (proposed 50).
- k for TopK (16 or 32).
- GO term size filter (proposed 5-500 genes, propagated to ancestors).
- Benjamini-Hochberg scope (proposed per feature, global as sensitivity).
- Window strands (proposed both).
