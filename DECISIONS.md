# Decisions

Analysis decisions, mirrored from issue #14. Each entry states what was decided and why.

## Settled

- **Main analysis uses the top-N genes per unit; activation > 0 is secondary.** A zero threshold on the baseline marks each dimension active in about half the genes, which makes the comparison with a sparse SAE unfair.
- **The SAE has 1,024 features and is trained on genomic windows.** About 3,000 genes are too few rows to train a wider dictionary; 1 kb windows with a 500 nt step add roughly 9,000 rows per strand.
- **Evo 2 7B in bf16 on the DGX Spark replaces Nucleotide Transformer v2 50M.** The Spark's GB10 GPU is not Hopper, so the FP8 checkpoints (40B, 20B, 1B) are out and the 7B runs without Transformer Engine; the Windows dev machine only downloads data and runs tests. Known risk: Evo 2 recommends torch 2.6 or 2.7, while the NGC 25.10 image that supports GB10 ships torch 2.9.
- **N for top-N is 50.** About 1.5 % of the roughly 3,000 genes keeps each unit's gene set specific while leaving enough genes for the hypergeometric test against GO terms.
- **GO terms with 5 to 500 genes, with annotations propagated to ancestors.** Very small terms give enrichments driven by one or two genes and very large ones are too generic to interpret; propagation lets a gene annotated to a specific term count toward its parents.
- **Benjamini-Hochberg is applied per feature, over that feature's terms; global BH is a sensitivity analysis.** Per-feature correction follows Gujral et al. 2025, which keeps the comparison with the base article direct; correcting across all features and terms at once is stricter and is reported to show how much the result depends on that choice.
- **Genomic windows are embedded on both strands.** Taking the forward strand and the reverse complement of every window matches the sense orientation of genes on either strand; it roughly doubles the window rows (about 18,000) and the embedding compute on the Spark.

## Open (see #14)

- Evo 2 layer (decided in #3, which still targets Nucleotide Transformer and has to be re-scoped).
- k for TopK (16 or 32; decided in #5 as the smaller one with explained variance >= 0.8 and fewer than 20 % dead features).
- Third reference (decided in #11: Crook et al. (Decode-gLM) if its authors, version and DOI can be verified; otherwise Dalla-Torre et al., Nature Methods 22, 287-297, 2025).
