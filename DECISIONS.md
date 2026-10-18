# Decisions

Analysis decisions, mirrored from issue #14. Each entry states what was decided and why.

## Settled

- **Main analysis uses the top-N genes per unit; activation > 0 is secondary.** A zero threshold on the baseline marks each dimension active in about half the genes, which makes the comparison with a sparse SAE unfair.
- **The SAE has 1,024 features and is trained on genomic windows.** About 3,000 genes are too few rows to train a wider dictionary; 1 kb windows with a 500 nt step add roughly 9,000 rows per strand.

## Open (see #14)

- Model layer (decided in #3).
- N for top-N (proposed 50).
- k for TopK (16 or 32).
- GO term size filter (proposed 5-500 genes, propagated to ancestors).
- Benjamini-Hochberg scope (proposed per feature, global as sensitivity).
- Window strands (proposed both).
