# 5. Data

All inputs are public and were downloaded on 2026-10-18 by `src.FetchData`, which records URL, version, SHA-256 and date of each file in `data/raw/manifest.json`:

- **Genome:** *Escherichia coli* K-12 MG1655, NCBI RefSeq GCF_000005845.2 (ASM584v2), sequence (FASTA) and annotation (GFF).
- **Function:** UniProt reference proteome UP000000625, release 2026_03 (accession, ordered locus name and GO terms).
- **Ontology:** `go-basic.obo`, Gene Ontology release 2026-07-26.

Genes are joined to UniProt by b-number: 4,289 of the 4,290 non-pseudo CDS match (99.98 %). GO annotations are propagated to their ancestors through `is_a` and `part_of`, and terms are kept when they annotate between 5 and 500 genes, which leaves 3,583 genes and 1,947 terms. Each gene enters Evo 2 as the first 1,000 nt of its CDS. Because about 3,500 genes are too few to train a dictionary, the SAE is trained on 18,564 genomic windows of 1 kb with a 500 nt step on both strands.

Embeddings are taken from layer `blocks.28.mlp.l3` of Evo 2 7B (Brixi et al., 2026), block 29 of 32, and mean-pooled over the sequence into 4,096 dimensions. Extraction ran on a DGX Spark (evo2 0.6.0, torch 2.9, flash-attn 2.8.3) at 0.32 s per gene and 0.38 s per window, 2.3 h in total.

# 6. Representation

A TopK SAE (Gujral et al., 2025) with 128 features and k = 32 was trained on the standardized window embeddings. The dictionary size and k came from a sweep of 64, 128 and 256 features and k of 16 and 32, keeping the largest dictionary with explained variance of at least 0.8 and fewer than 20 % dead features: mean-pooled embeddings are nearly low-rank (30 principal components explain 99 % of the variance), and 1,024 features left most of them dead. The chosen SAE explains 98.6 % of the validation variance with 10 % dead features.

Each SAE feature and each of the 4,096 original Evo 2 dimensions is a unit. For every unit, its 50 most active genes are tested for over-representation of each GO term (hypergeometric test, Benjamini-Hochberg correction within the unit, FDR 0.05). The 49 insertion-element genes with identical sequences are collapsed to 11 before testing, and genes tied at the cut of 50 are dropped, so a unit that is silent on genes gets no test.

**Figure 1** shows the distribution of the best adjusted p-value per unit. It plays the role of Fig. 3 in Gujral et al. (2025), but measures GO enrichment instead of automated interpretability with an LLM. 73 % of SAE features have at least one significant term (median −log10 p = 2.6), against 96 % of the original dimensions (median 5.0). The gap comes from features that are silent on genes: the SAE was trained on windows, and 42 of its 128 features are active in fewer than 50 genes. Among the 86 features active in at least 50 genes, 93 % are significant (median 4.5). With the secondary criterion of activation > 0, the SAE leads (75 % against 43 %), but there the baseline dimensions mark about half of all genes as active, so the sets are not comparable.

**Figure 2** lists the four strongest SAE features. Feature 26 groups insertion-sequence transposases (transposition, adjusted p = 3e-31) and feature 47 the lipopolysaccharide clusters *waa* and *rfb* (9e-11). Feature 118 (nucleotide binding) is led by highly expressed genes such as *rpoC*, *groL* and *tufA*, and feature 55 (pilus) by leader peptides of 150 nt or less; both are more likely to reflect expression and gene length than the GO term.

**Answer to the study question.** No: with the same test and the same number of genes per unit, the TopK SAE does not separate GO information better than the original dimensions of Evo 2. It concentrates comparable signal in 32 times fewer units, and some of its features are clearly interpretable, but its features are not more often or more strongly enriched than the original dimensions.
