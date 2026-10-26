# Candidate SAE features

TopK SAE with 128 features and k = 32 (#31), trained on genomic windows and applied to the 3,583 genes. Enrichment uses the 50 most active genes of each feature (top-N), a hypergeometric test and per-feature BH correction, after collapsing identical sequences (#26). The four features come from `results/CandidateFeatures.tsv`: two are presented as functional signal and two as confounders.

Genome-wide reference: median CDS length 900 nt, median GC 0.524, and 44 genes of 150 nt or less (1.2 %).

## Feature 26: transposition

- **Terms:** transposition (adjusted p 3.0e-31; 19 of the term's 23 genes are in the top 50), DNA transposition (1.3e-29), transposase activity (1.7e-26).
- **Genes:** insF1-5, insI1-3, insH5, insL1-3, insG, insE1.
- **Interpretation:** responds to insertion-sequence (IS) transposases. It is the strongest signal in the analysis and survives collapsing identical copies, so it does not come from counting the same gene several times. IS families have highly conserved sequences that differ from the rest of the genome, which explains why one feature groups them, but it also means the SAE may be recognizing a sequence family rather than a function in the abstract.

## Feature 47: lipopolysaccharide

- **Terms:** lipopolysaccharide biosynthetic process (8.7e-11; 13 of 58 genes), polysaccharide biosynthetic process (4.2e-08).
- **Genes:** waaS, waaU, waaL, waaZ, rfbX, wbbK, gnsB, nanX, yibA.
- **Interpretation:** groups the LPS core and O-antigen synthesis genes (the *waa* and *rfb* clusters). These clusters were acquired by horizontal transfer and are GC-poor: the top-50 median GC is 0.367 against 0.524 genome-wide. The enrichment is real, but the feature most likely encodes base composition, and the function shows up because these genes share that origin.

## Feature 118: nucleotide binding (confounder: expression)

- **Terms:** purine ribonucleotide binding (3.6e-17; 32 of 377 genes), ribonucleotide binding (2.1e-16).
- **Genes:** rpoC, groL, rpsA, rho, carB, tufA, tufB, rne, clpX, glnA.
- **Interpretation:** nearly all of the most active genes are among the most highly expressed in the cell (RNA polymerase, the GroEL chaperonin, elongation factor Tu, ribosomal protein S1). ATP/GTP binding terms are very broad and common among those proteins. The hypothesis is that the feature captures the codon bias of highly expressed genes, not nucleotide binding. This was not tested: it would need a comparison of activation against a codon adaptation index (CAI) or expression data.

## Feature 55: pilus (confounder: length)

- **Terms:** pilus (1.0e-13; 13 of 37 genes), cell adhesion (2.4e-11), transcriptional attenuation (9.5e-11; 7 of 9 genes).
- **Genes:** pheM, hisL, thrL, leuL, ivbL, pyrL, mgtL, rhoL, tnaC, ibsA, ibsC, shoB.
- **Interpretation:** 19 of the 50 most active genes are 150 nt or shorter, against 1.2 % genome-wide, and the top-50 median length is 472 nt. The top ones are attenuation leader peptides and small toxins. Because embeddings are mean-pooled over the sequence, a short gene yields a different vector from its length alone. The "pilus" term probably comes from fimbrial subunits, which are also shorter than the genome median (around 550 nt). It serves as a warning: a significant term is not enough to call a feature functional.

## Left out

Feature 36 (rank 3; carnitine metabolic process, 8.0e-15) is not interpreted: its most active genes (yfcE, ygjR, wcaA, kefF) are unrelated to the term, and the enrichment rests on a few genes.
