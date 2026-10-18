# sae-ecoli-dna

Sparse autoencoder on Nucleotide Transformer v2 50M embeddings of *E. coli* K-12 MG1655 genes, with GO enrichment compared against the model's original dimensions.

Study question: does a TopK SAE separate functional (GO) information better than the original dimensions of the model?

## Setup

```bash
python -m pip install -r requirements.txt
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `python -m src.FetchData` | `data/raw/` and `data/raw/manifest.json` (URL, version, sha256, date) |

Data sources: NCBI RefSeq GCF_000005845.2 (FASTA and GFF), UniProt proteome UP000000625 (accession, ordered locus, GO) and `go-basic.obo`. Parameters live in `Config.yaml`; analysis decisions in `DECISIONS.md`.

## Tests

```bash
python -m pytest
```
