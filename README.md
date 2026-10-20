# sae-ecoli-dna

Sparse autoencoder on Evo 2 7B embeddings of *E. coli* K-12 MG1655 genes, with GO enrichment compared against the model's original dimensions.

Study question: does a TopK SAE separate functional (GO) information better than the original dimensions of the model?

## Setup

### Dev machine (data download, tests)

```bash
python -m pip install -r requirements.txt
```

The model does not run here.

### DGX Spark (embeddings)

The Spark is arm64 with a GB10 GPU (Blackwell, sm_121, not Hopper), so only the bf16 7B checkpoints apply; the FP8 40B/20B/1B ones do not. Use the NGC image `nvcr.io/nvidia/pytorch:25.10-py3`, the first tag that supports GB10; it ships torch 2.9 + CUDA 13, so torch is not pinned in `requirements-evo2.txt`.

flash-attn does not support sm_121 officially. Uninstall the copy bundled in the image first (pip treats it as satisfied and skips the build), then build it with `ninja` installed (about 75 min; hours without it):

```bash
pip uninstall -y flash-attn
TORCH_CUDA_ARCH_LIST="12.0" FLASH_ATTN_CUDA_ARCHS="120" pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements-evo2.txt
python -m evo2.test.test_evo2_generation --model_name evo2_7b
```

The arch list is `12.0`, not `12.1`. Known risk: the Evo 2 README recommends torch 2.6 or 2.7; the image ships 2.9.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `python -m src.FetchData` | `data/raw/` and `data/raw/manifest.json` (URL, version, sha256, date) |

`--source <name>` (repeatable) downloads only the named `Config.yaml` sources and merges them into the existing manifest. On the DGX Spark, `bash scripts/FetchUniprotOnSpark.sh` creates `.venv`, installs only the fetch dependencies and downloads the UniProt proteome.

Data sources: NCBI RefSeq GCF_000005845.2 (FASTA and GFF), UniProt proteome UP000000625 (accession, ordered locus, GO) and `go-basic.obo`. Parameters live in `Config.yaml`; analysis decisions in `DECISIONS.md`.

## Tests

```bash
python -m pytest
```
