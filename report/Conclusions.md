# 7. Conclusions and future work

On mean-pooled Evo 2 7B embeddings of *E. coli* genes, a TopK SAE with 128 features does not separate GO information better than the 4,096 original dimensions: fewer of its features carry a significant term (73 % against 96 %) and their best terms are weaker. What the SAE does is compress: it reconstructs 98.6 % of the variance with 32 active features per sequence, and a few of its features are readable on their own, such as one for insertion-sequence transposases and one for lipopolysaccharide synthesis. These results describe one model, one layer and one GO test; they do not show that SAEs are unhelpful for genomic language models in general.

**Limitations.**

- *Seed.* TopK SAEs trained with different seeds learn different dictionaries; this SAE was trained once (seed 42), so individual features, including the four in Figure 2, may not reappear in another run.
- *Data size.* About 3,500 genes and 18,564 windows from one genome are a small dataset. Mean-pooled embeddings span about 30 effective dimensions, which capped the dictionary at 128 features, far from the 20,000 per layer that Gujral et al. (2025) trained on a protein model.
- *Training on windows, testing on genes.* 42 of the 128 features are active in fewer than 50 genes and add little to the gene-level test.
- *Mean pooling.* Averaging over the sequence keeps length and base composition in the vector, and two of the four candidate features follow gene length and, likely, expression level rather than function.

**Future work.**

- *Transcoders*, which Gujral et al. (2025) found as interpretable as SAEs, to compare architectures on the same data.
- *Several layers* of Evo 2, instead of one, to see where functional information is clearest.
- *Nucleotide level*: SAEs on per-position activations instead of pooled ones, to find motifs such as promoters or ribosome binding sites.
- *Automated interpretation with an LLM*, as in Gujral et al. (2025), to score features beyond GO.
- *Sequence similarity* as a control: checking whether a feature's top genes are homologous would separate learned function from shared sequence, as with the insertion elements.
