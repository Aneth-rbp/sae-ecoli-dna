import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src.FetchData import LoadConfig

SUMMARY_COLUMNS = ["unit", "mode", "setSize", "bestTerm", "bestTermName", "overlap", "termGenes",
                   "pValue", "adjustedP", "adjustedPGlobal", "isSignificant"]
SIGNIFICANT_COLUMNS = ["unit", "mode", "goId", "name", "overlap", "termGenes", "pValue", "adjustedP"]
MATRIX_NAMES = {"sae": "Sae", "baseline": "Baseline"}


def ReadGeneTerms(genesPath):
    genes = pd.read_csv(genesPath, sep="\t", usecols=["locusTag", "goTerms", "sequence"], dtype=str)
    return genes["locusTag"].tolist(), [set(field.split(";")) for field in genes["goTerms"]], genes["sequence"].tolist()


def CollapseIdenticalSequences(geneIds, geneTerms, sequences):
    # One gene per identical sequence (the first in table order), with the union of the copies' GO terms.
    firstBySequence = {}
    keptIds, keptTerms = [], []
    for geneId, terms, sequence in zip(geneIds, geneTerms, sequences):
        if sequence in firstBySequence:
            keptTerms[firstBySequence[sequence]] |= terms
            continue
        firstBySequence[sequence] = len(keptIds)
        keptIds.append(geneId)
        keptTerms.append(set(terms))
    return keptIds, keptTerms


def BuildMembership(geneTerms, termIds):
    termIndex = {termId: index for index, termId in enumerate(termIds)}
    membership = np.zeros((len(geneTerms), len(termIds)), dtype=bool)
    for geneIndex, terms in enumerate(geneTerms):
        membership[geneIndex, [termIndex[termId] for termId in terms if termId in termIndex]] = True
    return membership


def LoadAnnotation(processedDir):
    tableIds, tableTerms, sequences = ReadGeneTerms(processedDir / "Genes.tsv")
    geneIds, geneTerms = CollapseIdenticalSequences(tableIds, tableTerms, sequences)
    terms = pd.read_csv(processedDir / "GoTerms.tsv", sep="\t", dtype={"goId": str, "name": str})
    termIds = terms["goId"].tolist()
    return {"geneIds": geneIds, "tableIds": tableIds, "termIds": termIds,
            "termNames": dict(zip(termIds, terms["name"])), "membership": BuildMembership(geneTerms, termIds)}


def AlignRows(matrix, matrixIds, geneIds, tableIds=None):
    tableIds = geneIds if tableIds is None else tableIds
    if len(matrixIds) != matrix.shape[0] or len(set(matrixIds)) != len(matrixIds):
        raise ValueError(f"Matrix has {matrix.shape[0]} rows but {len(matrixIds)} ids, or ids repeat")
    if set(matrixIds) != set(tableIds):
        missing = len(set(tableIds) - set(matrixIds))
        extra = len(set(matrixIds) - set(tableIds))
        raise ValueError(f"Matrix ids do not match the gene table: {missing} missing, {extra} extra")
    rowIndex = {geneId: index for index, geneId in enumerate(matrixIds)}
    return matrix[[rowIndex[geneId] for geneId in geneIds]]


def SelectTopSets(matrix, topN):
    # A tie that crosses the cut is dropped whole, so table order never picks genes (silent SAE units tie at 0).
    cutValue = -np.sort(-matrix, axis=0)[min(topN, matrix.shape[0]) - 1]
    atOrAboveCut = matrix >= cutValue
    return np.where(atOrAboveCut.sum(axis=0) <= topN, atOrAboveCut, matrix > cutValue)


def SelectPositiveSets(matrix):
    return matrix > 0


def AdjustBenjaminiHochberg(pValues):
    order = np.argsort(pValues, axis=-1)
    sortedP = np.take_along_axis(pValues, order, axis=-1)
    testCount = np.sum(~np.isnan(pValues), axis=-1, keepdims=True)
    ranks = np.arange(1, pValues.shape[-1] + 1)
    scaled = np.where(np.isnan(sortedP), np.inf, sortedP * testCount / ranks)
    monotone = np.minimum.accumulate(scaled[..., ::-1], axis=-1)[..., ::-1]
    adjusted = np.empty_like(pValues, dtype=float)
    np.put_along_axis(adjusted, order, np.minimum(monotone, 1.0), axis=-1)
    return np.where(np.isnan(pValues), np.nan, adjusted)


def OverRepresentationP(overlap, setSize, termGenes, geneCount):
    # Many cells share the same counts, so the slow survival function runs once per distinct triple.
    pValues = np.full(overlap.shape, np.nan)
    isTested = overlap >= 1
    triples = np.stack(np.broadcast_arrays(overlap, setSize, termGenes), axis=-1)[isTested]
    uniqueTriples, inverse = np.unique(triples, axis=0, return_inverse=True)
    uniqueP = hypergeom.sf(uniqueTriples[:, 0] - 1, geneCount, uniqueTriples[:, 2], uniqueTriples[:, 1])
    pValues[isTested] = uniqueP[inverse.ravel()]
    return pValues


def ComputeHypergeometric(sets, membership):
    overlap = np.rint(sets.T.astype(np.float32) @ membership.astype(np.float32)).astype(np.int64)
    setSize = sets.sum(axis=0)[:, None]
    termGenes = membership.sum(axis=0)[None, :]
    pValues = OverRepresentationP(overlap, setSize, termGenes, membership.shape[0])
    return {"overlap": overlap, "setSize": setSize[:, 0], "termGenes": termGenes[0], "pValue": pValues,
            "adjustedP": AdjustBenjaminiHochberg(pValues),
            "adjustedPGlobal": AdjustBenjaminiHochberg(pValues.ravel()).reshape(pValues.shape)}


def PickBestTerms(tested):
    adjusted = np.where(np.isnan(tested["adjustedP"]), np.inf, tested["adjustedP"])
    isLowest = adjusted == adjusted.min(axis=1, keepdims=True)
    pValues = np.where(isLowest & ~np.isnan(tested["pValue"]), tested["pValue"], np.inf)
    return np.argmin(pValues, axis=1), np.isfinite(pValues.min(axis=1))


def BuildSummaryRows(tested, mode, annotation, fdr):
    bestIndex, hasTest = PickBestTerms(tested)
    units = np.arange(len(bestIndex))
    rows = pd.DataFrame({"unit": units, "mode": mode, "setSize": tested["setSize"]})
    bestTerms = [annotation["termIds"][index] for index in bestIndex]
    rows["bestTerm"] = np.where(hasTest, bestTerms, "")
    rows["bestTermName"] = np.where(hasTest, [annotation["termNames"][termId] for termId in bestTerms], "")
    for column in ["overlap", "pValue", "adjustedP", "adjustedPGlobal"]:
        rows[column] = np.where(hasTest, tested[column][units, bestIndex], np.nan)
    rows["termGenes"] = np.where(hasTest, tested["termGenes"][bestIndex], np.nan)
    rows["overlap"] = rows["overlap"].astype("Int64")
    rows["termGenes"] = rows["termGenes"].astype("Int64")
    rows["isSignificant"] = hasTest & (rows["adjustedP"] < fdr)
    return rows[SUMMARY_COLUMNS]


def BuildSignificantRows(tested, mode, annotation, fdr):
    units, termIndex = np.nonzero(np.nan_to_num(tested["adjustedP"], nan=1.0) < fdr)
    rows = pd.DataFrame({"unit": units, "mode": mode, "goId": np.array(annotation["termIds"], dtype=object)[termIndex]})
    rows["name"] = [annotation["termNames"][termId] for termId in rows["goId"]]
    rows["overlap"] = tested["overlap"][units, termIndex]
    rows["termGenes"] = tested["termGenes"][termIndex]
    rows["pValue"] = tested["pValue"][units, termIndex]
    rows["adjustedP"] = tested["adjustedP"][units, termIndex]
    return rows.sort_values(["unit", "adjustedP", "pValue"], kind="stable")[SIGNIFICANT_COLUMNS]


def EnrichMatrix(matrix, matrixIds, annotation, settings):
    aligned = AlignRows(np.asarray(matrix, dtype=np.float32), list(matrixIds), annotation["geneIds"],
                        annotation.get("tableIds"))
    setsByMode = {"topN": SelectTopSets(aligned, settings["topN"]), "positive": SelectPositiveSets(aligned)}
    summaries, significants = [], []
    for mode, sets in setsByMode.items():
        tested = ComputeHypergeometric(sets, annotation["membership"])
        summaries.append(BuildSummaryRows(tested, mode, annotation, settings["fdr"]))
        significants.append(BuildSignificantRows(tested, mode, annotation, settings["fdr"]))
    return pd.concat(summaries, ignore_index=True), pd.concat(significants, ignore_index=True)


def MatrixPaths(config):
    embeddingsDir = Path(config["paths"].get("embeddings", "data/embeddings"))
    resultsDir = Path(config["paths"]["results"])
    matrixPaths = {"sae": resultsDir / "sae" / "GeneActivations.npy", "baseline": embeddingsDir / "GeneEmbeddings.npy"}
    return matrixPaths, embeddingsDir / "GeneIds.txt", resultsDir / "enrichment"


def RunMatrix(name, matrixPath, matrixIds, annotation, config, outputDir):
    startTime = time.perf_counter()
    summary, significant = EnrichMatrix(np.load(matrixPath), matrixIds, annotation, config["enrichment"])
    prefix = MATRIX_NAMES[name]
    summary.to_csv(outputDir / f"{prefix}Enrichment.tsv", sep="\t", index=False)
    significant.to_csv(outputDir / f"{prefix}SignificantTerms.tsv", sep="\t", index=False)
    counts = summary.groupby("mode")["isSignificant"].sum().to_dict()
    print(f"{prefix}: significant units by mode {counts} in {time.perf_counter() - startTime:.1f} s")


def Main():
    parser = argparse.ArgumentParser(description="GO enrichment per unit of a genes x units matrix.")
    parser.add_argument("--matrix", choices=["sae", "baseline", "all"], default="all")
    arguments = parser.parse_args()
    config = LoadConfig()
    matrixPaths, idsPath, outputDir = MatrixPaths(config)
    outputDir.mkdir(parents=True, exist_ok=True)
    matrixIds = idsPath.read_text(encoding="utf-8").split()
    annotation = LoadAnnotation(Path(config["paths"]["processed"]))
    print(f"Genes: {len(annotation['tableIds'])} in the table, {len(annotation['geneIds'])} after collapsing "
          "identical sequences")
    names = list(MATRIX_NAMES) if arguments.matrix == "all" else [arguments.matrix]
    for name in names:
        RunMatrix(name, matrixPaths[name], matrixIds, annotation, config, outputDir)


if __name__ == "__main__":
    Main()
