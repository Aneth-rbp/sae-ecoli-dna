import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.FetchData import LoadConfig

MATRICES = ["Sae", "Baseline"]
MODES = ["topN", "positive"]
CANDIDATE_MODE = "topN"
COMPARISON_COLUMNS = ["matrix", "mode", "units", "unitsWithSignificantTerm", "percentUnitsSignificant",
                      "medianNegLog10BestP", "percentUnitsSignificantGlobal", "medianNegLog10BestPGlobal"]
CANDIDATE_COLUMNS = ["unit", "rank", "topTerms", "topGenes"]
DEFAULT_CANDIDATES = 5
TOP_TERMS = 5
TOP_GENES = 10
SMALLEST_P = np.finfo(float).tiny


def ReadEnrichment(path):
    table = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["", "NaN", "nan"])
    table["bestTerm"] = table["bestTerm"].fillna("").astype(str)
    table["isSignificant"] = table["isSignificant"].astype(str).str.lower() == "true"
    return table


def NegLog10(pValues):
    # Untested units count as p = 1 so every unit stays in the median.
    clipped = pValues.fillna(1.0).clip(lower=SMALLEST_P)
    return -np.log10(clipped)


def SummarizeMode(table, fdr):
    units = len(table)
    significant = int(table["isSignificant"].sum())
    significantGlobal = int((table["adjustedPGlobal"] < fdr).sum())
    return {
        "units": units,
        "unitsWithSignificantTerm": significant,
        "percentUnitsSignificant": 100 * significant / units if units else float("nan"),
        "medianNegLog10BestP": float(NegLog10(table["adjustedP"]).median()),
        "percentUnitsSignificantGlobal": 100 * significantGlobal / units if units else float("nan"),
        "medianNegLog10BestPGlobal": float(NegLog10(table["adjustedPGlobal"]).median()),
    }


def BuildComparison(tables, fdr):
    rows = []
    for matrix, table in tables.items():
        for mode in MODES:
            summary = SummarizeMode(table[table["mode"] == mode], fdr)
            rows.append({"matrix": matrix, "mode": mode, **summary})
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def SelectCandidates(table, count):
    significant = table[(table["mode"] == CANDIDATE_MODE) & table["isSignificant"]]
    ranked = significant.sort_values(["adjustedP", "pValue", "unit"], kind="stable")
    selected, usedTerms = [], set()
    for unit, bestTerm in zip(ranked["unit"], ranked["bestTerm"]):
        if len(selected) == count:
            break
        if bestTerm in usedTerms:
            continue
        usedTerms.add(bestTerm)
        selected.append(int(unit))
    return selected


def FormatTopTerms(terms, unit, limit):
    unitTerms = terms[(terms["unit"] == unit) & (terms["mode"] == CANDIDATE_MODE)]
    best = unitTerms.sort_values(["adjustedP", "pValue"], kind="stable").head(limit)
    return "; ".join(f"{row.goId} {row.name} ({row.adjustedP:.2e})" for row in best.itertuples())


def TopGeneNames(activations, geneIds, geneNames, unit, limit):
    if activations is None:
        return ""
    column = activations[:, unit]
    order = [row for row in np.argsort(-column, kind="stable")[:limit] if column[row] > 0]
    return "; ".join(geneNames.get(geneIds[row]) or geneIds[row] for row in order)


def BuildCandidateRows(enrichment, terms, genes, count):
    activations, geneIds, geneNames = genes
    return [
        {"unit": unit, "rank": rank, "topTerms": FormatTopTerms(terms, unit, TOP_TERMS),
         "topGenes": TopGeneNames(activations, geneIds, geneNames, unit, TOP_GENES)}
        for rank, unit in enumerate(SelectCandidates(enrichment, count), start=1)
    ]


def WriteCandidateMarkdown(rows, path):
    lines = ["# Candidate SAE features", ""]
    for row in rows:
        lines += [f"## Feature {row['unit']} (rank {row['rank']})", "",
                  f"Terms: {row['topTerms'] or '-'}", "", f"Genes: {row['topGenes'] or '-'}", "",
                  "Interpretation:", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def LoadGeneNames(processedDir):
    genes = pd.read_csv(Path(processedDir) / "Genes.tsv", sep="\t", usecols=["locusTag", "geneName"],
                        keep_default_na=False, dtype=str)
    return dict(zip(genes["locusTag"], genes["geneName"]))


def LoadGeneActivations(config, resultsDir):
    activationPath = resultsDir / "sae" / "GeneActivations.npy"
    idsPath = Path(config["paths"].get("embeddings", "data/embeddings")) / "GeneIds.txt"
    if not activationPath.exists() or not idsPath.exists():
        print(f"Warning: {activationPath} or {idsPath} missing; topGenes left empty.", file=sys.stderr)
        return None, [], {}
    geneIds = idsPath.read_text(encoding="utf-8").split()
    return np.load(activationPath), geneIds, LoadGeneNames(config["paths"]["processed"])


def ReadEnrichmentTables(enrichmentDir):
    paths = {matrix: enrichmentDir / f"{matrix}Enrichment.tsv" for matrix in MATRICES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        sys.exit(f"Missing enrichment results (run src.Enrichment first): {', '.join(missing)}")
    return {matrix: ReadEnrichment(path) for matrix, path in paths.items()}


def ParseCandidateCount(arguments, default):
    if "--candidates" in arguments:
        return int(arguments[arguments.index("--candidates") + 1])
    return default


def Main(arguments=None):
    config = LoadConfig()
    resultsDir = Path(config["paths"]["results"])
    resultsDir.mkdir(parents=True, exist_ok=True)
    enrichmentDir = resultsDir / "enrichment"
    configured = config.get("compare", {}).get("candidates", DEFAULT_CANDIDATES)
    count = ParseCandidateCount(sys.argv[1:] if arguments is None else arguments, configured)
    tables = ReadEnrichmentTables(enrichmentDir)
    comparison = BuildComparison(tables, config["enrichment"]["fdr"])
    comparison.to_csv(resultsDir / "Comparison.tsv", sep="\t", index=False, lineterminator="\n")
    terms = pd.read_csv(enrichmentDir / "SaeSignificantTerms.tsv", sep="\t", keep_default_na=False, na_values=[""])
    rows = BuildCandidateRows(tables["Sae"], terms, LoadGeneActivations(config, resultsDir), count)
    pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).to_csv(resultsDir / "CandidateFeatures.tsv", sep="\t", index=False, lineterminator="\n")
    WriteCandidateMarkdown(rows, resultsDir / "CandidateFeatures.md")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    Main()
