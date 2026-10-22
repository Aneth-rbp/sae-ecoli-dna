import math

import numpy as np
import pandas as pd
import pytest

from src.Compare import (
    BuildComparison,
    FormatTopTerms,
    Main,
    ReadEnrichment,
    SelectCandidates,
    SummarizeMode,
    TopGeneNames,
    WriteCandidateMarkdown,
)

ENRICHMENT_COLUMNS = ["unit", "mode", "setSize", "bestTerm", "bestTermName", "overlap", "termGenes",
                      "pValue", "adjustedP", "adjustedPGlobal", "isSignificant"]
NAN = float("nan")
ENRICHMENT_ROWS = [
    [0, "topN", 50, "GO:1", "one", 10, 40, 1e-8, 1e-6, 1e-4, True],
    [1, "topN", 50, "GO:2", "two", 5, 60, 1e-3, 1e-2, 0.2, True],
    [2, "topN", 50, "GO:1", "one", 8, 40, 1e-7, 1e-5, 1e-3, True],
    [3, "topN", 50, "", "", 0, 0, NAN, NAN, NAN, False],
    [4, "topN", 50, "GO:3", "three", 2, 300, 0.3, 0.9, 1.0, False],
    [0, "positive", 900, "GO:1", "one", 30, 40, 0.0, 0.0, 0.0, True],
    [1, "positive", 900, "GO:2", "two", 9, 60, 0.5, 0.5, 0.9, False],
]
TERM_COLUMNS = ["unit", "mode", "goId", "name", "overlap", "termGenes", "pValue", "adjustedP"]


def WriteEnrichment(tmp_path, name="SaeEnrichment.tsv"):
    path = tmp_path / name
    pd.DataFrame(ENRICHMENT_ROWS, columns=ENRICHMENT_COLUMNS).to_csv(path, sep="\t", index=False)
    return path


def TestSummarizeModeOnHandExample(tmp_path):
    table = ReadEnrichment(WriteEnrichment(tmp_path))
    summary = SummarizeMode(table[table["mode"] == "topN"], fdr=0.05)
    assert summary["units"] == 5
    assert summary["unitsWithSignificantTerm"] == 3
    assert summary["percentUnitsSignificant"] == pytest.approx(60.0)
    # Untested unit counts as p = 1: values 6, 2, 5, 0, -log10(0.9).
    assert summary["medianNegLog10BestP"] == pytest.approx(2.0)
    assert summary["percentUnitsSignificantGlobal"] == pytest.approx(40.0)
    assert summary["medianNegLog10BestPGlobal"] == pytest.approx(-math.log10(0.2))


def TestSummarizeModeClipsZeroPValue(tmp_path):
    table = ReadEnrichment(WriteEnrichment(tmp_path))
    summary = SummarizeMode(table[table["mode"] == "positive"], fdr=0.05)
    assert math.isfinite(summary["medianNegLog10BestP"])
    assert summary["percentUnitsSignificant"] == pytest.approx(50.0)


def TestBuildComparisonHasRowPerMatrixAndMode(tmp_path):
    tables = {"Sae": ReadEnrichment(WriteEnrichment(tmp_path)), "Baseline": ReadEnrichment(WriteEnrichment(tmp_path))}
    comparison = BuildComparison(tables, fdr=0.05)
    assert list(comparison.columns[:6]) == ["matrix", "mode", "units", "unitsWithSignificantTerm",
                                           "percentUnitsSignificant", "medianNegLog10BestP"]
    assert len(comparison) == 4


def TestSelectCandidatesSkipsRepeatedBestTerm(tmp_path):
    table = ReadEnrichment(WriteEnrichment(tmp_path))
    assert SelectCandidates(table, count=5) == [0, 1]
    assert SelectCandidates(table, count=1) == [0]


def TestFormatTopTermsSortsByAdjustedP():
    terms = pd.DataFrame([
        [0, "topN", "GO:2", "two", 3, 10, 1e-3, 0.02],
        [0, "topN", "GO:1", "one", 5, 10, 1e-6, 0.0001],
        [1, "topN", "GO:9", "nine", 5, 10, 1e-6, 0.0001],
    ], columns=TERM_COLUMNS)
    assert FormatTopTerms(terms, unit=0, limit=5) == "GO:1 one (1.00e-04); GO:2 two (2.00e-02)"


def TestTopGeneNamesUsesActivationOrderAndFallsBackToLocus():
    activations = np.array([[0.0, 0.5], [3.0, 0.1], [1.0, 0.0]], dtype=np.float32)
    geneIds = ["b0001", "b0002", "b0003"]
    names = {"b0001": "thrL", "b0002": "", "b0003": "thrB"}
    assert TopGeneNames(activations, geneIds, names, unit=0, limit=10) == "b0002; thrB"
    assert TopGeneNames(None, geneIds, names, unit=0, limit=10) == ""


def TestWriteCandidateMarkdownLeavesInterpretationEmpty(tmp_path):
    rows = [{"unit": 7, "rank": 1, "topTerms": "GO:1 one (1.00e-04)", "topGenes": "thrL"}]
    path = tmp_path / "CandidateFeatures.md"
    WriteCandidateMarkdown(rows, path)
    text = path.read_text(encoding="utf-8")
    assert "Feature 7" in text and "GO:1 one" in text and "Interpretation:" in text


def TestMainRunsWithoutActivationFile(tmp_path, monkeypatch, capsys):
    enrichmentDir = tmp_path / "results" / "enrichment"
    enrichmentDir.mkdir(parents=True)
    WriteEnrichment(enrichmentDir, "SaeEnrichment.tsv")
    WriteEnrichment(enrichmentDir, "BaselineEnrichment.tsv")
    terms = pd.DataFrame([[0, "topN", "GO:1", "one", 10, 40, 1e-8, 1e-6]], columns=TERM_COLUMNS)
    terms.to_csv(enrichmentDir / "SaeSignificantTerms.tsv", sep="\t", index=False)
    (tmp_path / "Config.yaml").write_text(
        "paths: {results: results, processed: data/processed, embeddings: data/embeddings}\n"
        "enrichment: {fdr: 0.05}\ncompare: {candidates: 5}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    Main([])
    candidates = pd.read_csv(tmp_path / "results" / "CandidateFeatures.tsv", sep="\t", keep_default_na=False)
    assert list(candidates["unit"]) == [0, 1]
    assert list(candidates["topGenes"]) == ["", ""]
    assert "Warning" in capsys.readouterr().err
    assert len(pd.read_csv(tmp_path / "results" / "Comparison.tsv", sep="\t")) == 4
    assert (tmp_path / "results" / "CandidateFeatures.md").exists()
