import math

import numpy as np
import pandas as pd
import pytest

from src.MakeFigures import (
    BuildFeatureMarkdown,
    BuildHistogramBins,
    BestNegLog10,
    PercentSignificant,
    ReadCandidateFeatures,
    WriteFigures,
)

ENRICHMENT_COLUMNS = [
    "unit", "mode", "setSize", "bestTerm", "bestTermName", "overlap", "termGenes",
    "pValue", "adjustedP", "adjustedPGlobal", "isSignificant",
]


def WriteEnrichment(targetPath, adjustedPs, mode="topN"):
    rows = []
    for unit, adjustedP in enumerate(adjustedPs):
        isTested = not math.isnan(adjustedP)
        rows.append({
            "unit": unit, "mode": mode, "setSize": 50,
            "bestTerm": f"GO:{unit:07d}" if isTested else "", "bestTermName": "term" if isTested else "",
            "overlap": 5, "termGenes": 20, "pValue": adjustedP, "adjustedP": adjustedP,
            "adjustedPGlobal": adjustedP, "isSignificant": isTested and adjustedP < 0.05,
        })
    pd.DataFrame(rows, columns=ENRICHMENT_COLUMNS).to_csv(targetPath, sep="\t", index=False)


def WriteCandidates(targetPath, count=6):
    rows = [{
        "unit": 100 + rank, "rank": rank,
        "topTerms": f"GO:0006412 translation (1.2e-08); GO:0005840 ribosome | large ({rank}.0e-05)",
        "topGenes": "rplA; rplB; rplC; rpsA; rpsB",
    } for rank in range(count, 0, -1)]
    pd.DataFrame(rows).to_csv(targetPath, sep="\t", index=False)


def BuildConfig(tmpPath):
    return {
        "paths": {"results": str(tmpPath / "results"), "figures": str(tmpPath / "figures")},
        "enrichment": {"fdr": 0.05},
        "figures": {"dpi": 100, "histogramWidthInches": 3.35, "histogramHeightInches": 2.3,
                    "tableWidthInches": 6.9, "fontSize": 7, "histogramBins": 20, "tableFeatures": 5,
                    "showPositiveMode": False},
    }


def WriteInputs(tmpPath):
    enrichmentDir = tmpPath / "results" / "enrichment"
    enrichmentDir.mkdir(parents=True)
    WriteEnrichment(enrichmentDir / "SaeEnrichment.tsv", [1e-10, 1e-3, 0.5, float("nan")])
    WriteEnrichment(enrichmentDir / "BaselineEnrichment.tsv", [0.01, 0.2, 0.9, 1.0, 0.04, float("nan")])
    WriteCandidates(tmpPath / "results" / "CandidateFeatures.tsv")


def TestBestNegLog10TreatsUntestedUnitsAsZero():
    table = pd.DataFrame({"mode": ["topN", "topN", "positive"], "adjustedP": [0.01, float("nan"), 1e-5]})
    assert BestNegLog10(table, "topN") == pytest.approx([2.0, 0.0])


def TestPercentSignificantCountsAllUnitsOfTheMode():
    table = pd.DataFrame({"mode": ["topN"] * 4, "isSignificant": [True, "False", "True", False]})
    assert PercentSignificant(table, "topN") == pytest.approx(50.0)


def TestHistogramBinsShareOneRangeForBothMatrices():
    bins = BuildHistogramBins([np.array([0.0, 3.0]), np.array([1.0, 7.5])], 10)
    assert bins[0] == 0.0
    assert bins[-1] >= 7.5
    assert len(bins) == 11


def TestHistogramBinsHandleAllZeroScores():
    bins = BuildHistogramBins([np.zeros(3), np.zeros(2)], 4)
    assert bins[-1] > bins[0]


def TestReadCandidateFeaturesKeepsTopRanksInOrder(tmp_path):
    WriteCandidates(tmp_path / "CandidateFeatures.tsv")
    candidates = ReadCandidateFeatures(tmp_path / "CandidateFeatures.tsv", 4)
    assert list(candidates["rank"]) == [1, 2, 3, 4]


def TestFeatureMarkdownEscapesPipesAndListsEveryFeature(tmp_path):
    WriteCandidates(tmp_path / "CandidateFeatures.tsv")
    markdown = BuildFeatureMarkdown(ReadCandidateFeatures(tmp_path / "CandidateFeatures.tsv", 5))
    lines = markdown.strip().splitlines()
    assert lines[0].startswith("| Feature |")
    assert len(lines) == 2 + 5
    assert "ribosome \\| large" in markdown


def TestWriteFiguresProducesEveryContractFile(tmp_path):
    WriteInputs(tmp_path)
    WriteFigures(BuildConfig(tmp_path))
    figuresDir = tmp_path / "figures"
    for name in ["PValueHistogram.png", "PValueHistogram.pdf", "FeatureTable.png", "FeatureTable.pdf", "FeatureTable.md"]:
        assert (figuresDir / name).stat().st_size > 0


def TestWriteFiguresDrawsPositiveModeWhenAsked(tmp_path):
    WriteInputs(tmp_path)
    enrichmentDir = tmp_path / "results" / "enrichment"
    WriteEnrichment(enrichmentDir / "SaeEnrichment.tsv", [1e-4, 0.3], mode="positive")
    WriteEnrichment(enrichmentDir / "BaselineEnrichment.tsv", [0.2, 0.6], mode="positive")
    config = BuildConfig(tmp_path)
    config["figures"]["showPositiveMode"] = True
    WriteFigures(config)
    assert (tmp_path / "figures" / "PValueHistogram.png").exists()
