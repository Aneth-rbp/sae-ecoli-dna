import numpy as np
import pytest
from scipy.stats import hypergeom

from src.Enrichment import (
    AdjustBenjaminiHochberg,
    AlignRows,
    BuildMembership,
    CollapseIdenticalSequences,
    ComputeHypergeometric,
    EnrichMatrix,
    SelectPositiveSets,
    SelectTopSets,
)

GENE_COUNT = 200
PLANTED_GENES = 20


def BuildAnnotation():
    geneIds = [f"b{index:04d}" for index in range(GENE_COUNT)]
    rng = np.random.default_rng(0)
    geneTerms = [{"GO:0000002"} if rng.random() < 0.3 else {"GO:0000003"} for _ in geneIds]
    for index in range(PLANTED_GENES):
        geneTerms[index] = geneTerms[index] | {"GO:0000001"}
    termIds = ["GO:0000001", "GO:0000002", "GO:0000003"]
    termNames = {"GO:0000001": "planted", "GO:0000002": "common", "GO:0000003": "other"}
    return {"geneIds": geneIds, "termIds": termIds, "termNames": termNames,
            "membership": BuildMembership(geneTerms, termIds)}


def BuildMatrix():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(GENE_COUNT, 2)).astype(np.float32)
    matrix[:PLANTED_GENES, 0] += 10
    return matrix


SETTINGS = {"topN": PLANTED_GENES, "fdr": 0.05}


def TestPlantedTermIsSignificantAndRandomUnitIsNot():
    annotation = BuildAnnotation()
    summary, significant = EnrichMatrix(BuildMatrix(), annotation["geneIds"], annotation, SETTINGS)
    topRows = summary[summary["mode"] == "topN"].set_index("unit")
    assert topRows.loc[0, "bestTerm"] == "GO:0000001"
    assert topRows.loc[0, "overlap"] == PLANTED_GENES
    assert bool(topRows.loc[0, "isSignificant"])
    assert not bool(topRows.loc[1, "isSignificant"])
    assert ((significant["unit"] == 0) & (significant["goId"] == "GO:0000001")).any()


def TestOutputColumnsFollowContract():
    annotation = BuildAnnotation()
    summary, significant = EnrichMatrix(BuildMatrix(), annotation["geneIds"], annotation, SETTINGS)
    assert list(summary.columns) == ["unit", "mode", "setSize", "bestTerm", "bestTermName", "overlap", "termGenes",
                                     "pValue", "adjustedP", "adjustedPGlobal", "isSignificant"]
    assert list(significant.columns) == ["unit", "mode", "goId", "name", "overlap", "termGenes", "pValue", "adjustedP"]
    assert len(summary) == 4


def TestBenjaminiHochbergMatchesHandExample():
    pValues = np.array([[0.01, 0.04, 0.03, np.nan, 0.005]])
    expected = np.array([[0.02, 0.04, 0.04, np.nan, 0.02]])
    np.testing.assert_allclose(AdjustBenjaminiHochberg(pValues), expected)


def TestBenjaminiHochbergCapsAtOne():
    adjusted = AdjustBenjaminiHochberg(np.array([0.9, 0.95, 0.99]))
    assert adjusted.max() <= 1.0


def TestTopSetsTakeExactlyNWithoutTiesAtTheCut():
    matrix = np.array([[1.0], [3.0], [4.0], [2.0], [0.0]])
    sets = SelectTopSets(matrix, 2)
    assert sets[:, 0].tolist() == [False, True, True, False, False]


def TestTopSetsDropGenesTiedAtTheCut():
    matrix = np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [5.0, 0.0], [0.0, 0.0]])
    sets = SelectTopSets(matrix, 2)
    assert sets[:, 0].tolist() == [False, False, False, True, False]
    assert sets[:, 1].tolist() == [False, False, True, False, False]


def TestTopSetsKeepAWholeTieThatFitsInN():
    matrix = np.array([[3.0], [3.0], [1.0]])
    assert SelectTopSets(matrix, 2)[:, 0].tolist() == [True, True, False]


def TestTopSetsOfASilentUnitAreEmpty():
    assert not SelectTopSets(np.zeros((5, 1)), 2).any()


def TestPositiveModeUsesZeroThresholdAndEmptyUnitGetsEmptyRow():
    annotation = BuildAnnotation()
    matrix = np.zeros((GENE_COUNT, 2), dtype=np.float32)
    matrix[:PLANTED_GENES, 0] = 1.0
    assert SelectPositiveSets(matrix).sum(axis=0).tolist() == [PLANTED_GENES, 0]
    summary, _ = EnrichMatrix(matrix, annotation["geneIds"], annotation, SETTINGS)
    positive = summary[summary["mode"] == "positive"].set_index("unit")
    assert positive.loc[0, "bestTerm"] == "GO:0000001" and bool(positive.loc[0, "isSignificant"])
    assert positive.loc[1, "setSize"] == 0 and positive.loc[1, "bestTerm"] == ""
    assert not bool(positive.loc[1, "isSignificant"])


def TestAlignRowsReordersById():
    matrix = np.array([[2.0], [1.0]])
    aligned = AlignRows(matrix, ["b0002", "b0001"], ["b0001", "b0002"])
    assert aligned[:, 0].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("matrixIds", [["b0001", "b0003"], ["b0001"], ["b0001", "b0001"]])
def TestAlignRowsFailsOnMismatchedIds(matrixIds):
    matrix = np.zeros((len(matrixIds), 1))
    with pytest.raises(ValueError):
        AlignRows(matrix, matrixIds, ["b0001", "b0002"])


def TestCollapseKeepsFirstCopyWithUnionOfTerms():
    keptIds, keptTerms = CollapseIdenticalSequences(
        ["b0001", "b0002", "b0003", "b0004"], [{"GO:1"}, {"GO:2"}, {"GO:3"}, {"GO:4"}], ["ACG", "TTT", "ACG", "ACG"])
    assert keptIds == ["b0001", "b0002"]
    assert keptTerms == [{"GO:1", "GO:3", "GO:4"}, {"GO:2"}]


def TestAlignRowsDropsCollapsedCopiesButChecksTheWholeTable():
    matrix = np.array([[3.0], [1.0], [2.0]])
    aligned = AlignRows(matrix, ["b0003", "b0001", "b0002"], ["b0001", "b0003"], ["b0001", "b0002", "b0003"])
    assert aligned[:, 0].tolist() == [1.0, 3.0]
    with pytest.raises(ValueError):
        AlignRows(matrix[:2], ["b0003", "b0001"], ["b0001", "b0003"], ["b0001", "b0002", "b0003"])


def TestPValuesMatchDirectHypergeometricTail():
    annotation = BuildAnnotation()
    sets = SelectTopSets(BuildMatrix(), PLANTED_GENES)
    tested = ComputeHypergeometric(sets, annotation["membership"])
    overlap = tested["overlap"]
    expected = hypergeom.sf(overlap - 1, GENE_COUNT, tested["termGenes"][None, :], PLANTED_GENES)
    np.testing.assert_allclose(tested["pValue"], np.where(overlap >= 1, expected, np.nan))
