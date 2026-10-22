import numpy as np
import pytest

from src.ExtractEmbeddings import (
    AssembleMatrix,
    CheckCheckpointInfo,
    GeneId,
    ListShards,
    MeanPool,
    ReadRecords,
    ResolveLayer,
    RunSet,
    ValidateMatrix,
    WindowId,
)

HIDDEN = 4
BASE_CODES = {"A": 1.0, "C": 2.0, "G": 3.0, "T": 4.0}
OPTIONS = {"info": {"model": "fake", "layer": "blocks.0"}, "checkpointEvery": 2}


def FakeEmbedder(sequence):
    codes = np.array([BASE_CODES[base] for base in sequence], dtype=np.float32)
    return np.stack([codes * (column + 1) for column in range(HIDDEN)], axis=1)


def ExpectedRow(sequence):
    return np.array([BASE_CODES[base] for base in sequence]).mean() * np.arange(1, HIDDEN + 1)


RECORDS = [("b0001", "ACGT"), ("b0002", "AAAA"), ("b0003", "GGTTC"), ("b0004", "T"), ("b0005", "CA")]


class InterruptingEmbedder:
    def __init__(self, failAfter):
        self.calls = 0
        self.failAfter = failAfter

    def __call__(self, sequence):
        self.calls += 1
        if self.calls > self.failAfter:
            raise KeyboardInterrupt
        return FakeEmbedder(sequence)


def test_mean_pool_averages_all_tokens():
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert MeanPool(tokens).tolist() == [3.0, 4.0]
    assert MeanPool(tokens).dtype == np.float32


def test_mean_pool_ignores_masked_padding():
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    assert MeanPool(tokens, mask=[1, 1, 0]).tolist() == [2.0, 3.0]


def test_ids_follow_contract(tmp_path):
    tsvPath = tmp_path / "Windows.tsv"
    tsvPath.write_text("seqId\tstart\tend\tstrand\tsequence\nNC_1\t1\t1000\t-\tACGT\n", encoding="utf-8")
    assert ReadRecords(tsvPath, WindowId) == [("NC_1:1-1000:-", "ACGT")]
    assert GeneId({"locusTag": "b0001"}) == "b0001"


def test_run_set_saves_aligned_matrix(tmp_path):
    shape = RunSet("genes", RECORDS, FakeEmbedder, tmp_path, OPTIONS)
    matrix = np.load(tmp_path / "GeneEmbeddings.npy")
    ids = (tmp_path / "GeneIds.txt").read_text(encoding="utf-8").split()
    assert shape == (5, HIDDEN)
    assert matrix.dtype == np.float32
    assert ids == [recordId for recordId, _ in RECORDS]
    for row, (_, sequence) in zip(matrix, RECORDS):
        np.testing.assert_allclose(row, ExpectedRow(sequence), rtol=1e-6)


def test_resume_after_interruption_gives_same_matrix(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        RunSet("genes", RECORDS, InterruptingEmbedder(failAfter=3), tmp_path / "resumed", OPTIONS)
    assert len(ListShards(tmp_path / "resumed" / "checkpoints" / "genes")) == 1
    resumedEmbedder = InterruptingEmbedder(failAfter=100)
    RunSet("genes", RECORDS, resumedEmbedder, tmp_path / "resumed", OPTIONS)
    RunSet("genes", RECORDS, FakeEmbedder, tmp_path / "clean", OPTIONS)
    assert resumedEmbedder.calls == 3
    np.testing.assert_array_equal(np.load(tmp_path / "resumed" / "GeneEmbeddings.npy"), np.load(tmp_path / "clean" / "GeneEmbeddings.npy"))


def test_assembly_uses_input_order_not_shard_order(tmp_path):
    RunSet("genes", RECORDS, FakeEmbedder, tmp_path, OPTIONS)
    ids, matrix = AssembleMatrix(list(reversed(RECORDS)), tmp_path / "checkpoints" / "genes")
    assert ids == [recordId for recordId, _ in reversed(RECORDS)]
    np.testing.assert_allclose(matrix[0], ExpectedRow(RECORDS[-1][1]), rtol=1e-6)


def test_limit_embeds_only_n_and_skips_assembly(tmp_path):
    assert RunSet("genes", RECORDS, FakeEmbedder, tmp_path, dict(OPTIONS, limit=2)) is None
    assert not (tmp_path / "GeneEmbeddings.npy").exists()
    with pytest.raises(ValueError, match="3 ids have no embedding"):
        AssembleMatrix(RECORDS, tmp_path / "checkpoints" / "genes")


def test_validate_rejects_nan_inf_duplicates_and_misalignment():
    matrix = np.ones((3, HIDDEN), dtype=np.float32)
    ValidateMatrix(["a", "b", "c"], matrix)
    with pytest.raises(ValueError, match="does not match"):
        ValidateMatrix(["a", "b"], matrix)
    with pytest.raises(ValueError, match="Duplicate"):
        ValidateMatrix(["a", "b", "a"], matrix)
    matrix[1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf, first id: b"):
        ValidateMatrix(["a", "b", "c"], matrix)


def test_nan_from_model_stops_run_before_saving(tmp_path):
    def NanEmbedder(sequence):
        return np.full((len(sequence), HIDDEN), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN"):
        RunSet("genes", RECORDS, NanEmbedder, tmp_path, OPTIONS)
    assert not (tmp_path / "GeneEmbeddings.npy").exists()


def test_checkpoints_from_another_layer_are_refused(tmp_path):
    CheckCheckpointInfo(tmp_path, {"model": "fake", "layer": "blocks.0"})
    with pytest.raises(SystemExit, match="blocks.0"):
        CheckCheckpointInfo(tmp_path, {"model": "fake", "layer": "blocks.1"})


def test_layer_is_required():
    with pytest.raises(SystemExit, match="#3"):
        ResolveLayer({"model": {"layer": None}}, None)
    assert ResolveLayer({"model": {"layer": None}}, "blocks.28.mlp.l3") == "blocks.28.mlp.l3"
    assert ResolveLayer({"model": {"layer": "blocks.20"}}, None) == "blocks.20"
