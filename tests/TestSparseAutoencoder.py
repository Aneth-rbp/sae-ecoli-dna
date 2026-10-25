import json

import numpy as np
import pytest

from src.AdamOptimizer import AdamStep, InitAdam
from src.SaeData import ApplyStandardization, FitStandardization, MakeSparseDictionaryData, SplitRows
from src.SparseAutoencoder import (
    Backward,
    Encode,
    Forward,
    InitParameters,
    NormalizeDecoder,
    ProjectDecoderGradient,
    ReconstructionLoss,
)
from src.TrainSae import BuildSettings, EvaluateSae, RunTraining, SelectCombination, TrainSae, TrainWithAutoK

INPUT_DIM = 12
FEATURES = 24
K = 3


def MakeParameters(seed=0, dtype=np.float64):
    return InitParameters(INPUT_DIM, FEATURES, np.random.default_rng(seed), dtype)


def MakeSettings(k=3, epochs=40):
    return {"features": 48, "k": k, "epochs": epochs, "batchSize": 64, "learningRate": 0.01,
            "validationFraction": 0.2, "seed": 42}


def TestEncodeReturnsGenesByFeatures():
    x = np.random.default_rng(1).normal(size=(7, INPUT_DIM))
    codes = Encode(MakeParameters(), x, K)
    assert codes.shape == (7, FEATURES)


def TestEncodeKeepsExactlyKActiveWhenPreactivationsArePositive():
    parameters = MakeParameters()
    parameters["encoderBias"][:] = 100.0
    x = np.random.default_rng(2).normal(size=(9, INPUT_DIM))
    codes = Encode(parameters, x, K)
    assert np.all((codes > 0).sum(axis=1) == K)


def TestEncodeNeverExceedsKActiveAndIsNonNegative():
    x = np.random.default_rng(3).normal(size=(50, INPUT_DIM))
    codes = Encode(MakeParameters(), x, K)
    assert np.all((codes != 0).sum(axis=1) <= K)
    assert codes.min() >= 0


def TestForwardReconstructionHasInputShape():
    x = np.random.default_rng(4).normal(size=(5, INPUT_DIM))
    assert Forward(MakeParameters(), x, K)["reconstruction"].shape == x.shape


def TestDecoderColumnsHaveUnitNorm():
    parameters = MakeParameters()
    parameters["decoder"] *= 3.0
    NormalizeDecoder(parameters)
    np.testing.assert_allclose(np.linalg.norm(parameters["decoder"], axis=1), 1.0)


def TestProjectedDecoderGradientIsOrthogonalToColumns():
    parameters = MakeParameters()
    gradient = np.random.default_rng(5).normal(size=parameters["decoder"].shape)
    projected = ProjectDecoderGradient(gradient, parameters["decoder"])
    np.testing.assert_allclose((projected * parameters["decoder"]).sum(axis=1), 0.0, atol=1e-12)


def NumericGradient(parameters, x, name, epsilon=1e-6):
    numeric = np.zeros_like(parameters[name])
    for index in np.ndindex(parameters[name].shape):
        original = parameters[name][index]
        parameters[name][index] = original + epsilon
        plus = ReconstructionLoss(Forward(parameters, x, K), x)
        parameters[name][index] = original - epsilon
        minus = ReconstructionLoss(Forward(parameters, x, K), x)
        parameters[name][index] = original
        numeric[index] = (plus - minus) / (2 * epsilon)
    return numeric


@pytest.mark.parametrize("name", ["encoder", "encoderBias", "decoder", "preBias"])
def TestBackwardMatchesFiniteDifferences(name):
    parameters = MakeParameters(seed=6)
    parameters["preBias"] = np.random.default_rng(7).normal(scale=0.1, size=INPUT_DIM)
    x = np.random.default_rng(8).normal(size=(6, INPUT_DIM))
    gradients = Backward(parameters, Forward(parameters, x, K), x)
    np.testing.assert_allclose(gradients[name], NumericGradient(parameters, x, name), rtol=1e-4, atol=1e-8)


def TestAdamStepMovesAgainstGradient():
    parameters = {"weights": np.array([1.0, -1.0])}
    state = InitAdam(parameters)
    AdamStep(parameters, {"weights": np.array([2.0, -3.0])}, state, 0.1)
    np.testing.assert_allclose(parameters["weights"], [0.9, -0.9])


def TestStandardizationUsesTrainStatisticsAndGuardsConstantColumns():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    mean, std = FitStandardization(train)
    standardized = ApplyStandardization(np.array([[2.0, 7.0]]), mean, std)
    np.testing.assert_allclose(standardized, [[0.0, 2.0]])


def TestSplitRowsIsDisjointAndReproducible():
    trainRows, validationRows = SplitRows(100, 0.2, 42)
    assert len(validationRows) == 20 and len(trainRows) == 80
    assert not set(trainRows) & set(validationRows)
    np.testing.assert_array_equal(SplitRows(100, 0.2, 42)[1], validationRows)


def TestSaeReconstructsDataFromSparseDictionary():
    data = MakeSparseDictionaryData(1500, 24, 32, 3, np.random.default_rng(9))
    result = RunTraining(data, MakeSettings(k=3))
    assert result["metrics"]["explainedVariance"] > 0.9
    assert result["metrics"]["l0"] <= 3


def TestEvaluateCountsFeaturesNeverActiveAsDead():
    parameters = InitParameters(4, 6, np.random.default_rng(0), np.float64)
    parameters["encoder"][:, 5] = -1e6
    x = np.abs(np.random.default_rng(1).normal(size=(30, 4)))
    metrics = EvaluateSae(parameters, x, 2)
    assert metrics["deadFraction"] >= 1 / 6


def Candidate(features, k, explainedVariance, deadFraction):
    return {"features": features, "k": k, "explainedVariance": explainedVariance, "deadFraction": deadFraction}


def TestSelectCombinationPicksLargestPassingDictionaryThenSmallestK():
    candidates = [Candidate(64, 16, 0.85, 0.05), Candidate(64, 32, 0.9, 0.05),
                  Candidate(128, 16, 0.75, 0.1), Candidate(128, 32, 0.85, 0.1), Candidate(128, 64, 0.9, 0.15),
                  Candidate(256, 16, 0.95, 0.6), Candidate(256, 32, 0.97, 0.5)]
    assert SelectCombination(candidates, 0.8, 0.2) == (128, 32)


def TestSelectCombinationReturnsNoneWhenNothingQualifies():
    candidates = [Candidate(64, 16, 0.9, 0.5), Candidate(128, 16, 0.7, 0.1)]
    assert SelectCombination(candidates, 0.8, 0.2) is None


def TestBuildSettingsOverridesFeatures():
    config = {"seed": 1, "sae": {"features": 8, "validationFraction": 0.2, "epochs": 3, "batchSize": 16,
                                 "learningRate": 0.001}}
    assert BuildSettings(config, 4, 32)["features"] == 32


def TestBuildSettingsReadsSaeBlock():
    config = {"seed": 42, "sae": {"features": 8, "k": None, "validationFraction": 0.2, "epochs": 3,
                                  "batchSize": 16, "learningRate": 0.001}}
    assert BuildSettings(config, 16) == {"features": 8, "k": 16, "epochs": 3, "batchSize": 16,
                                         "learningRate": 0.001, "validationFraction": 0.2, "seed": 42}


def WriteEmbeddings(embeddingsDir, rng):
    embeddingsDir.mkdir(parents=True)
    np.save(embeddingsDir / "WindowEmbeddings.npy", MakeSparseDictionaryData(400, 8, 16, 2, rng).astype(np.float32))
    np.save(embeddingsDir / "GeneEmbeddings.npy", MakeSparseDictionaryData(30, 8, 16, 2, rng).astype(np.float32))


def TestTrainSaeWritesContractOutputs(tmp_path):
    WriteEmbeddings(tmp_path / "embeddings", np.random.default_rng(10))
    config = {"seed": 42, "paths": {"embeddings": str(tmp_path / "embeddings"), "results": str(tmp_path / "results")},
              "sae": {"features": 16, "k": None, "validationFraction": 0.2, "epochs": 5, "batchSize": 32,
                      "learningRate": 0.01, "featureCandidates": [16], "kCandidates": [2, 4],
                      "minExplainedVariance": 0.0, "maxDeadFraction": 1.0}}
    TrainSae(config, "auto")
    saeDir = tmp_path / "results" / "sae"
    metrics = json.loads((saeDir / "SaeMetrics.json").read_text(encoding="utf-8"))
    activations = np.load(saeDir / "GeneActivations.npy")
    assert metrics["k"] == 2 and metrics["features"] == 16 and metrics["seed"] == 42
    assert activations.shape == (30, 16) and activations.dtype == np.float32
    assert np.all((activations != 0).sum(axis=1) <= 2)
    assert {"encoder", "decoder", "inputMean", "inputStd"} <= set(np.load(saeDir / "SaeParameters.npz").files)


def SweepConfig(maxDeadFraction):
    return {"seed": 42, "sae": {"features": 99, "validationFraction": 0.2, "epochs": 3, "batchSize": 32,
                                "learningRate": 0.01, "featureCandidates": [8, 16], "kCandidates": [2, 4],
                                "minExplainedVariance": 0.0, "maxDeadFraction": maxDeadFraction}}


def TestAutoSweepTrainsEveryCombinationAndKeepsTheChosenOne():
    windows = MakeSparseDictionaryData(200, 8, 16, 2, np.random.default_rng(3)).astype(np.float32)
    metrics = TrainWithAutoK(SweepConfig(1.0), windows)["metrics"]
    assert [(c["features"], c["k"]) for c in metrics["candidates"]] == [(8, 2), (8, 4), (16, 2), (16, 4)]
    assert set(metrics["candidates"][0]) == {"features", "k", "validationMse", "explainedVariance", "l0",
                                             "deadFraction"}
    assert (metrics["features"], metrics["k"]) == (16, 2)


def TestAutoSweepStopsWithTheTableWhenNothingQualifies():
    windows = MakeSparseDictionaryData(200, 8, 16, 2, np.random.default_rng(3)).astype(np.float32)
    with pytest.raises(SystemExit, match="features"):
        TrainWithAutoK(SweepConfig(-1.0), windows)
