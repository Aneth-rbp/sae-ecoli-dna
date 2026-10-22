import argparse
import json
import time

import numpy as np

from src.AdamOptimizer import AdamStep, InitAdam
from src.FetchData import LoadConfig
from src.SaeData import (
    ApplyStandardization,
    FitStandardization,
    LoadEmbeddings,
    MakeSparseDictionaryData,
    SaveSaeOutputs,
    SplitRows,
)
from src.SparseAutoencoder import Backward, Encode, Forward, InitParameters, NormalizeDecoder, ProjectDecoderGradient

DEAD_DEFINITION = "feature never active on the validation split"
SETTING_KEYS = ["features", "epochs", "batchSize", "learningRate", "validationFraction"]
SUMMARY_KEYS = ["validationMse", "explainedVariance", "l0", "deadFraction"]
SYNTHETIC_ROWS = 18564
SYNTHETIC_DIM = 4096


def BuildSettings(config, k):
    settings = {key: config["sae"][key] for key in SETTING_KEYS}
    settings["k"] = k
    settings["seed"] = config["seed"]
    return settings


def TrainStep(parameters, batch, k, optimizer, learningRate):
    gradients = Backward(parameters, Forward(parameters, batch, k), batch)
    gradients["decoder"] = ProjectDecoderGradient(gradients["decoder"], parameters["decoder"])
    AdamStep(parameters, gradients, optimizer, learningRate)
    NormalizeDecoder(parameters)


def TrainEpochs(parameters, train, settings, rng):
    optimizer = InitAdam(parameters)
    batchSize = settings["batchSize"]
    for _ in range(settings["epochs"]):
        order = rng.permutation(len(train))
        for start in range(0, len(train), batchSize):
            TrainStep(parameters, train[order[start:start + batchSize]], settings["k"], optimizer,
                      settings["learningRate"])


def EvaluateSae(parameters, x, k):
    forward = Forward(parameters, x, k)
    residual = forward["reconstruction"] - x
    totalVariance = float(((x - x.mean(axis=0)) ** 2).sum())
    activeCounts = (forward["codes"] > 0).sum(axis=1)
    return {
        "validationMse": float(np.mean(residual ** 2)),
        "explainedVariance": 1 - float((residual ** 2).sum()) / totalVariance,
        "l0": float(activeCounts.mean()),
        "deadFraction": float(np.mean(~(forward["codes"] > 0).any(axis=0))),
    }


def StandardizeSplits(data, settings):
    trainRows, validationRows = SplitRows(len(data), settings["validationFraction"], settings["seed"])
    inputMean, inputStd = FitStandardization(data[trainRows])
    train = ApplyStandardization(data[trainRows], inputMean, inputStd).astype(data.dtype)
    validation = ApplyStandardization(data[validationRows], inputMean, inputStd).astype(data.dtype)
    return train, validation, inputMean, inputStd


def BuildMetrics(settings, evaluation, train, validation, seconds):
    return {"k": settings["k"], "features": settings["features"], **evaluation, "seed": settings["seed"],
            "deadDefinition": DEAD_DEFINITION, "inputDim": int(train.shape[1]), "trainRows": len(train),
            "validationRows": len(validation), "trainingSeconds": round(seconds, 1)}


def RunTraining(data, settings):
    train, validation, inputMean, inputStd = StandardizeSplits(data, settings)
    rng = np.random.default_rng(settings["seed"])
    parameters = InitParameters(data.shape[1], settings["features"], rng, data.dtype)
    started = time.perf_counter()
    TrainEpochs(parameters, train, settings, rng)
    evaluation = EvaluateSae(parameters, validation, settings["k"])
    metrics = BuildMetrics(settings, evaluation, train, validation, time.perf_counter() - started)
    return {"parameters": parameters, "inputMean": inputMean, "inputStd": inputStd, "metrics": metrics}


def MeetsRule(metrics, minExplainedVariance, maxDeadFraction):
    return metrics["explainedVariance"] >= minExplainedVariance and metrics["deadFraction"] < maxDeadFraction


def SelectK(candidates, minExplainedVariance, maxDeadFraction):
    passing = [k for k, metrics in candidates.items() if MeetsRule(metrics, minExplainedVariance, maxDeadFraction)]
    return min(passing) if passing else None


def CandidateSummary(results):
    return {str(k): {key: result["metrics"][key] for key in SUMMARY_KEYS} for k, result in results.items()}


def TrainWithAutoK(config, windows):
    sae = config["sae"]
    results = {k: RunTraining(windows, BuildSettings(config, k)) for k in sae["kCandidates"]}
    chosenK = SelectK({k: result["metrics"] for k, result in results.items()},
                      sae["minExplainedVariance"], sae["maxDeadFraction"])
    if chosenK is None:
        raise SystemExit("No k candidate meets the rule: " + json.dumps(CandidateSummary(results), indent=2))
    chosen = results[chosenK]
    chosen["metrics"].update({"kSelection": "auto", "kCandidates": CandidateSummary(results),
                              "minExplainedVariance": sae["minExplainedVariance"],
                              "maxDeadFraction": sae["maxDeadFraction"]})
    return chosen


def TrainForK(config, windows, kOption):
    if kOption == "auto":
        return TrainWithAutoK(config, windows)
    result = RunTraining(windows, BuildSettings(config, int(kOption)))
    result["metrics"]["kSelection"] = "fixed"
    return result


def TrainSae(config, kOption):
    result = TrainForK(config, LoadEmbeddings(config, "WindowEmbeddings.npy"), kOption)
    genes = ApplyStandardization(LoadEmbeddings(config, "GeneEmbeddings.npy"), result["inputMean"], result["inputStd"])
    geneActivations = Encode(result["parameters"], genes.astype(np.float32), result["metrics"]["k"])
    SaveSaeOutputs(config, result, geneActivations)
    return result["metrics"]


def TrainSynthetic(config, kOption, rows, inputDim):
    k = config["sae"]["kCandidates"][0] if kOption == "auto" else int(kOption)
    settings = BuildSettings(config, k)
    rng = np.random.default_rng(config["seed"])
    data = MakeSparseDictionaryData(rows, inputDim, settings["features"], k, rng).astype(np.float32)
    return RunTraining(data, settings)["metrics"]


def ParseArguments():
    parser = argparse.ArgumentParser(description="Train the TopK sparse autoencoder on window embeddings.")
    parser.add_argument("--k", default=None, help="Number of active features, or 'auto' to apply the k rule.")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic data and write nothing.")
    parser.add_argument("--rows", type=int, default=SYNTHETIC_ROWS)
    parser.add_argument("--dim", type=int, default=SYNTHETIC_DIM)
    return parser.parse_args()


def Main():
    arguments = ParseArguments()
    config = LoadConfig()
    kOption = arguments.k or str(config["sae"]["k"] or "auto")
    if arguments.synthetic:
        metrics = TrainSynthetic(config, kOption, arguments.rows, arguments.dim)
    else:
        metrics = TrainSae(config, kOption)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    Main()
