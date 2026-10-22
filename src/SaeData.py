import json
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDINGS_DIR = "data/embeddings"


def FitStandardization(train):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def ApplyStandardization(x, mean, std):
    return (x - mean) / std


def SplitRows(rowCount, validationFraction, seed):
    order = np.random.default_rng(seed).permutation(rowCount)
    validationCount = int(round(rowCount * validationFraction))
    return np.sort(order[validationCount:]), np.sort(order[:validationCount])


def MakeSparseDictionaryData(rows, inputDim, atoms, active, rng):
    dictionary = rng.normal(size=(atoms, inputDim))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    coefficients = np.zeros((rows, atoms))
    chosen = np.argsort(rng.random((rows, atoms)), axis=1)[:, :active]
    np.put_along_axis(coefficients, chosen, rng.uniform(0.5, 2.0, size=(rows, active)), axis=1)
    return coefficients @ dictionary


def EmbeddingsDir(config):
    return Path(config["paths"].get("embeddings", DEFAULT_EMBEDDINGS_DIR))


def LoadEmbeddings(config, fileName):
    return np.load(EmbeddingsDir(config) / fileName).astype(np.float32, copy=False)


def SaeDir(config):
    return Path(config["paths"]["results"]) / "sae"


def SaveSaeOutputs(config, result, geneActivations):
    saeDir = SaeDir(config)
    saeDir.mkdir(parents=True, exist_ok=True)
    np.savez(saeDir / "SaeParameters.npz", **result["parameters"],
             inputMean=result["inputMean"], inputStd=result["inputStd"], k=result["metrics"]["k"])
    np.save(saeDir / "GeneActivations.npy", geneActivations.astype(np.float32))
    with open(saeDir / "SaeMetrics.json", "w", encoding="utf-8") as metricsFile:
        json.dump(result["metrics"], metricsFile, indent=2)
