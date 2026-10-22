import numpy as np


def NormalizeDecoder(parameters):
    decoder = parameters["decoder"]
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)


def InitParameters(inputDim, features, rng, dtype=np.float32):
    decoder = rng.normal(size=(features, inputDim)).astype(dtype)
    parameters = {
        "preBias": np.zeros(inputDim, dtype=dtype),
        "encoder": np.empty((inputDim, features), dtype=dtype),
        "encoderBias": np.zeros(features, dtype=dtype),
        "decoder": decoder,
    }
    NormalizeDecoder(parameters)
    # Encoder starts as the decoder transpose, following Gao et al. 2024.
    parameters["encoder"][:] = parameters["decoder"].T
    return parameters


def KeepTopK(preactivations, k):
    topIndices = np.argpartition(preactivations, -k, axis=1)[:, -k:]
    topValues = np.take_along_axis(preactivations, topIndices, axis=1)
    codes = np.zeros_like(preactivations)
    np.put_along_axis(codes, topIndices, np.maximum(topValues, 0), axis=1)
    return codes


def Forward(parameters, x, k):
    centered = x - parameters["preBias"]
    preactivations = centered @ parameters["encoder"] + parameters["encoderBias"]
    codes = KeepTopK(preactivations, k)
    reconstruction = codes @ parameters["decoder"] + parameters["preBias"]
    return {"centered": centered, "codes": codes, "reconstruction": reconstruction}


def Encode(parameters, x, k):
    return Forward(parameters, x, k)["codes"]


def ReconstructionLoss(forward, x):
    return float(np.mean((forward["reconstruction"] - x) ** 2))


def Backward(parameters, forward, x):
    residual = forward["reconstruction"] - x
    gradOutput = 2 * residual / residual.size
    gradPreactivations = (gradOutput @ parameters["decoder"].T) * (forward["codes"] > 0)
    gradCentered = gradPreactivations @ parameters["encoder"].T
    return {
        "decoder": forward["codes"].T @ gradOutput,
        "preBias": gradOutput.sum(axis=0) - gradCentered.sum(axis=0),
        "encoder": forward["centered"].T @ gradPreactivations,
        "encoderBias": gradPreactivations.sum(axis=0),
    }


def ProjectDecoderGradient(gradient, decoder):
    parallel = (gradient * decoder).sum(axis=1, keepdims=True)
    return gradient - parallel * decoder
