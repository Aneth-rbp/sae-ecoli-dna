import numpy as np

BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


def InitAdam(parameters):
    return {
        "step": 0,
        "firstMoment": {name: np.zeros_like(value) for name, value in parameters.items()},
        "secondMoment": {name: np.zeros_like(value) for name, value in parameters.items()},
    }


def UpdateMoments(state, name, gradient):
    firstMoment = state["firstMoment"][name]
    secondMoment = state["secondMoment"][name]
    firstMoment *= BETA1
    firstMoment += (1 - BETA1) * gradient
    secondMoment *= BETA2
    secondMoment += (1 - BETA2) * np.square(gradient)
    return firstMoment, secondMoment


def AdamStep(parameters, gradients, state, learningRate):
    state["step"] += 1
    stepSize = learningRate * np.sqrt(1 - BETA2 ** state["step"]) / (1 - BETA1 ** state["step"])
    for name, gradient in gradients.items():
        firstMoment, secondMoment = UpdateMoments(state, name, gradient)
        update = np.sqrt(secondMoment)
        update += EPSILON
        np.divide(firstMoment, update, out=update)
        update *= stepSize
        parameters[name] -= update
