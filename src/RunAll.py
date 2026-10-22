import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from src.FetchData import LoadConfig

SPARK_ONLY_MESSAGE = (
    "ExtractEmbeddings has no output here and needs CUDA and evo2, which only the DGX Spark has. "
    "Run it on the Spark (see issue #21), copy data/embeddings/ back and rerun RunAll."
)


def BuildSteps(config):
    paths = config["paths"]
    embeddings = paths.get("embeddings", "data/embeddings")
    results = paths["results"]
    figures = paths["figures"]
    return [
        {"name": "FetchData", "outputs": [f"{paths['raw']}/manifest.json"]},
        {"name": "PrepareGenes", "outputs": [f"{paths['processed']}/Genes.tsv"]},
        {"name": "ExtractEmbeddings", "isSparkOnly": True,
         "outputs": [f"{embeddings}/GeneEmbeddings.npy", f"{embeddings}/WindowEmbeddings.npy"]},
        {"name": "TrainSae", "outputs": [f"{results}/sae/GeneActivations.npy"]},
        {"name": "Enrichment",
         "outputs": [f"{results}/enrichment/SaeEnrichment.tsv", f"{results}/enrichment/BaselineEnrichment.tsv"]},
        {"name": "Compare", "outputs": [f"{results}/Comparison.tsv", f"{results}/CandidateFeatures.tsv"]},
        {"name": "MakeFigures", "outputs": [
            f"{figures}/{name}" for name in
            ["PValueHistogram.png", "PValueHistogram.pdf", "FeatureTable.png", "FeatureTable.pdf", "FeatureTable.md"]
        ]},
    ]


def StepNames(steps):
    return [step["name"] for step in steps]


def HasOutputs(step, root):
    return all((Path(root) / output).exists() for output in step["outputs"])


def IsStepPending(step, index, root, force, fromIndex):
    if step["name"] == force:
        return True
    if fromIndex is not None and index >= fromIndex:
        return True
    return not HasOutputs(step, root)


def PlanSteps(steps, root, force, fromStep):
    fromIndex = StepNames(steps).index(fromStep) if fromStep else None
    return [step for index, step in enumerate(steps) if IsStepPending(step, index, root, force, fromIndex)]


def IsSparkAvailable():
    if importlib.util.find_spec("evo2") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available()


def RunModule(module):
    subprocess.run([sys.executable, "-m", module], check=True)


def ParseArguments(argv, stepNames):
    parser = argparse.ArgumentParser(description="Run the pipeline in order, skipping steps whose outputs exist.")
    parser.add_argument("--force", choices=stepNames, help="Rerun this step even if its outputs exist.")
    parser.add_argument("--from", dest="fromStep", choices=stepNames, help="Rerun this step and every later one.")
    return parser.parse_args(argv)


def Main(argv=None, config=None, root=".", runner=RunModule, isSparkAvailable=IsSparkAvailable):
    steps = BuildSteps(config or LoadConfig())
    arguments = ParseArguments(argv, StepNames(steps))
    for step in PlanSteps(steps, root, arguments.force, arguments.fromStep):
        if step.get("isSparkOnly") and not isSparkAvailable():
            raise SystemExit(SPARK_ONLY_MESSAGE)
        print(f"Running {step['name']}", flush=True)
        runner(f"src.{step['name']}")


if __name__ == "__main__":
    Main()
