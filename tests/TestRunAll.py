import pytest

from src.RunAll import BuildSteps, Main, PlanSteps, StepNames

CONFIG = {"paths": {"raw": "data/raw", "processed": "data/processed", "results": "results", "figures": "figures"}}


def TouchOutputs(root, step):
    for output in step["outputs"]:
        target = root / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")


def TouchSteps(root, names):
    for step in BuildSteps(CONFIG):
        if step["name"] in names:
            TouchOutputs(root, step)


class FakeRunner:
    def __init__(self):
        self.modules = []

    def __call__(self, module):
        self.modules.append(module)


def RunMain(tmp_path, argv, isSparkAvailable=False):
    runner = FakeRunner()
    Main(argv, config=CONFIG, root=tmp_path, runner=runner, isSparkAvailable=lambda: isSparkAvailable)
    return runner.modules


def TestStepsRunInPipelineOrder():
    assert StepNames(BuildSteps(CONFIG)) == [
        "FetchData", "PrepareGenes", "ExtractEmbeddings", "TrainSae", "Enrichment", "Compare", "MakeFigures",
    ]


def TestEmbeddingsPathFollowsConfig():
    config = {"paths": dict(CONFIG["paths"], embeddings="elsewhere")}
    extract = next(step for step in BuildSteps(config) if step["name"] == "ExtractEmbeddings")
    assert all(output.startswith("elsewhere/") for output in extract["outputs"])


def TestPlanSkipsStepsWhoseOutputsExist(tmp_path):
    steps = BuildSteps(CONFIG)
    TouchSteps(tmp_path, {"FetchData", "PrepareGenes"})
    plan = PlanSteps(steps, tmp_path, force=None, fromStep=None)
    assert [step["name"] for step in plan] == ["ExtractEmbeddings", "TrainSae", "Enrichment", "Compare", "MakeFigures"]


def TestPlanRunsStepWithOnlySomeOutputs(tmp_path):
    steps = BuildSteps(CONFIG)
    enrichment = next(step for step in steps if step["name"] == "Enrichment")
    TouchSteps(tmp_path, set(StepNames(steps)))
    (tmp_path / enrichment["outputs"][0]).unlink()
    assert [step["name"] for step in PlanSteps(steps, tmp_path, None, None)] == ["Enrichment"]


def TestForceRerunsOnlyThatStep(tmp_path):
    steps = BuildSteps(CONFIG)
    TouchSteps(tmp_path, set(StepNames(steps)))
    assert [step["name"] for step in PlanSteps(steps, tmp_path, "Compare", None)] == ["Compare"]


def TestFromRerunsThatStepAndEveryLaterOne(tmp_path):
    steps = BuildSteps(CONFIG)
    TouchSteps(tmp_path, set(StepNames(steps)))
    assert [step["name"] for step in PlanSteps(steps, tmp_path, None, "Enrichment")] == ["Enrichment", "Compare", "MakeFigures"]


def TestMainRunsEachPendingModule(tmp_path):
    TouchSteps(tmp_path, {"FetchData", "PrepareGenes", "ExtractEmbeddings"})
    assert RunMain(tmp_path, []) == ["src.TrainSae", "src.Enrichment", "src.Compare", "src.MakeFigures"]


def TestMainStopsBeforeEmbeddingsWithoutSpark(tmp_path, capsys):
    TouchSteps(tmp_path, {"FetchData", "PrepareGenes"})
    runner = FakeRunner()
    with pytest.raises(SystemExit) as stop:
        Main([], config=CONFIG, root=tmp_path, runner=runner, isSparkAvailable=lambda: False)
    assert "#21" in str(stop.value)
    assert runner.modules == []


def TestMainRunsEmbeddingsOnSpark(tmp_path):
    TouchSteps(tmp_path, {"FetchData", "PrepareGenes"})
    assert RunMain(tmp_path, [], isSparkAvailable=True)[0] == "src.ExtractEmbeddings"


def TestMainRejectsUnknownStep(tmp_path):
    with pytest.raises(SystemExit):
        RunMain(tmp_path, ["--force", "Nope"])
