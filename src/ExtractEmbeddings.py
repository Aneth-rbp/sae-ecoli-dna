import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from src.FetchData import LoadConfig, Sha256File

DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
DEFAULT_CHECKPOINT_EVERY = 256
CHECKPOINT_INFO_NAME = "CheckpointInfo.json"
MANIFEST_NAME = "EmbeddingsManifest.json"
RUNTIME_PACKAGES = ("torch", "evo2", "flash-attn")
SHARD_PATTERN = "Shard*.npz"
SETS = {
    "genes": {"tsv": "Genes.tsv", "prefix": "Gene"},
    "windows": {"tsv": "Windows.tsv", "prefix": "Window"},
}


def GeneId(row):
    return row["locusTag"]


def WindowId(row):
    return f"{row['seqId']}:{row['start']}-{row['end']}:{row['strand']}"


ID_BUILDERS = {"genes": GeneId, "windows": WindowId}


def ReadRecords(tsvPath, idBuilder):
    with open(tsvPath, encoding="utf-8", newline="") as tsvFile:
        return [(idBuilder(row), row["sequence"]) for row in csv.DictReader(tsvFile, delimiter="\t")]


def ResolveLayer(config, layerArgument):
    layer = layerArgument or config["model"].get("layer")
    if not layer:
        raise SystemExit(
            "model.layer is null in Config.yaml; the layer is fixed in issue #3. "
            "Pass --layer (for example blocks.28.mlp.l3) to extract before then."
        )
    return layer


def MeanPool(tokenEmbeddings, mask=None):
    tokens = np.asarray(tokenEmbeddings, dtype=np.float32)
    if mask is None:
        return tokens.mean(axis=0)
    weights = np.asarray(mask, dtype=np.float32)
    return (tokens * weights[:, None]).sum(axis=0) / weights.sum()


def LoadEvo2Embedder(modelName, layerName, checkpointPath=None):
    import torch
    from evo2 import Evo2

    model = Evo2(modelName, local_path=checkpointPath)

    def Embed(sequence):
        inputIds = torch.tensor(model.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0).to("cuda:0")
        _, embeddings = model(inputIds, return_embeddings=True, layer_names=[layerName])
        return embeddings[layerName][0].float().cpu().numpy()

    return Embed


def CheckCheckpointInfo(checkpointDir, info):
    infoPath = checkpointDir / CHECKPOINT_INFO_NAME
    if not infoPath.exists():
        checkpointDir.mkdir(parents=True, exist_ok=True)
        infoPath.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return
    stored = json.loads(infoPath.read_text(encoding="utf-8"))
    if stored != info:
        raise SystemExit(f"Checkpoints in {checkpointDir} were made with {stored}, not {info}; move them away to start over.")


def ListShards(checkpointDir):
    return sorted(checkpointDir.glob(SHARD_PATTERN))


def ReadShard(shardPath):
    with np.load(shardPath) as shard:
        return [str(shardId) for shardId in shard["ids"]], shard["embeddings"]


def ReadDoneIds(checkpointDir):
    doneIds = set()
    for shardPath in ListShards(checkpointDir):
        doneIds.update(ReadShard(shardPath)[0])
    return doneIds


def ShardSeconds(checkpointDir):
    total = 0.0
    for shardPath in ListShards(checkpointDir):
        with np.load(shardPath) as shard:
            total += float(shard["seconds"]) if "seconds" in shard.files else 0.0
    return total


def WriteShard(checkpointDir, ids, matrix, seconds=0.0):
    shardPath = checkpointDir / f"Shard{len(ListShards(checkpointDir)):05d}.npz"
    temporaryPath = shardPath.with_suffix(".tmp")
    with open(temporaryPath, "wb") as shardFile:
        np.savez(shardFile, ids=np.array(ids), embeddings=matrix.astype(np.float32), seconds=np.float64(seconds))
    # Rename last so an interrupted write never leaves a half shard behind.
    os.replace(temporaryPath, shardPath)


def SplitBatches(records, batchSize):
    return [records[start:start + batchSize] for start in range(0, len(records), batchSize)]


def EmbedBatch(batch, embedder):
    return np.stack([MeanPool(embedder(sequence)) for _, sequence in batch])


def ExtractToCheckpoints(records, embedder, checkpointDir, checkpointEvery):
    for batch in SplitBatches(records, checkpointEvery):
        startTime = time.perf_counter()
        matrix = EmbedBatch(batch, embedder)
        WriteShard(checkpointDir, [recordId for recordId, _ in batch], matrix, time.perf_counter() - startTime)
        print(f"{checkpointDir.name}: {len(ListShards(checkpointDir))} shards written", flush=True)


def PendingRecords(records, checkpointDir, limit=None):
    doneIds = ReadDoneIds(checkpointDir)
    pending = [record for record in records if record[0] not in doneIds]
    return pending if limit is None else pending[:limit]


def AssembleMatrix(records, checkpointDir):
    rowsById = {}
    for shardPath in ListShards(checkpointDir):
        ids, matrix = ReadShard(shardPath)
        rowsById.update(zip(ids, matrix))
    missing = [recordId for recordId, _ in records if recordId not in rowsById]
    if missing:
        raise ValueError(f"{len(missing)} ids have no embedding yet, first: {missing[0]}")
    ids = [recordId for recordId, _ in records]
    return ids, np.stack([rowsById[recordId] for recordId in ids]).astype(np.float32)


def ValidateMatrix(ids, matrix):
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError(f"Matrix shape {matrix.shape} does not match {len(ids)} ids")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ids in the embedding matrix")
    badRows = np.flatnonzero(~np.isfinite(matrix).all(axis=1))
    if badRows.size:
        raise ValueError(f"{badRows.size} rows contain NaN or Inf, first id: {ids[badRows[0]]}")


def SaveMatrix(embeddingsDir, prefix, ids, matrix):
    embeddingsDir.mkdir(parents=True, exist_ok=True)
    np.save(embeddingsDir / f"{prefix}Embeddings.npy", matrix.astype(np.float32))
    (embeddingsDir / f"{prefix}Ids.txt").write_text("".join(f"{recordId}\n" for recordId in ids), encoding="utf-8")


def SetSummary(embeddingsDir, setName, shape):
    prefix = SETS[setName]["prefix"]
    files = [f"{prefix}Embeddings.npy", f"{prefix}Ids.txt"]
    return {
        "rows": shape[0],
        "columns": shape[1],
        "embeddingSeconds": round(ShardSeconds(embeddingsDir / "checkpoints" / setName), 1),
        "sha256": {fileName: Sha256File(embeddingsDir / fileName) for fileName in files},
    }


def PackageVersion(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def GitCommit():
    # safe.directory: on the Spark the repo is mounted into a container that runs as another user.
    try:
        result = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def WriteManifest(embeddingsDir, info, summaries, runtime):
    manifestPath = embeddingsDir / MANIFEST_NAME
    manifest = json.loads(manifestPath.read_text(encoding="utf-8")) if manifestPath.exists() else {"sets": {}}
    if {key: manifest.get(key) for key in info} != info:
        manifest = {"sets": {}}
    manifest.update(info)
    manifest.update(runtime)
    manifest["sets"].update(summaries)
    manifestPath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def RuntimeInfo():
    return {
        "commit": GitCommit(),
        "versions": {package: PackageVersion(package) for package in RUNTIME_PACKAGES},
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def RunSet(setName, records, embedder, embeddingsDir, options):
    checkpointDir = embeddingsDir / "checkpoints" / setName
    CheckCheckpointInfo(checkpointDir, options["info"])
    pending = PendingRecords(records, checkpointDir, options.get("limit"))
    startTime = time.perf_counter()
    ExtractToCheckpoints(pending, embedder, checkpointDir, options["checkpointEvery"])
    if pending:
        print(f"{setName}: {len(pending)} sequences, {(time.perf_counter() - startTime) / len(pending):.3f} s per sequence")
    if options.get("limit") is not None:
        return None
    ids, matrix = AssembleMatrix(records, checkpointDir)
    ValidateMatrix(ids, matrix)
    SaveMatrix(embeddingsDir, SETS[setName]["prefix"], ids, matrix)
    print(f"{setName}: saved {matrix.shape[0]} x {matrix.shape[1]}")
    return matrix.shape


def ParseArguments(argv=None):
    parser = argparse.ArgumentParser(description="Mean-pooled Evo 2 embeddings for genes and genomic windows.")
    parser.add_argument("--set", choices=["genes", "windows", "all"], default="all")
    parser.add_argument("--layer", default=None, help="Overrides model.layer from Config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Embed only N pending sequences and skip assembly.")
    parser.add_argument("--checkpoint", default=None, help="Overrides model.checkpointPath from Config.yaml.")
    return parser.parse_args(argv)


def BuildOptions(config, layer, limit):
    model = config["model"]
    return {
        "info": {"model": model["name"], "layer": layer},
        "checkpointEvery": model.get("checkpointEvery", DEFAULT_CHECKPOINT_EVERY),
        "limit": limit,
    }


def Main(argv=None):
    arguments = ParseArguments(argv)
    config = LoadConfig()
    layer = ResolveLayer(config, arguments.layer)
    processedDir = Path(config["paths"]["processed"])
    embeddingsDir = Path(config["paths"].get("embeddings", DEFAULT_EMBEDDINGS_DIR))
    setNames = list(SETS) if arguments.set == "all" else [arguments.set]
    options = BuildOptions(config, layer, arguments.limit)
    checkpointPath = arguments.checkpoint or config["model"].get("checkpointPath")
    embedder = LoadEvo2Embedder(config["model"]["name"], layer, checkpointPath)
    summaries = {}
    for setName in setNames:
        records = ReadRecords(processedDir / SETS[setName]["tsv"], ID_BUILDERS[setName])
        shape = RunSet(setName, records, embedder, embeddingsDir, options)
        if shape is not None:
            summaries[setName] = SetSummary(embeddingsDir, setName, shape)
    if summaries:
        WriteManifest(embeddingsDir, options["info"], summaries, RuntimeInfo())


if __name__ == "__main__":
    Main()
