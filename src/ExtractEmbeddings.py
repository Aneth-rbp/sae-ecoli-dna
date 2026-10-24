import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from src.FetchData import LoadConfig

DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
DEFAULT_CHECKPOINT_EVERY = 256
CHECKPOINT_INFO_NAME = "CheckpointInfo.json"
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


def WriteShard(checkpointDir, ids, matrix):
    shardPath = checkpointDir / f"Shard{len(ListShards(checkpointDir)):05d}.npz"
    temporaryPath = shardPath.with_suffix(".tmp")
    with open(temporaryPath, "wb") as shardFile:
        np.savez(shardFile, ids=np.array(ids), embeddings=matrix.astype(np.float32))
    # Rename last so an interrupted write never leaves a half shard behind.
    os.replace(temporaryPath, shardPath)


def SplitBatches(records, batchSize):
    return [records[start:start + batchSize] for start in range(0, len(records), batchSize)]


def EmbedBatch(batch, embedder):
    return np.stack([MeanPool(embedder(sequence)) for _, sequence in batch])


def ExtractToCheckpoints(records, embedder, checkpointDir, checkpointEvery):
    for batch in SplitBatches(records, checkpointEvery):
        WriteShard(checkpointDir, [recordId for recordId, _ in batch], EmbedBatch(batch, embedder))
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
    for setName in setNames:
        records = ReadRecords(processedDir / SETS[setName]["tsv"], ID_BUILDERS[setName])
        RunSet(setName, records, embedder, embeddingsDir, options)


if __name__ == "__main__":
    Main()
