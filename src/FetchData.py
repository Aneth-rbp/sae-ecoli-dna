import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

CHUNK_BYTES = 1 << 20
MANIFEST_NAME = "manifest.json"


def LoadConfig(configPath="Config.yaml"):
    with open(configPath, encoding="utf-8") as configFile:
        return yaml.safe_load(configFile)


def OpenText(filePath):
    if str(filePath).endswith(".gz"):
        return gzip.open(filePath, "rt", encoding="utf-8")
    return open(filePath, encoding="utf-8")


def Sha256File(filePath):
    digest = hashlib.sha256()
    with open(filePath, "rb") as binaryFile:
        for chunk in iter(lambda: binaryFile.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def CountGffCds(gffPath):
    with OpenText(gffPath) as gffFile:
        return sum(1 for line in gffFile if IsCdsLine(line))


def IsCdsLine(line):
    columns = line.split("\t")
    return len(columns) > 2 and columns[2] == "CDS"


def CountUniprotEntries(tsvPath):
    with OpenText(tsvPath) as tsvFile:
        return sum(1 for line in tsvFile if line.strip()) - 1


def ReadOboVersion(oboPath):
    with OpenText(oboPath) as oboFile:
        for line in oboFile:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("[Term]"):
                break
    return None


def BuildManifestEntry(url, filePath, version):
    return {
        "url": url,
        "file": Path(filePath).name,
        "version": version,
        "sha256": Sha256File(filePath),
        "downloadedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def DownloadFile(url, targetPath):
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(targetPath, "wb") as targetFile:
            for chunk in response.iter_content(CHUNK_BYTES):
                targetFile.write(chunk)
        return dict(response.headers)


def ResolveVersion(source, headers, filePath):
    if source.get("versionHeader"):
        return headers.get(source["versionHeader"])
    if source.get("versionFromObo"):
        return ReadOboVersion(filePath)
    return source.get("version")


def FetchSource(source, rawDir):
    targetPath = rawDir / source["file"]
    print(f"Downloading {source['file']}")
    headers = DownloadFile(source["url"], targetPath)
    version = ResolveVersion(source, headers, targetPath)
    return BuildManifestEntry(source["url"], targetPath, version)


def WriteManifest(manifest, rawDir):
    with open(rawDir / MANIFEST_NAME, "w", encoding="utf-8") as manifestFile:
        json.dump(manifest, manifestFile, indent=2)


def ReportCounts(sources, rawDir):
    cdsCount = CountGffCds(rawDir / sources["genomeGff"]["file"])
    uniprotCount = CountUniprotEntries(rawDir / sources["uniprotProteome"]["file"])
    print(f"CDS in GFF: {cdsCount}")
    print(f"UniProt entries: {uniprotCount}")


def Main():
    config = LoadConfig()
    rawDir = Path(config["paths"]["raw"])
    rawDir.mkdir(parents=True, exist_ok=True)
    sources = config["sources"]
    manifest = {name: FetchSource(source, rawDir) for name, source in sources.items()}
    WriteManifest(manifest, rawDir)
    ReportCounts(sources, rawDir)


if __name__ == "__main__":
    Main()
