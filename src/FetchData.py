import argparse
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


def LoadManifest(rawDir):
    manifestPath = rawDir / MANIFEST_NAME
    if not manifestPath.exists():
        return {}
    with open(manifestPath, encoding="utf-8") as manifestFile:
        return json.load(manifestFile)


def SelectSources(sources, names):
    if not names:
        return sources
    unknown = [name for name in names if name not in sources]
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}. Known: {', '.join(sources)}")
    return {name: sources[name] for name in names}


def WriteManifest(manifest, rawDir):
    with open(rawDir / MANIFEST_NAME, "w", encoding="utf-8") as manifestFile:
        json.dump(manifest, manifestFile, indent=2)


def ReportCounts(sources, rawDir):
    if "genomeGff" in sources:
        print(f"CDS in GFF: {CountGffCds(rawDir / sources['genomeGff']['file'])}")
    if "uniprotProteome" in sources:
        print(f"UniProt entries: {CountUniprotEntries(rawDir / sources['uniprotProteome']['file'])}")


def ParseArguments(argv=None):
    parser = argparse.ArgumentParser(description="Download the raw data listed in Config.yaml.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Config.yaml source to download (repeatable); all sources when omitted.",
    )
    return parser.parse_args(argv)


def Main(argv=None):
    arguments = ParseArguments(argv)
    config = LoadConfig()
    rawDir = Path(config["paths"]["raw"])
    rawDir.mkdir(parents=True, exist_ok=True)
    sources = SelectSources(config["sources"], arguments.sources)
    manifest = LoadManifest(rawDir)
    manifest.update({name: FetchSource(source, rawDir) for name, source in sources.items()})
    WriteManifest(manifest, rawDir)
    ReportCounts(sources, rawDir)


if __name__ == "__main__":
    Main()
