import gzip
import hashlib
import json

import pytest

from src.FetchData import (
    BuildManifestEntry,
    CountGffCds,
    CountUniprotEntries,
    LoadManifest,
    ReadOboVersion,
    SelectSources,
    Sha256File,
)

SOURCES = {"genomeGff": {"file": "a.gff.gz"}, "uniprotProteome": {"file": "b.tsv.gz"}}


def TestSha256FileMatchesHashlib(tmp_path):
    filePath = tmp_path / "sample.txt"
    filePath.write_bytes(b"ACGT" * 1000)
    assert Sha256File(filePath) == hashlib.sha256(b"ACGT" * 1000).hexdigest()


def TestCountGffCdsReadsGzipAndSkipsOtherFeatures(tmp_path):
    lines = [
        "##gff-version 3",
        "NC_000913.3\tRefSeq\tgene\t190\t255\t.\t+\t.\tID=gene-b0001",
        "NC_000913.3\tRefSeq\tCDS\t190\t255\t.\t+\t0\tID=cds-1",
        "NC_000913.3\tRefSeq\tCDS\t337\t2799\t.\t+\t0\tID=cds-2",
        "NC_000913.3\tRefSeq\ttRNA\t3000\t3076\t.\t-\t.\tID=rna-1",
    ]
    gffPath = tmp_path / "genomic.gff.gz"
    with gzip.open(gffPath, "wt") as gffFile:
        gffFile.write("\n".join(lines) + "\n")
    assert CountGffCds(gffPath) == 2


def TestCountUniprotEntriesExcludesHeader(tmp_path):
    tsvPath = tmp_path / "proteome.tsv.gz"
    with gzip.open(tsvPath, "wt") as tsvFile:
        tsvFile.write("Entry\tOrdered locus names\tGene Ontology IDs\n")
        tsvFile.write("P0A7B8\tb3931\tGO:0005737\n")
        tsvFile.write("P0A6F5\tb4143\tGO:0005524; GO:0006457\n")
    assert CountUniprotEntries(tsvPath) == 2


def TestReadOboVersionReturnsDataVersion(tmp_path):
    oboPath = tmp_path / "go-basic.obo"
    oboPath.write_text("format-version: 1.2\ndata-version: releases/2026-10-01\n\n[Term]\n")
    assert ReadOboVersion(oboPath) == "releases/2026-10-01"


def TestBuildManifestEntryHasRequiredFields(tmp_path):
    filePath = tmp_path / "file.bin"
    filePath.write_bytes(b"x")
    entry = BuildManifestEntry("https://example.org/file.bin", filePath, "v1")
    assert entry["url"] == "https://example.org/file.bin"
    assert entry["version"] == "v1"
    assert entry["sha256"] == hashlib.sha256(b"x").hexdigest()
    assert entry["file"] == "file.bin"
    assert "downloadedAt" in entry


def TestSelectSourcesReturnsAllWhenNoneNamed():
    assert SelectSources(SOURCES, None) == SOURCES


def TestSelectSourcesKeepsOnlyNamed():
    assert SelectSources(SOURCES, ["uniprotProteome"]) == {"uniprotProteome": {"file": "b.tsv.gz"}}


def TestSelectSourcesRejectsUnknownName():
    with pytest.raises(ValueError, match="uniprot"):
        SelectSources(SOURCES, ["uniprot"])


def TestLoadManifestIsEmptyWithoutFile(tmp_path):
    assert LoadManifest(tmp_path) == {}


def TestLoadManifestKeepsExistingEntries(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"goBasic": {"version": "v1"}}), encoding="utf-8")
    assert LoadManifest(tmp_path) == {"goBasic": {"version": "v1"}}
