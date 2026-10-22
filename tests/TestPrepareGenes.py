import gzip

import pytest

from src.GenomeSequence import BuildWindows, ReadFasta, ResolveStrands, ReverseComplement
from src.GoOntology import BuildAltIdMap, CountTermGenes, ParseObo, PropagateTerms, SelectTermsBySize
from src.PrepareGenes import (
    BuildGene,
    FilterGeneTerms,
    ReadGffCds,
    ReadUniprotByLocus,
    SelectLongestCdsPerLocus,
)

GENOME = {"chr": "ATGAAACCCGGGTTTTAA" + "CCCC" + "TTACGTGCAGCAT"}
GFF_LINES = [
    "##gff-version 3",
    "chr\tRefSeq\tgene\t1\t18\t.\t+\t.\tID=gene-b0001;locus_tag=b0001",
    "chr\tRefSeq\tCDS\t1\t18\t.\t+\t0\tID=cds-1;gene=fwdA;locus_tag=b0001",
    "chr\tRefSeq\tCDS\t23\t35\t.\t-\t0\tID=cds-2;gene=revB;locus_tag=b0002;product=a%3Bb",
    "chr\tRefSeq\tCDS\t1\t6\t.\t+\t0\tID=cds-3;gene=fwdA;locus_tag=b0001",
    "chr\tRefSeq\tCDS\t1\t9\t.\t+\t0\tID=cds-4;gene=pseC;locus_tag=b0003;pseudo=true",
]
OBO_TEXT = """format-version: 1.2

[Term]
id: GO:0000001
name: root
namespace: biological_process

[Term]
id: GO:0000002
name: middle
namespace: biological_process
is_a: GO:0000001 ! root

[Term]
id: GO:0000003
name: leaf
namespace: biological_process
alt_id: GO:0000099
is_a: GO:0000002 ! middle
relationship: part_of GO:0000004 ! whole
relationship: regulates GO:0000005 ! regulated

[Term]
id: GO:0000004
name: whole
namespace: biological_process

[Term]
id: GO:0000005
name: regulated
namespace: biological_process

[Term]
id: GO:0000006
name: obsolete thing
namespace: biological_process
is_obsolete: true

[Typedef]
id: part_of
name: part of
"""


def WriteGzip(path, text):
    with gzip.open(path, "wt", encoding="utf-8") as gzipFile:
        gzipFile.write(text)
    return path


def LoadLoci(tmp_path):
    gffPath = WriteGzip(tmp_path / "genomic.gff.gz", "\n".join(GFF_LINES) + "\n")
    return SelectLongestCdsPerLocus(ReadGffCds(gffPath))


def LoadOntology(tmp_path):
    oboPath = tmp_path / "go-basic.obo"
    oboPath.write_text(OBO_TEXT, encoding="utf-8")
    return ParseObo(oboPath)


def TestReverseComplementHandlesAmbiguousBases():
    assert ReverseComplement("ATGCN") == "NGCAT"


def TestReadFastaJoinsLinesPerRecord(tmp_path):
    fastaPath = WriteGzip(tmp_path / "genome.fna.gz", ">chr description\nacgt\nTTAA\n>plasmid\nGG\n")
    assert ReadFasta(fastaPath) == {"chr": "ACGTTTAA", "plasmid": "GG"}


def TestMinusStrandCdsIsReverseComplemented(tmp_path):
    gene = BuildGene(LoadLoci(tmp_path)["b0002"], GENOME, 1000)
    assert gene["sequence"] == "ATGCTGCACGTAA"
    assert gene["strand"] == "-"


def TestPlusStrandCdsKeepsGenomeOrientation(tmp_path):
    gene = BuildGene(LoadLoci(tmp_path)["b0001"], GENOME, 1000)
    assert gene["sequence"] == "ATGAAACCCGGGTTTTAA"


def TestCdsIsTrimmedFromItsStartCodon(tmp_path):
    gene = BuildGene(LoadLoci(tmp_path)["b0002"], GENOME, 5)
    assert gene["sequence"] == "ATGCT"
    assert gene["cdsLength"] == 13


def TestMultiSegmentCdsIsJoinedInGeneOrder():
    cds = {"locusTag": "b0004", "geneName": "", "seqId": "chr", "strand": "-", "segments": [(23, 28), (30, 35)]}
    assert BuildGene(cds, GENOME, 1000)["sequence"] == ReverseComplement(GENOME["chr"][22:28] + GENOME["chr"][29:35])


def TestLongestCdsWinsAndPseudogenesAreSkipped(tmp_path):
    loci = LoadLoci(tmp_path)
    assert set(loci) == {"b0001", "b0002"}
    assert loci["b0001"]["segments"] == [(1, 18)]


def TestUniprotJoinUsesOrderedLocusNotGeneName(tmp_path):
    tsvPath = WriteGzip(tmp_path / "proteome.tsv.gz", (
        "Entry\tReviewed\tGene Names (primary)\tGene Names (ordered locus)\tGene Ontology IDs\n"
        "P1\treviewed\trevB\tb0001 JW0001\tGO:0000003\n"
        "P2\treviewed\tfwdA\tb0002 JW0002\tGO:0000004; GO:0000005\n"
        "P3\tunreviewed\tx\tb0002\tGO:0000001\n"
        "P4\treviewed\tnoLocus\tJW9999\tGO:0000001\n"
    ))
    uniprotByLocus = ReadUniprotByLocus(tsvPath)
    assert set(uniprotByLocus) == {"b0001", "b0002"}
    assert uniprotByLocus["b0001"]["goIds"] == {"GO:0000003"}
    assert uniprotByLocus["b0002"] == {"accessions": ["P2", "P3"], "goIds": {"GO:0000001", "GO:0000004", "GO:0000005"}}


def TestPropagationFollowsIsAAndPartOfOnly(tmp_path):
    ontology = LoadOntology(tmp_path)
    propagated = PropagateTerms({"GO:0000003"}, ontology, BuildAltIdMap(ontology), {})
    assert propagated == {"GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"}


def TestPropagationResolvesAltIdsAndDropsObsoleteTerms(tmp_path):
    ontology = LoadOntology(tmp_path)
    propagated = PropagateTerms({"GO:0000099", "GO:0000006", "GO:1234567"}, ontology, BuildAltIdMap(ontology), {})
    assert "GO:0000003" in propagated
    assert "GO:0000006" not in propagated


def TestTermSizeFilterKeepsInclusiveRangeAndDropsEmptyGenes():
    geneTerms = {"b1": {"GO:A", "GO:B"}, "b2": {"GO:A"}, "b3": {"GO:A", "GO:C"}, "b4": {"GO:C"}}
    keptTerms = SelectTermsBySize(CountTermGenes(geneTerms), 2, 2)
    assert keptTerms == {"GO:C"}
    assert FilterGeneTerms(geneTerms, keptTerms) == {"b3": {"GO:C"}, "b4": {"GO:C"}}


def TestWindowsCoverBothStrandsWithStep():
    windows = BuildWindows({"chr": "AACCGGTTA"}, 4, 2, ResolveStrands("both"))
    assert [(w["start"], w["strand"], w["sequence"]) for w in windows] == [
        (1, "+", "AACC"), (1, "-", "GGTT"),
        (3, "+", "CCGG"), (3, "-", "CCGG"),
        (5, "+", "GGTT"), (5, "-", "AACC"),
    ]


def TestResolveStrandsRejectsUnknownName():
    with pytest.raises(ValueError, match="both"):
        ResolveStrands("sideways")
