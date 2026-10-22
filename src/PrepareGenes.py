import csv
import json
import re
from pathlib import Path
from urllib.parse import unquote

from src.FetchData import LoadConfig, OpenText
from src.GenomeSequence import BuildWindows, ReadFasta, ResolveStrands, ReverseComplement
from src.GoOntology import BuildAltIdMap, CountTermGenes, ParseObo, PropagateTerms, SelectTermsBySize

LOCUS_PATTERN = re.compile(r"^b\d{4}$")
UNIPROT_LOCUS_COLUMN = "Gene Names (ordered locus)"
UNIPROT_GO_COLUMN = "Gene Ontology IDs"
GENE_COLUMNS = ["locusTag", "geneName", "accessions", "strand", "start", "end", "cdsLength", "goTerms", "sequence"]
TERM_COLUMNS = ["goId", "name", "namespace", "geneCount"]
WINDOW_COLUMNS = ["seqId", "start", "end", "strand", "sequence"]
# Both strands until Config.yaml carries the key, decided in #14.
DEFAULT_WINDOW_STRANDS = "both"


def ParseGffAttributes(field):
    pairs = (item.split("=", 1) for item in field.strip().split(";") if "=" in item)
    return {key: unquote(value) for key, value in pairs}


def IsUsableCds(columns, attributes):
    return columns[2] == "CDS" and "locus_tag" in attributes and attributes.get("pseudo") != "true"


def ReadGffCds(gffPath):
    cdsById = {}
    with OpenText(gffPath) as gffFile:
        for line in gffFile:
            columns = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(columns) < 9:
                continue
            attributes = ParseGffAttributes(columns[8])
            if not IsUsableCds(columns, attributes):
                continue
            cds = cdsById.setdefault(attributes["ID"], {
                "locusTag": attributes["locus_tag"], "geneName": attributes.get("gene", ""),
                "seqId": columns[0], "strand": columns[6], "segments": [],
            })
            cds["segments"].append((int(columns[3]), int(columns[4])))
    return cdsById


def CdsLength(cds):
    return sum(end - start + 1 for start, end in cds["segments"])


def SelectLongestCdsPerLocus(cdsById):
    longest = {}
    for cds in cdsById.values():
        current = longest.get(cds["locusTag"])
        if current is None or CdsLength(cds) > CdsLength(current):
            longest[cds["locusTag"]] = cds
    return longest


def ExtractCdsSequence(cds, genome):
    chromosome = genome[cds["seqId"]]
    sequence = "".join(chromosome[start - 1:end] for start, end in sorted(cds["segments"]))
    return ReverseComplement(sequence) if cds["strand"] == "-" else sequence


def BuildGene(cds, genome, maxNucleotides):
    sequence = ExtractCdsSequence(cds, genome)
    return {
        "locusTag": cds["locusTag"], "geneName": cds["geneName"], "strand": cds["strand"],
        "start": min(start for start, _ in cds["segments"]), "end": max(end for _, end in cds["segments"]),
        "cdsLength": len(sequence), "sequence": sequence[:maxNucleotides],
    }


def ParseLocusNumbers(field):
    return [token for token in field.split() if LOCUS_PATTERN.match(token)]


def ReadUniprotByLocus(tsvPath):
    uniprotByLocus = {}
    with OpenText(tsvPath) as tsvFile:
        for row in csv.DictReader(tsvFile, delimiter="\t"):
            goIds = {goId.strip() for goId in row[UNIPROT_GO_COLUMN].split(";") if goId.strip()}
            for locusTag in ParseLocusNumbers(row[UNIPROT_LOCUS_COLUMN]):
                entry = uniprotByLocus.setdefault(locusTag, {"accessions": [], "goIds": set()})
                entry["accessions"].append(row["Entry"])
                entry["goIds"] |= goIds
    return uniprotByLocus


def PropagateGeneTerms(genes, uniprotByLocus, ontology):
    altIdMap = BuildAltIdMap(ontology)
    cache = {}
    geneTerms = {}
    for locusTag in genes:
        goIds = uniprotByLocus.get(locusTag, {}).get("goIds", set())
        terms = PropagateTerms(goIds, ontology, altIdMap, cache)
        if terms:
            geneTerms[locusTag] = terms
    return geneTerms


def FilterGeneTerms(geneTerms, keptTerms):
    filtered = {locusTag: terms & keptTerms for locusTag, terms in geneTerms.items()}
    return {locusTag: terms for locusTag, terms in filtered.items() if terms}


def WriteTsv(rows, columns, targetPath):
    with open(targetPath, "w", encoding="utf-8", newline="") as targetFile:
        writer = csv.DictWriter(targetFile, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def BuildGeneRows(genes, geneTerms, uniprotByLocus):
    rows = []
    for locusTag in sorted(geneTerms):
        row = dict(genes[locusTag])
        row["accessions"] = ";".join(uniprotByLocus[locusTag]["accessions"])
        row["goTerms"] = ";".join(sorted(geneTerms[locusTag]))
        rows.append(row)
    return rows


def BuildTermRows(termCounts, keptTerms, ontology):
    return [
        {"goId": termId, "name": ontology[termId]["name"], "namespace": ontology[termId]["namespace"], "geneCount": termCounts[termId]}
        for termId in sorted(keptTerms)
    ]


def LoadGenes(config, rawDir):
    sources = config["sources"]
    genome = ReadFasta(rawDir / sources["genomeFasta"]["file"])
    loci = SelectLongestCdsPerLocus(ReadGffCds(rawDir / sources["genomeGff"]["file"]))
    genes = {locusTag: BuildGene(cds, genome, config["model"]["maxNucleotides"]) for locusTag, cds in loci.items()}
    return genome, genes


def BuildSummary(genes, uniprotByLocus, geneTerms, filteredTerms, termCounts, keptTerms):
    matched = sum(1 for locusTag in genes if locusTag in uniprotByLocus)
    return {
        "gffLoci": len(genes),
        "matchedToUniprot": matched,
        "matchedPercent": round(100 * matched / len(genes), 2),
        "genesWithGoBeforeFilter": len(geneTerms),
        "genesWithGo": len(filteredTerms),
        "goTermsAfterPropagation": len(termCounts),
        "goTermsAfterFilter": len(keptTerms),
    }


def PrepareGeneTable(config, rawDir):
    genome, genes = LoadGenes(config, rawDir)
    uniprotByLocus = ReadUniprotByLocus(rawDir / config["sources"]["uniprotProteome"]["file"])
    ontology = ParseObo(rawDir / config["sources"]["goBasic"]["file"])
    geneTerms = PropagateGeneTerms(genes, uniprotByLocus, ontology)
    termCounts = CountTermGenes(geneTerms)
    enrichment = config["enrichment"]
    keptTerms = SelectTermsBySize(termCounts, enrichment["minTermGenes"], enrichment["maxTermGenes"])
    filteredTerms = FilterGeneTerms(geneTerms, keptTerms)
    summary = BuildSummary(genes, uniprotByLocus, geneTerms, filteredTerms, termCounts, keptTerms)
    geneRows = BuildGeneRows(genes, filteredTerms, uniprotByLocus)
    termRows = BuildTermRows(CountTermGenes(filteredTerms), keptTerms, ontology)
    return genome, geneRows, termRows, summary


def Main():
    config = LoadConfig()
    rawDir = Path(config["paths"]["raw"])
    processedDir = Path(config["paths"]["processed"])
    processedDir.mkdir(parents=True, exist_ok=True)
    genome, geneRows, termRows, summary = PrepareGeneTable(config, rawDir)
    strands = ResolveStrands(config["windows"].get("strands", DEFAULT_WINDOW_STRANDS))
    windows = BuildWindows(genome, config["windows"]["length"], config["windows"]["step"], strands)
    summary["windows"] = len(windows)
    WriteTsv(geneRows, GENE_COLUMNS, processedDir / "Genes.tsv")
    WriteTsv(termRows, TERM_COLUMNS, processedDir / "GoTerms.tsv")
    WriteTsv(windows, WINDOW_COLUMNS, processedDir / "Windows.tsv")
    with open(processedDir / "PrepareGenesSummary.json", "w", encoding="utf-8") as summaryFile:
        json.dump(summary, summaryFile, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    Main()
