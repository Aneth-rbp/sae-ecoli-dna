from collections import Counter

from src.FetchData import OpenText

PROPAGATION_RELATIONS = ("part_of",)


def NewTerm(termId):
    return {"id": termId, "name": "", "namespace": "", "parents": set(), "altIds": [], "isObsolete": False}


def ApplyOboTag(term, tag, value):
    if tag in ("name", "namespace"):
        term[tag] = value
    elif tag == "alt_id":
        term["altIds"].append(value)
    elif tag == "is_a":
        term["parents"].add(value.split()[0])
    elif tag == "relationship" and value.split()[0] in PROPAGATION_RELATIONS:
        term["parents"].add(value.split()[1])
    elif tag == "is_obsolete":
        term["isObsolete"] = value == "true"


def ReadOboStanzas(oboPath):
    terms = []
    current = None
    with OpenText(oboPath) as oboFile:
        for line in oboFile:
            line = line.strip()
            if line.startswith("["):
                current = {} if line == "[Term]" else None
                if current is not None:
                    terms.append(current)
            elif current is not None and ": " in line:
                tag, value = line.split(": ", 1)
                current.setdefault(tag, []).append(value.split(" ! ")[0])
    return terms


def ParseObo(oboPath):
    ontology = {}
    for stanza in ReadOboStanzas(oboPath):
        term = NewTerm(stanza["id"][0])
        for tag, values in stanza.items():
            for value in values:
                ApplyOboTag(term, tag, value)
        if not term["isObsolete"]:
            ontology[term["id"]] = term
    return ontology


def BuildAltIdMap(ontology):
    altIdMap = {termId: termId for termId in ontology}
    for termId, term in ontology.items():
        altIdMap.update({altId: termId for altId in term["altIds"]})
    return altIdMap


def CollectAncestors(termId, ontology, cache):
    if termId not in cache:
        ancestors = {termId}
        for parentId in ontology[termId]["parents"]:
            if parentId in ontology:
                ancestors |= CollectAncestors(parentId, ontology, cache)
        cache[termId] = ancestors
    return cache[termId]


def PropagateTerms(termIds, ontology, altIdMap, cache):
    propagated = set()
    for termId in termIds:
        primaryId = altIdMap.get(termId)
        if primaryId:
            propagated |= CollectAncestors(primaryId, ontology, cache)
    return propagated


def CountTermGenes(geneTerms):
    return Counter(termId for terms in geneTerms.values() for termId in terms)


def SelectTermsBySize(termCounts, minGenes, maxGenes):
    return {termId for termId, count in termCounts.items() if minGenes <= count <= maxGenes}
