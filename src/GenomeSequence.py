from src.FetchData import OpenText

COMPLEMENT = str.maketrans("ACGTRYKMSWBDHVNacgtrykmswbdhvn", "TGCAYRMKSWVHDBNtgcayrmkswvhdbn")
STRANDS_BY_NAME = {"both": ("+", "-"), "forward": ("+",), "reverse": ("-",)}


def ReverseComplement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def ReadFasta(fastaPath):
    records = {}
    recordId = None
    with OpenText(fastaPath) as fastaFile:
        for line in fastaFile:
            line = line.strip()
            if line.startswith(">"):
                recordId = line[1:].split()[0]
                records[recordId] = []
            elif recordId:
                records[recordId].append(line.upper())
    return {recordId: "".join(chunks) for recordId, chunks in records.items()}


def ResolveStrands(strandsName):
    if strandsName not in STRANDS_BY_NAME:
        raise ValueError(f"Unknown window strands '{strandsName}'. Known: {', '.join(STRANDS_BY_NAME)}")
    return STRANDS_BY_NAME[strandsName]


def BuildWindows(genome, length, step, strands):
    windows = []
    for seqId, sequence in genome.items():
        for start in range(0, len(sequence) - length + 1, step):
            forward = sequence[start:start + length]
            for strand in strands:
                windowSequence = forward if strand == "+" else ReverseComplement(forward)
                windows.append({"seqId": seqId, "start": start + 1, "end": start + length, "strand": strand, "sequence": windowSequence})
    return windows
