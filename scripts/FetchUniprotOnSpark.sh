#!/usr/bin/env bash
# Downloads the UniProt proteome on the DGX Spark into data/raw and updates the manifest.
set -euo pipefail

repoRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repoRoot"

python="${PYTHON:-python3.12}"
if ! command -v "$python" >/dev/null 2>&1; then
    echo "error: $python not found; set PYTHON to a Python 3.11 or 3.12 interpreter" >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    "$python" -m venv .venv || {
        echo "error: venv creation failed; on Ubuntu install python3.12-venv" >&2
        exit 1
    }
fi

# Only the fetch dependencies, with the pins from requirements.txt; the model stack is not needed here.
fetchRequirements=$(grep -E '^(requests|PyYAML)==' requirements.txt)
.venv/bin/python -m pip install --quiet $fetchRequirements

.venv/bin/python -m src.FetchData --source uniprotProteome

.venv/bin/python - <<'EOF'
import json

entry = json.load(open("data/raw/manifest.json", encoding="utf-8"))["uniprotProteome"]
print(f"UniProt release {entry['version']}, sha256 {entry['sha256']}")
EOF
