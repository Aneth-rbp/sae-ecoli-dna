import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.FetchData import LoadConfig

MATRICES = [
    {"name": "Sae", "label": "SAE", "color": "#2a78d6", "histtype": "stepfilled", "alpha": 0.45},
    {"name": "Baseline", "label": "Evo 2 dimensions", "color": "#eb6834", "histtype": "step", "alpha": 1.0},
]
MODE_TITLES = {"topN": "Top-N genes", "positive": "Activation > 0"}
TABLE_HEADER = ["Feature", "Top GO terms (adj. p)", "Top genes"]
TABLE_COLUMN_WIDTHS = [0.09, 0.61, 0.30]
TABLE_WRAP_CHARACTERS = [8, 72, 36]
TABLE_CELL_PAD = 0.015
DEFAULT_FIGURES = {
    "dpi": 300, "histogramWidthInches": 3.35, "histogramHeightInches": 2.3, "tableWidthInches": 6.9,
    "fontSize": 7, "histogramBins": 30, "tableFeatures": 5, "showPositiveMode": False,
}


def ReadEnrichment(enrichmentPath):
    return pd.read_csv(enrichmentPath, sep="\t", dtype={"bestTerm": str, "bestTermName": str})


def BestNegLog10(table, mode):
    adjustedP = table.loc[table["mode"] == mode, "adjustedP"].astype(float).fillna(1.0)
    return list(-np.log10(adjustedP.clip(lower=np.finfo(float).tiny)) + 0.0)


def PercentSignificant(table, mode):
    flags = table.loc[table["mode"] == mode, "isSignificant"].astype(str).str.lower() == "true"
    return 100 * flags.mean() if len(flags) else 0.0


def BuildHistogramBins(scoreLists, binCount):
    upper = max((float(np.max(scores)) for scores in scoreLists if len(scores)), default=0.0)
    return np.linspace(0.0, max(upper, 1.0) * 1.02, binCount + 1)


def DrawModePanel(axis, tables, mode, settings, fdr):
    scoreLists = [np.asarray(BestNegLog10(tables[matrix["name"]], mode)) for matrix in MATRICES]
    bins = BuildHistogramBins(scoreLists, settings["histogramBins"])
    for matrix, scores in zip(MATRICES, scoreLists):
        percent = PercentSignificant(tables[matrix["name"]], mode)
        weights = np.full(len(scores), 100 / max(len(scores), 1))
        axis.hist(scores, bins=bins, weights=weights, color=matrix["color"], histtype=matrix["histtype"],
                  alpha=matrix["alpha"], linewidth=1.2,
                  label=f"{matrix['label']} (n={len(scores)}, {percent:.0f}% sig.)")
    axis.axvline(-np.log10(fdr), color="#444444", linestyle="--", linewidth=0.8, label=f"adj. p = {fdr:g}")
    axis.set_xlim(bins[0], bins[-1])
    axis.set_title(MODE_TITLES.get(mode, mode))
    axis.set_xlabel(r"$-\log_{10}$(adj. p), best GO term")
    axis.legend(frameon=False, loc="upper right")


def DrawHistogram(tables, settings, fdr):
    modes = ["topN", "positive"] if settings["showPositiveMode"] else ["topN"]
    height = settings["histogramHeightInches"] * len(modes)
    figure, axes = plt.subplots(len(modes), 1, figsize=(settings["histogramWidthInches"], height), squeeze=False)
    for axis, mode in zip(axes[:, 0], modes):
        DrawModePanel(axis, tables, mode, settings, fdr)
        axis.set_ylabel("Units (%)")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def ReadCandidateFeatures(candidatesPath, count):
    candidates = pd.read_csv(candidatesPath, sep="\t", dtype={"topTerms": str, "topGenes": str}).fillna("")
    return candidates.sort_values("rank").head(count).reset_index(drop=True)


def SplitItems(field):
    return [item.strip() for item in str(field).split(";") if item.strip()]


def BuildTableCells(candidates):
    rows = []
    for candidate in candidates.itertuples():
        cells = [str(candidate.unit), "\n".join(SplitItems(candidate.topTerms)), ", ".join(SplitItems(candidate.topGenes))]
        rows.append([WrapCell(cell, width) for cell, width in zip(cells, TABLE_WRAP_CHARACTERS)])
    return rows


def WrapCell(text, width):
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def DrawFeatureTable(candidates, settings):
    rows = BuildTableCells(candidates)
    lineCounts = [max(cell.count("\n") + 1 for cell in row) for row in rows]
    lineHeight = settings["fontSize"] * 1.45 / 72
    height = lineHeight * (sum(lineCounts) + 1) + 0.6 * lineHeight * (len(rows) + 1)
    figure = plt.figure(figsize=(settings["tableWidthInches"], height))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=TABLE_HEADER, colWidths=TABLE_COLUMN_WIDTHS, loc="upper left",
                       cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(settings["fontSize"])
    StyleTable(table, [1] + lineCounts)
    return figure


def StyleTable(table, lineCounts):
    total = sum(count + 0.6 for count in lineCounts)
    for (row, _), cell in table.get_celld().items():
        cell.set_height((lineCounts[row] + 0.6) / total)
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.5)
        cell.PAD = TABLE_CELL_PAD
        cell.get_text().set_verticalalignment("center")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")


def EscapeMarkdown(text):
    return str(text).replace("|", "\\|")


def BuildFeatureMarkdown(candidates):
    lines = ["| " + " | ".join(TABLE_HEADER) + " |", "|---|---|---|"]
    for candidate in candidates.itertuples():
        terms = "<br>".join(EscapeMarkdown(term) for term in SplitItems(candidate.topTerms))
        genes = ", ".join(EscapeMarkdown(gene) for gene in SplitItems(candidate.topGenes))
        lines.append(f"| {candidate.unit} | {terms} | {genes} |")
    return "\n".join(lines) + "\n"


def SaveFigure(figure, basePath, dpi):
    for suffix in [".png", ".pdf"]:
        figure.savefig(basePath.with_suffix(suffix), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def WriteFigures(config):
    settings = {**DEFAULT_FIGURES, **config.get("figures", {})}
    resultsDir = Path(config["paths"]["results"])
    figuresDir = Path(config["paths"]["figures"])
    figuresDir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": settings["fontSize"], "pdf.fonttype": 42})
    tables = {matrix["name"]: ReadEnrichment(resultsDir / "enrichment" / f"{matrix['name']}Enrichment.tsv") for matrix in MATRICES}
    SaveFigure(DrawHistogram(tables, settings, config["enrichment"]["fdr"]), figuresDir / "PValueHistogram", settings["dpi"])
    candidates = ReadCandidateFeatures(resultsDir / "CandidateFeatures.tsv", settings["tableFeatures"])
    SaveFigure(DrawFeatureTable(candidates, settings), figuresDir / "FeatureTable", settings["dpi"])
    (figuresDir / "FeatureTable.md").write_text(BuildFeatureMarkdown(candidates), encoding="utf-8")


def ParseArguments(argv=None):
    parser = argparse.ArgumentParser(description="Draw the report figures from the enrichment and comparison results.")
    parser.add_argument("--positive", action="store_true", help="Add the activation > 0 mode as a second histogram panel.")
    return parser.parse_args(argv)


def Main(argv=None):
    arguments = ParseArguments(argv)
    config = LoadConfig()
    if arguments.positive:
        config.setdefault("figures", {})["showPositiveMode"] = True
    WriteFigures(config)


if __name__ == "__main__":
    Main()
