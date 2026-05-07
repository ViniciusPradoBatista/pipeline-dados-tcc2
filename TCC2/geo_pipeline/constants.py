"""Plataformas GEO conhecidas e sinônimos de condição usados no pipeline."""

from typing import Dict

KNOWN_PLATFORMS: Dict[str, str] = {
    "GPL19117": "Affymetrix miRNA 4.0 (multispecies)",
    "GPL18941": "3D-Gene Human miRNA (Toray)",
    "GPL21263": "3D-Gene Human miRNA V21 (Toray)",
    "GPL18402": "Affymetrix miRNA 4.0",
    "GPL16384": "Affymetrix miRNA 3.0",
    "GPL8786": "Affymetrix miRNA 1.0",
    "GPL10850": "Agilent Human miRNA V2",
    "GPL18058": "Agilent Human miRNA V3",
    "GPL11487": "Agilent Human miRNA V4",
    "GPL7731": "Agilent Human miRNA V1",
    "GPL8179": "Illumina Human v2",
    "GPL6480": "Agilent Whole Human Genome",
    "GPL570": "Affymetrix HG-U133 Plus 2.0",
    "GPL96": "Affymetrix HG-U133A",
}

HEALTHY_SYNONYMS_STRICT = [
    "healthy control",
    "healthy controls",
    "healthy subject",
    "healthy subjects",
    "healthy volunteer",
    "normal control",
    "normal controls",
    "normal",
    "unaffected",
]

HEALTHY_SYNONYMS_BROAD = [
    "control",
    "controls",
    "non-cancer control",
    "non cancer control",
    "benign control",
]

PATHOLOGICAL_SYNONYMS = [
    "cancer",
    "carcinoma",
    "adenocarcinoma",
    "tumor",
    "tumour",
    "neoplasm",
    "malignant",
    "pdac",
    "pancreatic cancer",
    "cholangiocarcinoma",
    "disease",
    "diseases",
]
