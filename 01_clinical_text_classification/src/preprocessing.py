"""
Medical text preprocessing pipeline.
Handles abbreviation expansion, normalization, and tokenization specific to
clinical documentation where abbreviations like "pt" and "hx" are everywhere.
"""

import re
from typing import Optional

MEDICAL_ABBREVIATIONS = {
    "pt": "patient",
    "pts": "patients",
    "hx": "history",
    "dx": "diagnosis",
    "rx": "prescription",
    "tx": "treatment",
    "sx": "symptoms",
    "fx": "fracture",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "n/v": "nausea and vomiting",
    "bpm": "beats per minute",
    "yo": "year old",
    "y/o": "year old",
    "w/": "with",
    "w/o": "without",
    "s/p": "status post",
    "b/l": "bilateral",
    "c/o": "complaining of",
    "d/c": "discharge",
    "f/u": "follow up",
    "h/o": "history of",
    "r/o": "rule out",
    "abd": "abdominal",
    "ams": "altered mental status",
    "bka": "below knee amputation",
    "bmp": "basic metabolic panel",
    "bun": "blood urea nitrogen",
    "cabg": "coronary artery bypass graft",
    "cbc": "complete blood count",
    "cmp": "comprehensive metabolic panel",
    "copd": "chronic obstructive pulmonary disease",
    "cva": "cerebrovascular accident",
    "dnr": "do not resuscitate",
    "dvt": "deep vein thrombosis",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "er": "emergency room",
    "etoh": "alcohol",
    "gcs": "glasgow coma scale",
    "gi": "gastrointestinal",
    "gtt": "drip",
    "hpi": "history of present illness",
    "icu": "intensive care unit",
    "im": "intramuscular",
    "iv": "intravenous",
    "mi": "myocardial infarction",
    "npo": "nothing by mouth",
    "nsr": "normal sinus rhythm",
    "or": "operating room",
    "pe": "pulmonary embolism",
    "prn": "as needed",
    "rom": "range of motion",
    "tia": "transient ischemic attack",
    "uri": "upper respiratory infection",
    "uti": "urinary tract infection",
    "wbc": "white blood cell",
    "wnl": "within normal limits",
}


def expand_abbreviations(clinical_text: str) -> str:
    expanded = clinical_text
    for abbrev, full_form in sorted(MEDICAL_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(abbrev) + r'\b', re.IGNORECASE)
        expanded = pattern.sub(full_form, expanded)
    return expanded


def clean_clinical_text(clinical_text: str) -> str:
    text = clinical_text.lower()
    text = expand_abbreviations(text)

    # keep decimal points in numbers like "6.8" and "O2"
    text = re.sub(r'[^a-z0-9\s\.]', ' ', text)
    text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(clinical_text: str) -> list[str]:
    cleaned = clean_clinical_text(clinical_text)
    tokens = cleaned.split()
    return tokens


def remove_stopwords(tokens: list[str], extra_stopwords: Optional[set] = None) -> list[str]:
    # lightweight stopword list — we keep medical terms that happen to be common
    base_stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "each", "every", "all", "any", "few", "more", "most", "other",
        "some", "such", "than", "too", "very", "just", "also", "then",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "them", "their", "we", "our", "you", "your",
    }
    if extra_stopwords:
        base_stopwords |= extra_stopwords

    return [t for t in tokens if t not in base_stopwords]


def preprocess_pipeline(clinical_text: str, remove_stops: bool = True) -> str:
    tokens = tokenize(clinical_text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    sample_notes = [
        "72 yo male. HPI: Pt c/o severe CP radiating to L arm. Hx of MI x2. BP 90/60.",
        "45 y/o female. Routine f/u for HTN. Vitals WNL. Continue current Rx.",
        "Pt presents w/ SOB and AMS. GCS 8. Intubation required. Code blue called.",
    ]

    for clinical_note in sample_notes:
        print(f"Original:    {clinical_note}")
        print(f"Cleaned:     {clean_clinical_text(clinical_note)}")
        print(f"Preprocessed:{preprocess_pipeline(clinical_note)}")
        print()
