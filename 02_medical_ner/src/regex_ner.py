"""
Rule-based NER using regex patterns for structured medical codes and common entities.
This catches the low-hanging fruit: ICD-10 codes, CPT codes, NDC numbers, dosages,
and a curated list of drug names. It won't catch free-text diagnoses, but it's
reliable for what it does find — and it runs in milliseconds.
"""

import re
import json
import os
from typing import Optional

ICD10_PATTERN = re.compile(r'\b([A-Z]\d{2}\.\d{1,2})\b')
CPT_CODE_PATTERN = re.compile(r'\b(\d{5})\b')
NDC_PATTERN = re.compile(r'\b(\d{5}-\d{4}-\d{2})\b')
DOSAGE_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ml|mL|units|g|mcg/kg|mg/kg|mg/dL|mEq/L|mmol/L|ng/mL|pg/mL|IU/L|mIU/L|mcmol/L|U/L|mm/hr|g/dL)\b')
PERCENTAGE_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*%')
VITALS_PATTERN = re.compile(r'\b(BP|HR|RR|SpO2|O2\s*sat|temp|temperature)\s*[:\s]*(\d+(?:[/\.]\d+)?)\b', re.IGNORECASE)

COMMON_DRUGS = [
    "metformin", "lisinopril", "amlodipine", "metoprolol", "atorvastatin",
    "omeprazole", "losartan", "albuterol", "gabapentin", "hydrochlorothiazide",
    "levothyroxine", "acetaminophen", "ibuprofen", "amoxicillin", "azithromycin",
    "prednisone", "furosemide", "pantoprazole", "clopidogrel", "warfarin",
    "sertraline", "duloxetine", "fluoxetine", "escitalopram", "citalopram",
    "venlafaxine", "bupropion", "trazodone", "quetiapine", "olanzapine",
    "risperidone", "aripiprazole", "lamotrigine", "carbamazepine", "valproic acid",
    "phenytoin", "levetiracetam", "topiramate", "pregabalin", "tramadol",
    "oxycodone", "hydrocodone", "morphine", "fentanyl", "codeine",
    "ciprofloxacin", "levofloxacin", "doxycycline", "clindamycin", "vancomycin",
    "meropenem", "ceftriaxone", "cefazolin", "piperacillin", "ampicillin",
    "insulin", "glipizide", "glyburide", "sitagliptin", "empagliflozin",
    "pioglitazone", "semaglutide", "liraglutide", "dapagliflozin", "canagliflozin",
    "rosuvastatin", "simvastatin", "pravastatin", "ezetimibe", "fenofibrate",
    "aspirin", "rivaroxaban", "apixaban", "dabigatran", "enoxaparin",
    "heparin", "alteplase", "tenecteplase", "nitroglycerin", "diltiazem",
    "verapamil", "digoxin", "amiodarone", "sotalol", "flecainide",
    "spironolactone", "hydrochlorothiazide", "chlorthalidone", "bumetanide",
    "allopurinol", "colchicine", "indomethacin", "naproxen", "celecoxib",
    "methotrexate", "hydroxychloroquine", "sulfasalazine", "adalimumab",
    "infliximab", "etanercept", "rituximab", "trastuzumab", "pembrolizumab",
    "dexamethasone", "methylprednisolone", "hydrocortisone", "fludrocortisone",
    "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine",
    "ipratropium", "tiotropium", "fluticasone", "budesonide", "montelukast",
    "loratadine", "cetirizine", "diphenhydramine", "famotidine", "ranitidine",
    "ondansetron", "metoclopramide", "promethazine", "lactulose", "docusate",
    "kayexalate", "ergocalciferol", "alendronate", "ferrous sulfate",
    "magnesium sulfate", "sodium phosphate", "potassium chloride",
]

CPT_CONTEXT_WORDS = [
    "cpt", "procedure", "code", "billed", "billing", "service",
    "performed", "ordered", "scheduled", "completed",
]


def extract_icd10_codes(clinical_text: str) -> list[dict]:
    entities = []
    for match in ICD10_PATTERN.finditer(clinical_text):
        entities.append({
            "text": match.group(0),
            "type": "ICD10_CODE",
            "start": match.start(),
            "end": match.end(),
        })
    return entities


def extract_cpt_codes(clinical_text: str) -> list[dict]:
    """
    5-digit codes are ambiguous — only flag them as CPT if there's billing
    context nearby. Otherwise we'd false-positive on zip codes, phone numbers, etc.
    """
    entities = []
    text_lower = clinical_text.lower()
    for match in CPT_CODE_PATTERN.finditer(clinical_text):
        context_window = text_lower[max(0, match.start() - 50):match.end() + 50]
        if any(ctx_word in context_window for ctx_word in CPT_CONTEXT_WORDS):
            entities.append({
                "text": match.group(0),
                "type": "CPT_CODE",
                "start": match.start(),
                "end": match.end(),
            })
    return entities


def extract_ndc_codes(clinical_text: str) -> list[dict]:
    entities = []
    for match in NDC_PATTERN.finditer(clinical_text):
        entities.append({
            "text": match.group(0),
            "type": "NDC_CODE",
            "start": match.start(),
            "end": match.end(),
        })
    return entities


def extract_dosages(clinical_text: str) -> list[dict]:
    entities = []
    for match in DOSAGE_PATTERN.finditer(clinical_text):
        entities.append({
            "text": match.group(0),
            "type": "DOSAGE",
            "start": match.start(),
            "end": match.end(),
        })
    return entities


def extract_medications(clinical_text: str) -> list[dict]:
    entities = []
    text_lower = clinical_text.lower()
    for drug in COMMON_DRUGS:
        pattern = re.compile(r'\b' + re.escape(drug) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            entities.append({
                "text": clinical_text[match.start():match.end()],
                "type": "MEDICATION",
                "start": match.start(),
                "end": match.end(),
            })
    return entities


def extract_vitals(clinical_text: str) -> list[dict]:
    entities = []
    for match in VITALS_PATTERN.finditer(clinical_text):
        entities.append({
            "text": match.group(0),
            "type": "VITAL_SIGN",
            "start": match.start(),
            "end": match.end(),
        })
    return entities


def extract_all_entities(clinical_text: str) -> list[dict]:
    all_entities = []
    all_entities.extend(extract_icd10_codes(clinical_text))
    all_entities.extend(extract_cpt_codes(clinical_text))
    all_entities.extend(extract_ndc_codes(clinical_text))
    all_entities.extend(extract_dosages(clinical_text))
    all_entities.extend(extract_medications(clinical_text))
    all_entities.extend(extract_vitals(clinical_text))

    # deduplicate overlapping spans — keep the longer match
    all_entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    filtered = []
    last_end = -1
    for entity in all_entities:
        if entity["start"] >= last_end:
            filtered.append(entity)
            last_end = entity["end"]

    return filtered


def evaluate_regex_ner(data_path: str):
    with open(data_path, "r") as f:
        dataset = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for entry in dataset:
        clinical_text = entry["text"]
        gold_entities = {(e["start"], e["end"], e["type"]) for e in entry["entities"]}

        predicted = extract_all_entities(clinical_text)
        pred_entities = set()
        for p in predicted:
            for g_start, g_end, g_type in gold_entities:
                if abs(p["start"] - g_start) <= 3 and abs(p["end"] - g_end) <= 3:
                    pred_entities.add((g_start, g_end, g_type))

        tp = len(pred_entities & gold_entities)
        fp = len(predicted) - tp
        fn = len(gold_entities) - tp

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 60)
    print("Regex NER Evaluation")
    print("=" * 60)
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    sample_texts = [
        "Patient diagnosed with E11.65 (type 2 diabetes with hyperglycemia). "
        "Started on metformin 500mg twice daily. NDC: 00093-7214-01. "
        "CPT code 99213 billed for this visit.",

        "Labs: WBC 15.2, creatinine 1.8 mg/dL, potassium 5.1 mEq/L. "
        "BP 145/92, HR 98 bpm. Started vancomycin 1g IV q12h.",

        "ICD-10: J18.9 (pneumonia), K21.0 (GERD). "
        "Continue omeprazole 20mg daily, azithromycin 500mg x5 days.",
    ]

    for clinical_text in sample_texts:
        print(f"\nText: {clinical_text[:80]}...")
        entities = extract_all_entities(clinical_text)
        for entity in entities:
            print(f"  [{entity['type']:15s}] {entity['text']}")

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ner_annotations.json")
    if os.path.exists(data_path):
        print("\n")
        evaluate_regex_ner(data_path)
