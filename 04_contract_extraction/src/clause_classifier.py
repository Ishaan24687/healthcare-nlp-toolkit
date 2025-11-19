"""
Classify contract clauses into categories: pricing, SLA, termination,
compliance, confidentiality, liability, indemnification.
Uses TF-IDF + Random Forest — simple but effective for this structured text.
"""

import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from collections import Counter

CLAUSE_CATEGORIES = [
    "pricing",
    "sla",
    "termination",
    "compliance",
    "confidentiality",
    "liability",
    "indemnification",
    "general",
]

# manually labeled training data from contract patterns
TRAINING_CLAUSES = [
    ("Administrative fee of $3.25 per member per month for pharmacy benefit management services.", "pricing"),
    ("Dispensing fee of $1.75 per prescription for retail pharmacy claims.", "pricing"),
    ("Annual license fee of $450,000 payable in quarterly installments.", "pricing"),
    ("Implementation fee of $150,000 including configuration and training.", "pricing"),
    ("Generic prescriptions reimbursed at Maximum Allowable Cost plus dispensing fee.", "pricing"),
    ("Specialty medications priced at AWP minus fifteen percent plus dispensing fee of $2.50.", "pricing"),
    ("Rebate sharing pass through ninety percent of manufacturer rebates received.", "pricing"),
    ("Per-visit fee of $12.00 per completed telehealth visit.", "pricing"),
    ("Stop loss premium estimated at $45.00 PEPM for specific deductible coverage.", "pricing"),
    ("Annual fee adjustment shall not exceed three percent or CPI increase.", "pricing"),
    ("Monthly minimum fee of $5,000 regardless of visit volume.", "pricing"),
    ("Custom analytics projects billed at $250 per hour.", "pricing"),
    ("Transaction fees query based exchange $0.50 per transaction.", "pricing"),
    ("Inpatient services reimbursed at per diem rate of $2,800 for medical surgical admissions.", "pricing"),
    ("Brand name prescriptions reimbursed at AWP minus sixteen percent.", "pricing"),
    ("Process 99.5% of clean claims within twenty-four hours of receipt.", "sla"),
    ("System availability of 99.9% measured on a monthly basis.", "sla"),
    ("Answer eighty percent of member calls within thirty seconds.", "sla"),
    ("Complete prior authorization reviews within forty-eight hours.", "sla"),
    ("Service credit equal to two percent of monthly administrative fees for each missed SLA.", "sla"),
    ("Claims turnaround ninety-five percent within ten business days.", "sla"),
    ("Financial claims accuracy rate of ninety-nine percent.", "sla"),
    ("Platform availability 99.95% system uptime measured monthly.", "sla"),
    ("Clinical decision support queries return results within two seconds.", "sla"),
    ("Critical issues one hour response four hour resolution.", "sla"),
    ("Report delivery by the fifteenth business day of the following month.", "sla"),
    ("Data refresh within forty-eight hours of receipt.", "sla"),
    ("Page load times under three seconds for ninety-five percent of requests.", "sla"),
    ("Video quality minimum 720p for ninety-eight percent of sessions.", "sla"),
    ("Daily automated backups with recovery point objective of four hours.", "sla"),
    ("Terminate upon sixty days written notice for material breach uncured for thirty days.", "termination"),
    ("Client may terminate without cause upon one hundred eighty days prior written notice.", "termination"),
    ("Early termination fee equal to $500,000.", "termination"),
    ("Automatic renewal for successive one year periods unless ninety days notice.", "termination"),
    ("Run-out fee of $15.00 PEPM for six months following termination.", "termination"),
    ("Transition assistance for up to ninety days at reduced rate.", "termination"),
    ("Return or destroy all data within thirty days of termination.", "termination"),
    ("Cease all access to the platform and return all PHI within thirty days.", "termination"),
    ("HIPAA compliance as amended by the HITECH Act and all regulations promulgated thereunder.", "compliance"),
    ("Comply with all applicable state pharmacy practice acts.", "compliance"),
    ("Maintain SOC 2 Type II certification and comply with data protection regulations.", "compliance"),
    ("Comply with 21 CFR Part 11 requirements for electronic records and signatures.", "compliance"),
    ("Comply with Federal Anti-Kickback Statute and the Stark Law.", "compliance"),
    ("DEA registration and state pharmacy licenses at all locations.", "compliance"),
    ("Compliance with 42 CFR Part 2 Substance Abuse Confidentiality.", "compliance"),
    ("Maintain current credentialing and comply with NCQA standards.", "compliance"),
    ("Execute a Business Associate Agreement in compliance with HIPAA.", "compliance"),
    ("System shall comply with Good Clinical Practice ICH E6 requirements.", "compliance"),
    ("PHI encrypted using AES-256 encryption or equivalent.", "compliance"),
    ("Each party agrees to hold in confidence all Confidential Information.", "confidentiality"),
    ("Not disclose information to any third party without prior written consent.", "confidentiality"),
    ("Confidentiality obligations survive termination for five years.", "confidentiality"),
    ("Protect Confidential Information using reasonable care.", "confidentiality"),
    ("Patient data remains property of the health system.", "confidentiality"),
    ("All claims data and member information remain the property of Plan Sponsor.", "confidentiality"),
    ("Aggregate liability shall not exceed total fees paid during twelve month period.", "liability"),
    ("Neither party liable for indirect incidental consequential or punitive damages.", "liability"),
    ("Expressly disclaims liability for clinical decisions made using the Software.", "liability"),
    ("Total liability limited to $2,000,000 or total fees paid in prior 24 months.", "liability"),
    ("Vendor's total aggregate liability shall not exceed $3,500,000.", "liability"),
    ("In no event shall either party's aggregate liability exceed the total fees.", "liability"),
    ("Indemnify defend and hold harmless from third party claims.", "indemnification"),
    ("Indemnification up to a maximum of $5,000,000 per occurrence.", "indemnification"),
    ("Mutual indemnification each party against claims from negligent acts.", "indemnification"),
    ("Maintain professional liability insurance limits not less than $1,000,000 per occurrence.", "indemnification"),
    ("Indemnify for regulatory penalties caused by system non-compliance.", "indemnification"),
    ("Insurance requirements professional liability $1,000,000 per occurrence and $3,000,000 aggregate.", "indemnification"),
    ("Initial term three years commencing on the Effective Date.", "general"),
    ("This Agreement shall be governed by the laws of the State of Delaware.", "general"),
    ("Amendment only by written instrument signed by both parties.", "general"),
    ("Force majeure neither party liable for failure due to events beyond reasonable control.", "general"),
    ("Audit rights upon thirty days written notice no more than once per calendar year.", "general"),
]


def build_clause_classifier():
    clause_texts = [t for t, _ in TRAINING_CLAUSES]
    clause_labels = [l for _, l in TRAINING_CLAUSES]

    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        stop_words="english",
    )
    X = tfidf_vectorizer.fit_transform(clause_texts)

    clause_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )
    clause_classifier.fit(X, clause_labels)

    cv_scores = cross_val_score(clause_classifier, X, clause_labels, cv=3, scoring="f1_macro")
    print(f"3-fold CV F1 (macro): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return tfidf_vectorizer, clause_classifier


def classify_contract_clauses(contract_text: str, vectorizer, classifier) -> list[dict]:
    clauses = _split_into_clauses(contract_text)

    classified = []
    for clause in clauses:
        clause_text = clause["text"]
        if len(clause_text.split()) < 5:
            continue

        features = vectorizer.transform([clause_text])
        predicted_category = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        confidence = max(probabilities)

        classified.append({
            "section_id": clause.get("section_id", ""),
            "text": clause_text[:200],
            "category": predicted_category,
            "confidence": float(confidence),
        })

    return classified


def _split_into_clauses(contract_text: str) -> list[dict]:
    """Split contract text into individual clauses based on section numbering."""
    clause_pattern = re.compile(r'(\d+\.\d+)\s+(.+?)(?=\n\d+\.\d+\s|\nSECTION\s|\Z)', re.DOTALL)
    clauses = []

    for match in clause_pattern.finditer(contract_text):
        clause_text = match.group(2).strip()
        clause_text = re.sub(r'\s+', ' ', clause_text)
        clauses.append({
            "section_id": match.group(1),
            "text": clause_text,
        })

    if not clauses:
        paragraphs = contract_text.split('\n\n')
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para.split()) >= 5:
                clauses.append({"section_id": str(i), "text": para})

    return clauses


def classify_all_contracts(data_path: str, output_dir: str = "outputs"):
    print("=" * 60)
    print("Contract Clause Classification")
    print("=" * 60)

    with open(data_path, "r") as f:
        contracts = json.load(f)

    print(f"\nBuilding clause classifier from {len(TRAINING_CLAUSES)} labeled examples...")
    print(f"Categories: {sorted(set(l for _, l in TRAINING_CLAUSES))}")
    print(f"Distribution: {dict(Counter(l for _, l in TRAINING_CLAUSES))}")

    vectorizer, classifier = build_clause_classifier()

    all_results = []
    for contract in contracts:
        print(f"\n{contract['id']}: {contract['title']}")
        classified = classify_contract_clauses(contract["text"], vectorizer, classifier)

        category_counts = Counter(c["category"] for c in classified)
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat:20s}: {count} clauses")

        all_results.append({
            "contract_id": contract["id"],
            "title": contract["title"],
            "clauses": classified,
            "category_distribution": dict(category_counts),
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "classified_clauses.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nClassification results saved to {output_path}")

    total_clauses = sum(len(r["clauses"]) for r in all_results)
    overall_dist = Counter()
    for r in all_results:
        overall_dist.update(r["category_distribution"])

    print(f"\nOverall: {total_clauses} clauses classified across {len(contracts)} contracts")
    print("Category distribution:")
    for cat, count in sorted(overall_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:3d} ({count/total_clauses*100:.1f}%)")

    return all_results


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "sample_contracts.json")
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(data_path):
        print("Contract data not found. Generating samples first...")
        from sample_contracts import save_contracts
        save_contracts(os.path.join(base_dir, "data"))

    classify_all_contracts(data_path, output_dir)
