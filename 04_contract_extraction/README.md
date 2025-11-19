# Contract Entity Extraction

Extract key terms, SLAs, pricing structures, and compliance requirements from healthcare contracts. At Lantern I built something similar using Azure Document Intelligence — this version uses pure Python to demonstrate the NLP logic without cloud dependencies.

The pipeline goes: raw text → section parsing → clause classification → structured entity extraction. Each stage is independent so you can swap in better components (e.g., replace regex date extraction with a learned model) without touching the rest.

## Pipeline

1. **Document Parser** — Split raw contract text into sections using regex + heuristics
2. **Clause Classifier** — TF-IDF + Random Forest to classify clauses into categories (pricing, SLA, termination, compliance, etc.)
3. **Term Extractor** — Extract structured entities: dates, monetary amounts, percentages, party names, SLA metrics

## Entity Types

| Entity | Method | Examples |
|--------|--------|----------|
| Dates | Regex + dateutil | "January 1, 2024", "12/31/2025" |
| Monetary Amounts | Regex | "$1,500,000", "$25.00 per member" |
| Percentages | Regex | "99.5% uptime", "15% discount" |
| Party Names | Heuristic extraction | "Lantern Care Holdings", "ABC Health Plan" |
| SLA Metrics | Pattern matching | "99.9% system availability", "48-hour turnaround" |
| Compliance References | Keyword matching | "HIPAA", "HITECH", "42 CFR Part 2" |

## Run

```bash
python src/sample_contracts.py
python src/document_parser.py
python src/clause_classifier.py
python src/term_extractor.py
```
