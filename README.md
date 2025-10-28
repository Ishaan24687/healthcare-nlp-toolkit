# Healthcare NLP Toolkit

I put together these four NLP projects while working on healthcare text problems. Each one taught me something different about processing clinical and administrative text.

The projects go from straightforward classification up through entity extraction on messy contract documents. I built them incrementally — lessons from earlier projects informed decisions in later ones (for example, discovering that domain-specific embeddings matter in Project 1 shaped how I approached NER in Project 2).

## Projects

| # | Project | Approach | Key Results |
|---|---------|----------|-------------|
| 1 | [Clinical Text Classification](01_clinical_text_classification/) | TF-IDF baseline → BiLSTM → Bio_ClinicalBERT fine-tuning | Bio_ClinicalBERT hit 0.94 F1 on urgency classification; TF-IDF baseline was surprisingly strong at 0.87 |
| 2 | [Medical Named Entity Recognition](02_medical_ner/) | Regex rules → Custom spaCy NER → Transformer token classification | Hybrid approach (rules + learned model) gave best results; regex alone caught 78% of entities |
| 3 | [CPT Code Analysis](03_cpt_analysis/) | TF-IDF clustering + statistical reimbursement analysis + anomaly detection | Found ASC codes in the South are ~15% below national average; identified pricing outliers via Isolation Forest |
| 4 | [Contract Entity Extraction](04_contract_extraction/) | Section parsing → clause classification → structured term extraction | Extracted dates, monetary amounts, SLA metrics, and compliance references from raw contract text |

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

## Tech Stack

- **Frameworks**: PyTorch, HuggingFace Transformers, spaCy, scikit-learn
- **Models**: Bio_ClinicalBERT, BiLSTM, TF-IDF + classical ML
- **Analysis**: scipy stats, Isolation Forest, KMeans clustering

## Structure

```
healthcare-nlp-toolkit/
├── 01_clinical_text_classification/
│   └── src/
│       ├── data.py                 # synthetic clinical note generation
│       ├── preprocessing.py        # medical text cleaning pipeline
│       ├── tfidf_baseline.py       # TF-IDF + Logistic Regression
│       ├── lstm_model.py           # PyTorch BiLSTM classifier
│       ├── bert_finetune.py        # Bio_ClinicalBERT fine-tuning
│       └── compare_models.py       # cross-model evaluation
├── 02_medical_ner/
│   └── src/
│       ├── data.py                 # IOB-annotated clinical sentences
│       ├── regex_ner.py            # rule-based entity extraction
│       ├── spacy_ner.py            # custom spaCy NER training
│       └── transformer_ner.py      # BERT token classification
├── 03_cpt_analysis/
│   └── src/
│       ├── cpt_data.py             # synthetic CPT code dataset
│       ├── tfidf_clustering.py     # procedure description clustering
│       ├── reimbursement_analysis.py   # statistical pricing analysis
│       └── pricing_anomalies.py    # Isolation Forest anomaly detection
├── 04_contract_extraction/
│   └── src/
│       ├── sample_contracts.py     # synthetic healthcare contracts
│       ├── document_parser.py      # section/structure extraction
│       ├── clause_classifier.py    # clause type classification
│       └── term_extractor.py       # structured entity extraction
├── requirements.txt
└── README.md
```
