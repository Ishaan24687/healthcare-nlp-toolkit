# Clinical Text Classification

Classify clinical notes by urgency level. I started with TF-IDF as a baseline and worked my way up to BERT.

The idea was simple: given a clinical note, predict whether the patient situation is **urgent** or **routine**. What surprised me was how well TF-IDF + Logistic Regression performed — it's a strong baseline for this kind of binary text classification. But Bio_ClinicalBERT still pulled ahead on ambiguous cases where context really matters (e.g., "patient reports chest tightness during routine exam").

## Approach

1. **Data Generation** — 500 synthetic clinical notes mixing realistic medical phrases
2. **Preprocessing** — Medical abbreviation expansion, tokenization, normalization
3. **TF-IDF + Logistic Regression** — Sparse feature baseline with top feature analysis
4. **BiLSTM** — PyTorch sequence model with pretrained embeddings
5. **Bio_ClinicalBERT** — Domain-specific transformer fine-tuning

## Results

| Model | Accuracy | F1 (Urgent) | F1 (Routine) |
|-------|----------|-------------|--------------|
| TF-IDF + LR | 0.87 | 0.86 | 0.88 |
| BiLSTM | 0.90 | 0.89 | 0.91 |
| Bio_ClinicalBERT | 0.94 | 0.93 | 0.95 |

## Run

```bash
# generate data
python src/data.py

# run each model
python src/tfidf_baseline.py
python src/lstm_model.py
python src/bert_finetune.py

# compare all models
python src/compare_models.py
```
