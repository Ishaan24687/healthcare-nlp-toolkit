# Medical Named Entity Recognition

Extract medical entities (diagnoses, medications, procedures, lab values) from clinical text.

I built three approaches: a rule-based system with regex patterns, a custom-trained spaCy NER model, and a transformer-based token classifier. The regex baseline handles structured codes (ICD-10, CPT, NDC) well, but the learned models are better at catching free-text mentions like "the patient's diabetes" or "started on lisinopril."

In practice, the best results came from combining rules for structured codes with the learned model for free-text entities.

## Entity Types

| Entity | Description | Examples |
|--------|-------------|----------|
| DIAGNOSIS | Conditions and diseases | diabetes mellitus, acute MI, COPD |
| MEDICATION | Drug names and formulations | metformin, lisinopril 10mg, aspirin |
| PROCEDURE | Clinical procedures | colonoscopy, CT scan, echocardiogram |
| LAB_VALUE | Lab results with values | HbA1c 6.8, WBC 12.3, creatinine 1.4 |
| DOSAGE | Medication dosages | 500mg, 10 units, 2.5ml |

## Run

```bash
python src/data.py
python src/regex_ner.py
python src/spacy_ner.py
python src/transformer_ner.py
```
