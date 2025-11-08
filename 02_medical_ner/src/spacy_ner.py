"""
Train a custom spaCy NER model on annotated clinical data.
Combines an EntityRuler for known patterns with a statistical NER component
that learns from the annotated examples. The ruler handles structured codes
while the trained model picks up free-text entity mentions.
"""
# TODO: experiment with spaCy's transformer-based pipeline (en_core_web_trf)
# for potentially better out-of-the-box performance on clinical text

import json
import os
import random

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin


ENTITY_PATTERNS = [
    {"label": "DOSAGE", "pattern": [{"SHAPE": "ddd"}, {"LOWER": {"IN": ["mg", "mcg", "ml", "units", "g"]}}]},
    {"label": "DOSAGE", "pattern": [{"SHAPE": "dddd"}, {"LOWER": {"IN": ["mg", "mcg", "ml", "units", "g"]}}]},
    {"label": "DOSAGE", "pattern": [{"SHAPE": "dd"}, {"LOWER": {"IN": ["mg", "mcg", "ml", "units", "g"]}}]},
    {"label": "DOSAGE", "pattern": [{"TEXT": {"REGEX": r"^\d+mg$"}}]},
    {"label": "DOSAGE", "pattern": [{"TEXT": {"REGEX": r"^\d+mcg$"}}]},
    {"label": "DOSAGE", "pattern": [{"TEXT": {"REGEX": r"^\d+ml$"}}]},
    {"label": "DOSAGE", "pattern": [{"TEXT": {"REGEX": r"^\d+g$"}}]},
]


def load_training_data(data_path: str) -> list[tuple[str, dict]]:
    with open(data_path, "r") as f:
        dataset = json.load(f)

    training_data = []
    for entry in dataset:
        text = entry["text"]
        entities = [(e["start"], e["end"], e["type"]) for e in entry["entities"]]

        # filter overlapping entities — spaCy doesn't allow them
        entities.sort(key=lambda x: x[0])
        clean_entities = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                clean_entities.append((start, end, label))
                last_end = end

        training_data.append((text, {"entities": clean_entities}))

    return training_data


def create_spacy_model(entity_labels: list[str]) -> spacy.Language:
    nlp = spacy.blank("en")

    ruler = nlp.add_pipe("entity_ruler", before="ner") if "ner" in nlp.pipe_names else nlp.add_pipe("entity_ruler")
    ruler.add_patterns(ENTITY_PATTERNS)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for label in entity_labels:
        ner.add_label(label)

    return nlp


def train_spacy_ner(
    data_path: str,
    output_dir: str = "models/spacy_ner",
    n_iter: int = 30,
    dropout: float = 0.35,
):
    print("=" * 60)
    print("spaCy NER Training")
    print("=" * 60)

    training_data = load_training_data(data_path)
    print(f"Loaded {len(training_data)} training examples")

    random.seed(42)
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    entity_labels = set()
    for _, annotations in training_data:
        for _, _, label in annotations["entities"]:
            entity_labels.add(label)
    print(f"Entity types: {sorted(entity_labels)}")

    nlp = create_spacy_model(list(entity_labels))

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    except Exception:
                        continue

                if examples:
                    nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)

            if (iteration + 1) % 5 == 0 or iteration == 0:
                val_scores = evaluate_ner(nlp, val_data)
                print(
                    f"Iter {iteration+1:3d}/{n_iter} | "
                    f"Loss: {losses.get('ner', 0):.4f} | "
                    f"P: {val_scores['precision']:.4f} | "
                    f"R: {val_scores['recall']:.4f} | "
                    f"F1: {val_scores['f1']:.4f}"
                )

    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"\nModel saved to {output_dir}")

    print("\n" + "=" * 60)
    print("Final Evaluation on Validation Set")
    print("=" * 60)
    final_scores = evaluate_ner(nlp, val_data, verbose=True)

    return nlp, final_scores


def evaluate_ner(
    nlp: spacy.Language,
    eval_data: list[tuple[str, dict]],
    verbose: bool = False,
) -> dict:
    entity_metrics = {}

    for text, annotations in eval_data:
        doc = nlp(text)
        pred_entities = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        gold_entities = {(s, e, l) for s, e, l in annotations["entities"]}

        for entity_type in {e[2] for e in gold_entities | pred_entities}:
            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {"tp": 0, "fp": 0, "fn": 0}

            pred_of_type = {e for e in pred_entities if e[2] == entity_type}
            gold_of_type = {e for e in gold_entities if e[2] == entity_type}

            # relaxed matching — allow 3-char offset tolerance
            matched_gold = set()
            matched_pred = set()
            for p in pred_of_type:
                for g in gold_of_type:
                    if g not in matched_gold and abs(p[0] - g[0]) <= 3 and abs(p[1] - g[1]) <= 3:
                        matched_gold.add(g)
                        matched_pred.add(p)
                        break

            entity_metrics[entity_type]["tp"] += len(matched_pred)
            entity_metrics[entity_type]["fp"] += len(pred_of_type) - len(matched_pred)
            entity_metrics[entity_type]["fn"] += len(gold_of_type) - len(matched_gold)

    total_tp = sum(m["tp"] for m in entity_metrics.values())
    total_fp = sum(m["fp"] for m in entity_metrics.values())
    total_fn = sum(m["fn"] for m in entity_metrics.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if verbose:
        print(f"\n{'Entity Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 65)
        for entity_type, m in sorted(entity_metrics.items()):
            p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
            r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            support = m["tp"] + m["fn"]
            print(f"{entity_type:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f} {support:>10d}")
        print("-" * 65)
        print(f"{'OVERALL':<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {total_tp + total_fn:>10d}")

    return {"precision": precision, "recall": recall, "f1": f1, "per_entity": entity_metrics}


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "ner_annotations.json")
    output_dir = os.path.join(base_dir, "models", "spacy_ner")

    if not os.path.exists(data_path):
        print("NER data not found. Generating annotations first...")
        from data import save_ner_dataset
        save_ner_dataset(os.path.join(base_dir, "data"))

    nlp, scores = train_spacy_ner(data_path, output_dir)

    print("\n\nTesting on sample sentences:")
    test_sentences = [
        "Patient on metformin 1000mg for diabetes, HbA1c improved to 7.1%.",
        "CT scan shows acute appendicitis, surgical consult requested.",
        "Potassium 3.2 mEq/L, started IV potassium chloride 40 mEq.",
    ]
    for sentence in test_sentences:
        doc = nlp(sentence)
        print(f"\n  Text: {sentence}")
        for ent in doc.ents:
            print(f"    [{ent.label_:15s}] {ent.text}")
