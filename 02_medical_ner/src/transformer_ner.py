"""
Token classification using HuggingFace Transformers for medical NER.
Fine-tunes a small BERT model on IOB-tagged clinical data.
Compared to the spaCy approach, the transformer has a better shot at
generalizing to unseen entity mentions because of its contextual embeddings.
"""
# TODO: try emilyalsentzer/Bio_ClinicalBERT for token classification — the
# domain-specific pretraining should help a lot here too

import json
import os
import sys
import numpy as np
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split


class NERDataset(Dataset):
    def __init__(
        self,
        texts: list[list[str]],
        tags: list[list[str]],
        tokenizer,
        tag2id: dict[str, int],
        max_length: int = 128,
    ):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        tag_labels = self.tags[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # align labels with wordpiece tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.tag2id.get(tag_labels[word_idx], 0))
            else:
                # for continuation wordpieces, use I- tag if the original was B-
                original_tag = tag_labels[word_idx]
                if original_tag.startswith("B-"):
                    continuation_tag = "I-" + original_tag[2:]
                    aligned_labels.append(self.tag2id.get(continuation_tag, self.tag2id.get(original_tag, 0)))
                else:
                    aligned_labels.append(self.tag2id.get(original_tag, 0))
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def build_tag_mapping(all_tags: list[list[str]]) -> tuple[dict, dict]:
    unique_tags = sorted(set(tag for tags in all_tags for tag in tags))
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    return tag2id, id2tag


def compute_entity_f1(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
    id2tag: dict[int, str],
) -> dict:
    entity_type_counts = {}

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_entities = _extract_entities_from_tags(true_seq)
        pred_entities = _extract_entities_from_tags(pred_seq)

        all_types = set(e[2] for e in true_entities) | set(e[2] for e in pred_entities)
        for entity_type in all_types:
            if entity_type not in entity_type_counts:
                entity_type_counts[entity_type] = {"tp": 0, "fp": 0, "fn": 0}

            true_of_type = {e for e in true_entities if e[2] == entity_type}
            pred_of_type = {e for e in pred_entities if e[2] == entity_type}

            tp = len(true_of_type & pred_of_type)
            entity_type_counts[entity_type]["tp"] += tp
            entity_type_counts[entity_type]["fp"] += len(pred_of_type) - tp
            entity_type_counts[entity_type]["fn"] += len(true_of_type) - tp

    results = {}
    for entity_type, counts in sorted(entity_type_counts.items()):
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[entity_type] = {"precision": p, "recall": r, "f1": f1, "support": counts["tp"] + counts["fn"]}

    return results


def _extract_entities_from_tags(tags: list[str]) -> set[tuple]:
    entities = set()
    current_entity = None
    current_start = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.add((current_start, i, current_entity))
            current_entity = tag[2:]
            current_start = i
        elif tag.startswith("I-") and current_entity == tag[2:]:
            continue
        else:
            if current_entity:
                entities.add((current_start, i, current_entity))
                current_entity = None

    if current_entity:
        entities.add((current_start, len(tags), current_entity))

    return entities


def train_transformer_ner(
    data_path: str,
    output_dir: str = "models/transformer_ner",
    model_name: str = "bert-base-uncased",
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    max_length: int = 128,
):
    print("=" * 60)
    print("Transformer NER (Token Classification)")
    print("=" * 60)

    with open(data_path, "r") as f:
        dataset = json.load(f)

    all_tokens = [d["tokens"] for d in dataset]
    all_tags = [d["tags"] for d in dataset]

    tag2id, id2tag = build_tag_mapping(all_tags)
    num_labels = len(tag2id)
    print(f"Tag set ({num_labels} tags): {list(tag2id.keys())}")

    X_train, X_test, y_train, y_test = train_test_split(
        list(range(len(all_tokens))), list(range(len(all_tags))),
        test_size=0.2, random_state=42,
    )

    train_tokens = [all_tokens[i] for i in X_train]
    train_tags = [all_tags[i] for i in X_train]
    test_tokens = [all_tokens[i] for i in X_test]
    test_tags = [all_tags[i] for i in X_test]

    print(f"Train: {len(train_tokens)} | Test: {len(test_tokens)}")

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2tag,
        label2id=tag2id,
    )

    train_dataset = NERDataset(train_tokens, train_tags, tokenizer, tag2id, max_length)
    test_dataset = NERDataset(test_tokens, test_tags, tokenizer, tag2id, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    ner_model.to(device)

    optimizer = torch.optim.AdamW(ner_model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps,
    )

    for epoch in range(num_epochs):
        ner_model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ner_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")

    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    ner_model.eval()
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu()

            for i in range(len(labels)):
                true_seq = []
                pred_seq = []
                for j in range(len(labels[i])):
                    if labels[i][j].item() != -100:
                        true_seq.append(id2tag[labels[i][j].item()])
                        pred_seq.append(id2tag[predictions[i][j].item()])
                all_true_labels.append(true_seq)
                all_pred_labels.append(pred_seq)

    entity_scores = compute_entity_f1(all_true_labels, all_pred_labels, id2tag)

    print(f"\n{'Entity Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)
    for entity_type, scores in entity_scores.items():
        print(
            f"{entity_type:<20} "
            f"{scores['precision']:>10.4f} "
            f"{scores['recall']:>10.4f} "
            f"{scores['f1']:>10.4f} "
            f"{scores['support']:>10d}"
        )

    os.makedirs(output_dir, exist_ok=True)
    ner_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {"model": model_name, "entity_scores": entity_scores}
    with open(os.path.join(output_dir, "ner_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nModel and results saved to {output_dir}")

    return results


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "ner_annotations.json")
    output_dir = os.path.join(base_dir, "models", "transformer_ner")

    if not os.path.exists(data_path):
        print("NER data not found. Generating annotations first...")
        from data import save_ner_dataset
        save_ner_dataset(os.path.join(base_dir, "data"))

    train_transformer_ner(data_path, output_dir)
