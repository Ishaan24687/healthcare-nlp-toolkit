"""
Fine-tune Bio_ClinicalBERT for clinical note urgency classification.
Bio_ClinicalBERT crushed the generic BERT here — domain-specific pretraining
really matters for medical text. Trained on MIMIC-III clinical notes, it already
understands abbreviations and clinical language patterns.
"""
# TODO: try PubMedBERT and compare — it's trained on abstracts rather than
# clinical notes, so it might do worse on this task but better on research text

import json
import os
import sys
import numpy as np
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score


class ClinicalBERTDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_bert_classifier(
    data_path: str,
    output_dir: str = "models",
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
):
    print("=" * 60)
    print("Bio_ClinicalBERT Fine-tuning")
    print("=" * 60)

    with open(data_path, "r") as f:
        dataset = json.load(f)

    texts = [d["text"] for d in dataset]
    labels = [d["label_id"] for d in dataset]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clinical_bert = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    train_dataset = ClinicalBERTDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = ClinicalBERTDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = ClinicalBERTDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    clinical_bert.to(device)

    optimizer = torch.optim.AdamW(
        clinical_bert.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"Total steps: {total_training_steps} | Warmup: {warmup_steps}")

    best_val_f1 = 0.0
    training_history = []

    for epoch in range(num_epochs):
        clinical_bert.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = clinical_bert(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clinical_bert.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)

        train_accuracy = correct / total

        clinical_bert.eval()
        val_preds, val_true = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)

                outputs = clinical_bert(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_labels.cpu().numpy())

        val_f1 = f1_score(val_true, val_preds, average="weighted")
        val_accuracy = accuracy_score(val_true, val_preds)

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
            "val_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        }
        training_history.append(epoch_stats)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_stats['train_loss']:.4f} | "
            f"Val Loss: {epoch_stats['val_loss']:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(output_dir, exist_ok=True)
            clinical_bert.save_pretrained(os.path.join(output_dir, "bert_model"))
            tokenizer.save_pretrained(os.path.join(output_dir, "bert_model"))
            print(f"  -> New best model saved (F1: {val_f1:.4f})")

    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    clinical_bert.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = clinical_bert(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch_labels.cpu().numpy())

    test_accuracy = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="weighted")

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print("\nClassification Report:")
    report = classification_report(test_true, test_preds, target_names=["routine", "urgent"])
    print(report)

    results = {
        "model": "Bio_ClinicalBERT",
        "accuracy": float(test_accuracy),
        "f1_weighted": float(test_f1),
        "best_val_f1": float(best_val_f1),
        "training_history": training_history,
        "report": classification_report(test_true, test_preds, target_names=["routine", "urgent"], output_dict=True),
    }
    with open(os.path.join(output_dir, "bert_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/bert_results.json")

    return results


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "clinical_notes.json")
    output_dir = os.path.join(base_dir, "models")

    if not os.path.exists(data_path):
        print("Data not found. Generating dataset first...")
        from data import generate_dataset, save_dataset
        dataset = generate_dataset(n_samples=500)
        save_dataset(dataset, os.path.join(base_dir, "data"))

    train_bert_classifier(data_path, output_dir)
