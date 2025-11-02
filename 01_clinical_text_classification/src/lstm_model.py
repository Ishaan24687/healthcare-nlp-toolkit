"""
BiLSTM classifier for clinical note urgency prediction.
Embedding → BiLSTM → Dense → Sigmoid. The bidirectional setup helps because
clinical context can appear anywhere in the note — "code blue" at the end of a
paragraph is just as urgent as at the beginning.
"""

import json
import os
import sys
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import tokenize, remove_stopwords


class ClinicalNoteDataset(Dataset):
    def __init__(self, encoded_texts: list[list[int]], labels: list[int], max_len: int = 128):
        self.encoded_texts = encoded_texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.encoded_texts[idx][:self.max_len]
        padding_length = self.max_len - len(tokens)
        padded = tokens + [0] * padding_length

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)

        # simple attention — weight each timestep by learned importance
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        output = self.fc(self.dropout(context))
        return output.squeeze(-1)


def build_vocabulary(texts: list[list[str]], min_freq: int = 2) -> dict[str, int]:
    counter = Counter(token for tokens in texts for token in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counter.most_common():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_texts(tokenized_texts: list[list[str]], vocab: dict[str, int]) -> list[list[int]]:
    unk_id = vocab["<UNK>"]
    return [[vocab.get(t, unk_id) for t in tokens] for tokens in tokenized_texts]


def train_lstm(data_path: str, output_dir: str = "models"):
    print("=" * 60)
    print("BiLSTM Clinical Note Classifier")
    print("=" * 60)

    with open(data_path, "r") as f:
        dataset = json.load(f)

    texts = [d["text"] for d in dataset]
    labels = [d["label_id"] for d in dataset]

    print(f"\nTokenizing {len(texts)} clinical notes...")
    tokenized = [remove_stopwords(tokenize(clinical_note)) for clinical_note in texts]

    X_train_tok, X_test_tok, y_train, y_test = train_test_split(
        tokenized, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vocab = build_vocabulary(X_train_tok, min_freq=2)
    print(f"Vocabulary size: {len(vocab)}")

    X_train_enc = encode_texts(X_train_tok, vocab)
    X_test_enc = encode_texts(X_test_tok, vocab)

    max_sequence_length = 128
    train_dataset = ClinicalNoteDataset(X_train_enc, y_train, max_len=max_sequence_length)
    test_dataset = ClinicalNoteDataset(X_test_enc, y_test, max_len=max_sequence_length)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    clinical_classifier = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in clinical_classifier.parameters())
    print(f"Model parameters: {total_params:,}")

    # weighted loss to handle class imbalance
    pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(clinical_classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    num_epochs = 20
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        clinical_classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = clinical_classifier(batch_texts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clinical_classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)

        train_accuracy = correct / total

        clinical_classifier.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                batch_texts = batch_texts.to(device)
                logits = clinical_classifier(batch_texts)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_labels.numpy())

        val_f1 = f1_score(val_true, val_preds, average="weighted")
        val_accuracy = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(output_dir, exist_ok=True)
            torch.save(clinical_classifier.state_dict(), os.path.join(output_dir, "lstm_model.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {total_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

    clinical_classifier.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_texts, batch_labels in test_loader:
            batch_texts = batch_texts.to(device)
            logits = clinical_classifier(batch_texts)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch_labels.numpy())

    all_preds = [int(p) for p in all_preds]
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="weighted")

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final Test F1 (weighted): {f1:.4f}")
    print("\nClassification Report:")
    report = classification_report(all_true, all_preds, target_names=["routine", "urgent"])
    print(report)

    results = {
        "model": "BiLSTM",
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "best_val_f1": float(best_val_f1),
        "vocab_size": len(vocab),
        "total_params": total_params,
        "report": classification_report(all_true, all_preds, target_names=["routine", "urgent"], output_dict=True),
    }
    with open(os.path.join(output_dir, "lstm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/lstm_results.json")

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

    train_lstm(data_path, output_dir)
