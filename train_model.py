"""
train_model.py — Fine-tune AfroXLMR for sympathizer detection
=============================================================
Why AfroXLMR (Davlan/afro-xlmr-base):
  - Pretrained on 17 African languages INCLUDING Nigerian Pidgin (Naija),
    which is very close to Cameroonian Pidgin. It already understands the
    Pidgin grammar and vocabulary your users write in.
  - Understands English too, and handles code-switching between them.
  - Deep enough (12 layers) to learn negation ("I do NOT support...").

This replaces the tiny 2-layer English-only BERT that could not understand
Pidgin or negation.

We ALSO keep the TF-IDF baseline so you can compare the two in your report.

Run:
    pip install transformers torch accelerate scikit-learn datasets
    python train_model.py
"""

import os, json, pickle, numpy as np, pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Davlan/afro-xlmr-base"   # African-language model incl. Pidgin
MAX_LEN    = 96
EPOCHS     = 8
BATCH      = 8
LR         = 2e-5
SEED       = 42

os.makedirs("models", exist_ok=True)

print("\n" + "="*60)
print("  SENTINELAI — TRAINING (AfroXLMR + TF-IDF baseline)")
print("="*60 + "\n")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/labeled_dataset.csv").dropna(subset=["text","label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 3].reset_index(drop=True)
print(f"Dataset: {len(df)} rows")
print(df["label"].value_counts().to_string(), "\n")

le = LabelEncoder()
df["y"] = le.fit_transform(df["label"])
label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
json.dump(label_map, open("models/label_map.json","w"), indent=2)
print(f"Labels: {label_map}\n")

# Stratified split (small data → 80/20 with stratify)
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["y"], random_state=SEED
)
print(f"Train: {len(train_df)}   Test: {len(test_df)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# PART A — TF-IDF + Logistic Regression baseline (for comparison in report)
# ══════════════════════════════════════════════════════════════════════════════
print("-"*60)
print("  PART A: TF-IDF + Logistic Regression baseline")
print("-"*60 + "\n")

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
Xtr = tfidf.fit_transform(train_df["text"])
Xte = tfidf.transform(test_df["text"])

lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
lr.fit(Xtr, train_df["y"])
pred_lr = lr.predict(Xte)
acc_lr  = accuracy_score(test_df["y"], pred_lr)
print(f"Baseline accuracy: {acc_lr:.1%}\n")
print(classification_report(test_df["y"], pred_lr, target_names=le.classes_, zero_division=0))

pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl","wb"))
pickle.dump(lr,    open("models/logistic_model.pkl","wb"))
pickle.dump(le,    open("models/label_encoder.pkl","wb"))
print("Baseline saved.\n")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Fine-tune AfroXLMR
# ══════════════════════════════════════════════════════════════════════════════
print("-"*60)
print(f"  PART B: Fine-tuning {MODEL_NAME}")
print("-"*60 + "\n")
print("First run downloads ~1.1GB. This is a one-time download.\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df[["text","y"]].rename(columns={"y":"labels"}))
test_ds  = Dataset.from_pandas(test_df [["text","y"]].rename(columns={"y":"labels"}))
train_ds = train_ds.map(tok, batched=True).remove_columns(["text"])
test_ds  = test_ds.map(tok, batched=True).remove_columns(["text"])
if "__index_level_0__" in train_ds.column_names:
    train_ds = train_ds.remove_columns(["__index_level_0__"])
    test_ds  = test_ds.remove_columns(["__index_level_0__"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_map),
    id2label={int(k): v for k, v in label_map.items()},
    label2id={v: int(k) for k, v in label_map.items()},
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="macro"),
    }

args = TrainingArguments(
    output_dir="models/afroxlmr_ckpt",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,          # keep only the single best checkpoint (saves disk)
    save_only_model=True,        # do NOT save bulky optimizer state (~1.8GB each)
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    seed=SEED,
    dataloader_pin_memory=False, # silences the pin_memory warning on CPU
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=test_ds,
    processing_class=tokenizer, data_collator=collator,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Training... (CPU: ~15-30 min, GPU: ~2 min)\n")
trainer.train()

res = trainer.evaluate()
print(f"\nAfroXLMR accuracy: {res.get('eval_accuracy',0):.1%}")
print(f"AfroXLMR macro-F1: {res.get('eval_f1',0):.1%}\n")

trainer.save_model("models/afroxlmr_model")
tokenizer.save_pretrained("models/afroxlmr_model")
print("AfroXLMR saved to models/afroxlmr_model/\n")

# ── Sanity test: the exact sentences that failed before ───────────────────────
print("-"*60)
print("  SANITY TEST on previously-failing sentences")
print("-"*60 + "\n")

from transformers import pipeline
clf = pipeline("text-classification", model="models/afroxlmr_model",
               tokenizer="models/afroxlmr_model", truncation=True, max_length=MAX_LEN)

tests = [
    ("I do NOT support the fighters",                         "not_sympathizer"),
    ("Government should stop the fighters and end all fight", "not_sympathizer"),
    ("People are suffering in the regions",                   "not_sympathizer"),
    ("we support the amba boys and the struggle",             "sympathizer"),
    ("make we send money go help de fighters for bush",       "sympathizer"),
    ("government must arrest all the amba boys",              "not_sympathizer"),
    ("army should protect villages from amba",                "not_sympathizer"),
    ("i no support the amba boys at all",                     "not_sympathizer"),
    ("join the amba boys fight for freedom",                  "sympathizer"),
    ("we need peace abeg no more fighting",                   "not_sympathizer"),
    # implicit support cases (the new focus)
    ("those boys are defending us fr",                        "sympathizer"),
    ("at least the boys are doing something",                 "sympathizer"),
    ("you cannot blame the boys after what they did to us",   "sympathizer"),
    ("the bush boys deserve our backup",                      "sympathizer"),
    # contrast cases that look similar but are NOT support
    ("those boys are destroying us not defending us",         "not_sympathizer"),
    ("the boys are kidnapping our children for ransom",       "not_sympathizer"),
]

correct = 0
for text, expected in tests:
    out = clf(text)[0]
    # Model now returns proper class names directly (id2label set above)
    pred = out["label"]
    if pred not in ("sympathizer", "not_sympathizer"):
        # fallback if a generic LABEL_x slips through
        lid = pred.split("_")[-1]
        pred = label_map.get(lid, pred)
    ok = "OK " if pred == expected else "XX "
    if pred == expected: correct += 1
    print(f"{ok}[{out['score']*100:5.1f}%] pred={pred:16s} expected={expected:16s}")
    print(f"    \"{text}\"")
print(f"\nSanity score: {correct}/{len(tests)} correct\n")

# ── Save report ───────────────────────────────────────────────────────────────
report = f"""SENTINELAI EVALUATION REPORT
============================
Dataset: {len(df)} rows ({len(train_df)} train / {len(test_df)} test)
Base model: {MODEL_NAME}

TF-IDF + Logistic Regression baseline
  Accuracy: {acc_lr:.1%}

AfroXLMR (fine-tuned)
  Accuracy: {res.get('eval_accuracy',0):.1%}
  Macro-F1: {res.get('eval_f1',0):.1%}

Sanity test on previously-failing sentences: {correct}/{len(tests)}
"""
open("models/evaluation_report.txt","w").write(report)
print("Report saved to models/evaluation_report.txt")
print("\n" + "="*60)
print("  TRAINING COMPLETE — next: restart  python api/app.py")
print("="*60)
