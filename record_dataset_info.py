"""
record_dataset_info.py — write dataset/model metadata into the database
========================================================================
Reads data/labeled_dataset.csv, counts the classes, and records the
metadata so the dashboard / API /dataset/info endpoint can show
"trained on N examples, X/Y split, model = afroxlmr".

Run after training:
    python record_dataset_info.py
"""
import pandas as pd
import database as db

db.init_db()
df = pd.read_csv("data/labeled_dataset.csv")
counts = df["label"].value_counts().to_dict()
total = len(df)
symp = int(counts.get("sympathizer", 0))
nots = int(counts.get("not_sympathizer", 0))

db.set_dataset_info(total, symp, nots, "afroxlmr")
print(f"Recorded dataset info: {total} rows ({symp} sympathizer / {nots} not_sympathizer), model=afroxlmr")
print("The /dataset/info endpoint will now show this.")
