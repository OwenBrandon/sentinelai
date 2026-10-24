"""
diagnose.py — direct test of AfroXLMR on hard sentences
=======================================================
Bypasses the API and the dashboard toggle completely. Loads the saved
AfroXLMR model straight from disk and prints its prediction + confidence
for a set of tricky sentences, including the "support the violence" family
that we suspect is a genuine gap.

Run:
    python diagnose.py
"""
import os, json
from transformers import pipeline

MODELS = "models/afroxlmr_model"
label_map = json.load(open("models/label_map.json"))

print("Loading AfroXLMR from disk...\n")
clf = pipeline("text-classification", model=MODELS, tokenizer=MODELS,
               truncation=True, max_length=96)

def predict(text):
    out = clf(text)[0]
    lab = out["label"]
    if lab not in ("sympathizer","not_sympathizer"):
        lab = label_map.get(lab.split("_")[-1], lab)
    return lab, out["score"]*100

tests = [
    # (sentence, what a human analyst would say)
    ("we support this violence from amba boys",     "sympathizer"),
    ("we support this separatist group",            "sympathizer"),
    ("we support the amba boys",                    "sympathizer"),
    ("i support this violence",                     "sympathizer"),
    ("the killing is justified they had no choice", "sympathizer"),
    ("they were right to burn the school",          "sympathizer"),
    ("the attacks on soldiers are a good thing",    "sympathizer"),
    ("we condemn this violence from amba boys",     "not_sympathizer"),
    ("the violence must stop on all sides",         "not_sympathizer"),
    ("justice for the people",                      "not_sympathizer"),
    ("god help us all",                             "not_sympathizer"),
    ("i do not support the fighters",               "not_sympathizer"),
    ("government must arrest the amba boys",         "not_sympathizer"),
    ("those boys are defending us fr",              "sympathizer"),
    # sideways-sympathy cases collected from real testing
    ("the army should leave those amba boys alone",  "sympathizer"),
    ("the army is over wicked with those amba boys", "sympathizer"),
    ("army don di over kill that bush boy them ya",  "sympathizer"),
    ("some people are sacrificing everything for our freedom", "sympathizer"),
    # contrast: army criticism WITHOUT fighter sympathy should stay neutral
    ("both the army and the boys are wicked to us",  "not_sympathizer"),
    ("i grieve for every soldier and civilian killed", "not_sympathizer"),
]

print(f"{'PRED':17s} {'CONF':>6s}  {'EXPECT':17s}  OK  TEXT")
print("-"*90)
correct=0
for text, exp in tests:
    lab, conf = predict(text)
    ok = "OK" if lab==exp else "XX"
    if lab==exp: correct+=1
    print(f"{lab:17s} {conf:5.1f}%  {exp:17s}  {ok}  {text}")
print("-"*90)
print(f"Score: {correct}/{len(tests)}")
