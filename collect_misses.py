"""
collect_misses.py — disciplined error collection
=================================================
Interactive tool. Type a sentence; it shows AfroXLMR's prediction.
You say whether it's right. If wrong, it logs the sentence + the
correct label to data/misses.csv.

After collecting a batch (aim for 20-40), we analyse the patterns
together, fix the dataset once, and retrain once.

Run:
    python collect_misses.py

Commands while running:
    (just type a sentence to test it)
    y   = model was right (not logged)
    s   = model was WRONG, correct label is sympathizer
    n   = model was WRONG, correct label is not_sympathizer
    q   = quit and show summary
"""
import os, json, csv
from transformers import pipeline

MODELS = "models/afroxlmr_model"
label_map = json.load(open("models/label_map.json"))
MISS_FILE = "data/misses.csv"

print("Loading AfroXLMR...\n")
clf = pipeline("text-classification", model=MODELS, tokenizer=MODELS,
               truncation=True, max_length=96)

def predict(text):
    out = clf(text)[0]
    lab = out["label"]
    if lab not in ("sympathizer","not_sympathizer"):
        lab = label_map.get(lab.split("_")[-1], lab)
    return lab, out["score"]*100

# Make sure miss file has a header
new_file = not os.path.exists(MISS_FILE)
os.makedirs("data", exist_ok=True)
f = open(MISS_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(f)
if new_file:
    writer.writerow(["text", "label", "model_said", "confidence"])

logged = 0
print("="*60)
print("MISS COLLECTOR  —  type a sentence, then judge the prediction")
print("  y = right    s = wrong/should be sympathizer")
print("  n = wrong/should be not_sympathizer    q = quit")
print("="*60 + "\n")

while True:
    text = input("sentence> ").strip()
    if not text:
        continue
    if text.lower() == "q":
        break

    lab, conf = predict(text)
    print(f"   MODEL: {lab}  ({conf:.1f}%)")
    verdict = input("   right? [y/s/n/q]> ").strip().lower()

    if verdict == "q":
        break
    elif verdict == "y":
        print("   ok, correct.\n")
    elif verdict == "s":
        writer.writerow([text, "sympathizer", lab, round(conf,1)]); f.flush()
        logged += 1
        print(f"   logged as sympathizer (miss #{logged})\n")
    elif verdict == "n":
        writer.writerow([text, "not_sympathizer", lab, round(conf,1)]); f.flush()
        logged += 1
        print(f"   logged as not_sympathizer (miss #{logged})\n")
    else:
        print("   (skipped — unrecognised input)\n")

f.close()
print(f"\nCollected {logged} misses this session.")
print(f"All misses saved in: {MISS_FILE}")
print("When you have 20-40, share the file and we fix the patterns together.")
