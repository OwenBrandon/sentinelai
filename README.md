# SentinelAI — English & Cameroon Pidgin Text Classification

An NLP system that classifies short social-media-style text in **English and Cameroon Pidgin English** as expressing support for, or opposition to, armed separatist and bandit groups. It is built as a **decision-support tool for human analysts**: it returns a confidence score, a risk band and a per-word explanation, and routes uncertain cases to manual review. It never takes an action on its own.

It includes the full pipeline:

**dataset construction → baseline → transformer fine-tuning → evaluation → REST API → analyst dashboard**

> B.Tech final-year project, 2026.

---

## Results

Evaluated on a stratified 80/20 split (seed 42): 964 train / 242 test.

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| TF-IDF (1–2 grams) + Logistic Regression | **98.8%** | **0.988** | Mean 98.2% over 10 random splits (range 95.9–99.2%) |
| Fine-tuned AfroXLMR-base (`Davlan/afro-xlmr-base`) | 97.1% | 0.971 | Best epoch selected on the same held-out split |
| AfroXLMR on the 16-case diagnostic suite | 15 / 16 | — | Hand-picked hard cases (see `diagnose.py`) |

### What the numbers mean

**The simple baseline matches or beats the transformer here, and that is an expected result rather than a failure.** The dataset is mostly built from controlled templates (see below), so the classes are well separated by keywords such as "support", "condemn" and "denounce". A bag-of-n-grams model exploits that very efficiently.

Where a pretrained multilingual model *should* help is on sentences where the keywords mislead, such as negation ("you cannot blame the boys…"), indirect support, and Pidgin phrasing. A random test split barely measures those cases. That is why the project includes a **diagnostic suite** and an **error-collection loop** (`collect_misses.py`) instead of relying on a single accuracy number.

**Caveats, stated plainly:**
- 1,206 examples is a small dataset, and a 242-example test set gives roughly ±1–2% uncertainty.
- AfroXLMR's checkpoint was selected on the same split it is scored on, so its number is slightly optimistic.
- Template-generated text is cleaner than real social media. Performance on real posts is expected to be lower and has not been measured.

---

## Dataset

`data/labeled_dataset.csv`: **1,206 rows** (606 `sympathizer` / 600 `not_sympathizer`), in English and Cameroon Pidgin.

It was built by `prepare_data.py` from three sources:
1. **Curated seed sentences**, written by hand as an anchor quality set.
2. **An intent-aware template generator**, which combines real vocabulary (subjects, supportive or oppositional verbs, local euphemisms for fighters, Pidgin connectors) across sentence templates. Supportive templates only draw supportive verbs, so every label is correct by construction.
3. **Collected misses** (`data/misses.csv`): real, human-verified errors made by the model, folded back in.

The script then removes duplicates, drops any sentence that appears in both classes, and trims the classes to the same size.

The dataset contains **no real user posts, usernames or personal data**.

---

## Architecture

```
 Analyst ──► Dashboard (dashboard/index.html, vanilla JS)
                 │
                 ▼
          FastAPI service (api/app.py)
           ├─ TF-IDF + Logistic Regression  (always available)
           ├─ AfroXLMR pipeline             (if weights present)
           ├─ LIME word-level explanations
           ├─ Confidence routing: flag / review / ignore
           └─ SQLite persistence (database.py → data/sentinel.db)
```

### Confidence routing
| Confidence | Prediction | Outcome |
|---|---|---|
| ≥ 85% | sympathizer | **Flag** for an analyst |
| 65–85% | any | **Review** queue |
| < 65% | any | Ignored as uncertain |

You can change the thresholds at runtime through `/config`.

### API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET / POST | `/config` | Read or update the confidence thresholds |
| GET | `/model/info` | Active model and availability |
| POST | `/model/switch` | Switch between baseline and AfroXLMR |
| POST | `/predict` | Classify one text |
| POST | `/predict/batch` | Classify up to 100 texts |
| POST | `/explain` | LIME per-word contributions |
| GET | `/stats` | Prediction statistics |
| GET | `/history` | Recent predictions |
| GET | `/history/flagged` | Flagged predictions |
| GET | `/history/review` | Review queue |
| GET | `/dataset/info` | Training-data metadata |
| POST | `/admin/clear` | Clear history |

---

## Quick start

```bash
pip install -r requirements.txt

# Run the API (the baseline model is included, so this works immediately)
cd api
python app.py            # http://localhost:8000  ·  docs at /docs
```

Then open `dashboard/index.html` in a browser.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "we anglophone denounce de amba boys"}'
```

### Reproduce everything
```bash
python prepare_data.py        # regenerate data/labeled_dataset.csv
python train_model.py         # retrain the baseline and fine-tune AfroXLMR (CPU: ~15–30 min)
python diagnose.py            # run the 16-case diagnostic suite
python record_dataset_info.py # record dataset metadata for the dashboard
python collect_misses.py      # interactively log model errors to data/misses.csv
```

### Model weights
The fine-tuned AfroXLMR weights (~1.1 GB) are too large for GitHub and are not included. You can regenerate them with `train_model.py`. Without them, the API runs on the baseline model.
<!-- Add a Hugging Face link here once the weights are uploaded -->

---

## Repository layout

```
api/app.py              FastAPI service
dashboard/index.html    Analyst dashboard
database.py             SQLite persistence layer
prepare_data.py         Dataset construction (seed + generator + misses)
train_model.py          Baseline + AfroXLMR training and evaluation
diagnose.py             Hard-case diagnostic suite
collect_misses.py       Interactive error collection
record_dataset_info.py  Writes dataset metadata for the dashboard
data/                   labeled_dataset.csv, misses.csv
models/                 Baseline model, label map, evaluation report
```

---

## Responsible use

Classifying political speech can cause real harm if it's misused. This system:
- **Supports human judgement and does not replace it.** No prediction should lead to action against anyone without human review.
- **Will produce false positives.** Strong words, sarcasm and reporting about violence can all be misread.
- **Has not been validated on real social media data** and should not be used operationally until it has been.
- Should only be used within the law and with respect for freedom of expression.

---

## Tech stack

Python · scikit-learn · Hugging Face Transformers · PyTorch · AfroXLMR · LIME · FastAPI · SQLite · HTML/CSS/JS

## Author

**Owan Brandon Leku**, Software Engineer, Buea, Cameroon
[GitHub](https://github.com/OwenBrandon)
