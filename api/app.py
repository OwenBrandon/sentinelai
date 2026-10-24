"""
app.py — SentinelAI API (AfroXLMR + confidence-threshold tiers)
===============================================================
Models:
  - baseline : TF-IDF + Logistic Regression (fast, for comparison)
  - afroxlmr : Fine-tuned AfroXLMR (understands English + Pidgin + negation)

PRIORITY 1 — Confidence Threshold System
  Each prediction returns an operational ACTION, not just a label:
    HIGH confidence sympathizer  -> "flag"   (red,    high risk)
    MEDIUM confidence            -> "review" (amber,  needs human)
    LOW confidence               -> "ignore" (grey,   uncertain/neutral)
  Thresholds are configurable via /config.
"""

import os, sys, json, pickle, logging
from datetime import datetime
from typing import List, Optional
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db   # SQLite persistence layer

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sentinel")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

# ── Confidence thresholds (Priority 1) ────────────────────────────────────────
CONFIG = {
    "high_threshold": 85.0,   # >= this AND sympathizer  -> flag
    "low_threshold":  65.0,   # <  this                  -> ignore (uncertain)
                              # between the two           -> review
}

# ── Load baseline ─────────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR,"tfidf_vectorizer.pkl"),"rb") as f: tfidf = pickle.load(f)
with open(os.path.join(MODELS_DIR,"logistic_model.pkl"),"rb") as f: lr_model = pickle.load(f)
with open(os.path.join(MODELS_DIR,"label_map.json")) as f: label_map = json.load(f)
log.info("Baseline model loaded.")

# ── Load AfroXLMR ─────────────────────────────────────────────────────────────
afro_pipe = None
def load_afro():
    global afro_pipe
    path = os.path.join(MODELS_DIR, "afroxlmr_model")
    if not os.path.exists(path):
        log.warning("AfroXLMR not found at %s", path); return False
    try:
        from transformers import pipeline
        afro_pipe = pipeline("text-classification", model=path, tokenizer=path,
                             truncation=True, max_length=96)
        log.info("AfroXLMR model loaded."); return True
    except Exception as e:
        log.warning("Could not load AfroXLMR: %s", e); return False

afro_available = load_afro()
active_model = {"name": "afroxlmr" if afro_available else "baseline"}

# ── LIME explainer (Priority 2 — explainability) ──────────────────────────────
# LIME perturbs the input (removes words) and watches how the prediction shifts,
# producing a per-word contribution score. Positive = pushed toward sympathizer,
# negative = pushed toward not_sympathizer.
lime_explainer = None
try:
    from lime.lime_text import LimeTextExplainer
    # class order must match label_map: index 0 = not_sympathizer, 1 = sympathizer
    ordered_classes = [label_map[str(i)] for i in range(len(label_map))]
    lime_explainer = LimeTextExplainer(
        class_names=ordered_classes,
        bow=False,            # respect word order (model is contextual, not bag-of-words)
        split_expression=r"\s+",
    )
    log.info("LIME explainer ready (classes=%s).", ordered_classes)
except Exception as e:
    log.warning("LIME not available (%s). /explain will be disabled.", e)

def _proba_fn(texts):
    """Return an (n, 2) probability array for a list of texts, for LIME.
    Column order matches label_map: [not_sympathizer, sympathizer]."""
    import numpy as _np
    out = []
    if active_model["name"] == "afroxlmr" and afro_available:
        # ask the pipeline for ALL class scores
        results = afro_pipe(list(texts), top_k=None, truncation=True, max_length=96)
        for scores in results:
            row = {s["label"]: s["score"] for s in scores}
            # normalise label names (handle LABEL_x fallback)
            norm = {}
            for k, v in row.items():
                kk = k if k in ("sympathizer", "not_sympathizer") else label_map.get(k.split("_")[-1], k)
                norm[kk] = v
            out.append([norm.get("not_sympathizer", 0.0), norm.get("sympathizer", 0.0)])
    else:
        vecs = tfidf.transform(list(texts))
        probs = lr_model.predict_proba(vecs)  # columns already in label order
        out = probs.tolist()
    return _np.array(out)

# ── State ─────────────────────────────────────────────────────────────────────
# Predictions now persist in SQLite (survives restarts). We keep a small
# in-memory deque only as a convenience cache; the database is source of truth.
db.init_db()
log.info("Database initialised (data/sentinel.db).")
history = deque(maxlen=500)

app = FastAPI(title="SentinelAI API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TextInput(BaseModel):
    text: str
    source: Optional[str] = "manual"
    model: Optional[str] = None
    author: Optional[str] = None      # username/handle from source data
    platform: Optional[str] = None    # platform the statement was made on

class BatchItem(BaseModel):
    text: str
    author: Optional[str] = None
    platform: Optional[str] = None

class BatchInput(BaseModel):
    # accepts either plain strings (legacy) or {text,author,platform} items
    texts: List = []
    source: Optional[str] = "manual"
    model: Optional[str] = None

class ModelSwitch(BaseModel):
    model: str

class ConfigUpdate(BaseModel):
    high_threshold: Optional[float] = None
    low_threshold: Optional[float] = None

# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_baseline(text):
    vec = tfidf.transform([text])
    pid = int(lr_model.predict(vec)[0])
    conf = float(np.max(lr_model.predict_proba(vec)[0]))*100
    return label_map[str(pid)], conf

def predict_afro(text):
    out = afro_pipe(text[:500])[0]
    label = out["label"]
    if label not in ("sympathizer", "not_sympathizer"):
        # fallback for generic LABEL_x
        lid = label.split("_")[-1]
        label = label_map.get(lid, label)
    return label, out["score"]*100

def decide_action(label, confidence):
    """Priority 1: map (label, confidence) -> operational action + risk."""
    is_symp = label == "sympathizer"
    if is_symp and confidence >= CONFIG["high_threshold"]:
        return "flag",   "high"
    if confidence < CONFIG["low_threshold"]:
        return "ignore", "low"      # too uncertain to act on
    # medium band, or high-confidence not_sympathizer
    if is_symp:
        return "review", "medium"
    return "ignore", "low"

def classify(text, source="manual", model_name=None, author=None, platform=None):
    text = text.strip()
    if not text: raise ValueError("Empty text.")
    used = model_name or active_model["name"]
    if used == "afroxlmr" and afro_available:
        label, conf = predict_afro(text)
    else:
        label, conf = predict_baseline(text); used = "baseline"

    action, risk = decide_action(label, conf)
    result = {
        "text": text, "label": label, "confidence": round(conf,2),
        "is_sympathizer": label=="sympathizer",
        "action": action, "risk_level": risk,
        "model_used": used, "source": source,
        "author": author or "—", "platform": platform or source,
        "timestamp": datetime.utcnow().isoformat()+"Z",
    }
    db.log_prediction(result)   # persist to SQLite
    history.append(result)      # in-memory convenience cache
    return result

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status":"online","version":"3.0.0","active_model":active_model["name"],
            "afroxlmr_available":afro_available,"config":CONFIG}

@app.get("/config")
def get_config(): return CONFIG

@app.post("/config")
def set_config(c: ConfigUpdate):
    if c.high_threshold is not None: CONFIG["high_threshold"] = c.high_threshold
    if c.low_threshold  is not None: CONFIG["low_threshold"]  = c.low_threshold
    return CONFIG

@app.get("/model/info")
def model_info():
    return {"active_model":active_model["name"],"afroxlmr_available":afro_available,
            "models":{"baseline":{"type":"TF-IDF + Logistic Regression"},
                      "afroxlmr":{"type":"Fine-tuned AfroXLMR (EN+Pidgin)","available":afro_available}}}

@app.post("/model/switch")
def switch(b: ModelSwitch):
    if b.model not in ["baseline","afroxlmr"]:
        raise HTTPException(400,"model must be baseline or afroxlmr")
    if b.model=="afroxlmr" and not afro_available:
        raise HTTPException(400,"AfroXLMR not available — run train_model.py")
    active_model["name"]=b.model
    return {"active_model":b.model}

@app.post("/predict")
def predict(b: TextInput):
    try:
        r = classify(b.text,b.source,b.model,b.author,b.platform)
        log.info("[%s] %s %.1f%% -> %s | %s", r["model_used"],r["label"],r["confidence"],r["action"],r["text"][:50])
        return r
    except ValueError as e: raise HTTPException(400,str(e))

@app.post("/explain")
def explain(b: TextInput):
    """Priority 2: explain WHY a text was classified the way it was.
    Returns per-word contributions: positive = toward sympathizer,
    negative = toward not_sympathizer. Powered by LIME."""
    text = b.text.strip()
    if not text:
        raise HTTPException(400, "Empty text.")
    if lime_explainer is None:
        raise HTTPException(503, "LIME explainer not available on this server.")

    # First get the actual prediction so the UI can show label + confidence
    result = classify(text, b.source, b.model)

    try:
        # num_samples kept modest so it stays responsive on CPU
        exp = lime_explainer.explain_instance(
            text, _proba_fn,
            num_features=min(12, len(text.split())),
            num_samples=200,
            labels=(1,),   # explain w.r.t. the 'sympathizer' class (index 1)
        )
        # list of (word, weight); weight>0 pushes toward sympathizer
        contributions = [
            {"word": w, "weight": round(float(wt), 4)}
            for w, wt in exp.as_list(label=1)
        ]
    except Exception as e:
        log.error("LIME failed: %s", e)
        raise HTTPException(500, f"Explanation failed: {e}")

    return {
        **result,
        "explanation": contributions,
        "explanation_note": "weight > 0 pushes toward sympathizer; < 0 toward not_sympathizer",
    }

@app.post("/predict/batch")
def predict_batch(b: BatchInput):
    if len(b.texts)>250: raise HTTPException(400,"max 250 texts per chunk")
    out=[]
    for item in b.texts:
        # item may be a plain string (legacy) or a dict {text,author,platform}
        if isinstance(item, str):
            text, author, platform = item, None, None
        elif isinstance(item, dict):
            text     = item.get("text","")
            author   = item.get("author")
            platform = item.get("platform")
        else:
            out.append({"text":str(item),"error":"bad item"}); continue
        try:
            out.append(classify(text, b.source, b.model, author, platform))
        except ValueError:
            out.append({"text":text,"error":"empty"})
    return {"results":out,"count":len(out)}

@app.get("/stats")
def get_stats():
    s = db.get_stats()
    tot = s["total_predictions"] or 1
    return {**s,
            "active_model":active_model["name"],
            "sympathizer_rate":round(s["sympathizer_count"]/tot*100,1),
            "not_sympathizer_rate":round(s["not_sympathizer_count"]/tot*100,1),
            "flag_rate":round(s["flag_count"]/tot*100,1),
            "review_rate":round(s["review_count"]/tot*100,1)}

@app.get("/history")
def get_history(limit:int=50):
    rows = db.get_predictions(limit=min(limit,500))
    # add is_sympathizer for dashboard compatibility
    for r in rows: r["is_sympathizer"] = (r["label"]=="sympathizer")
    return {"predictions":rows,"count":len(rows)}

@app.get("/history/flagged")
def get_flagged(limit:int=50):
    rows = db.get_predictions(limit=min(limit,500), only_action="flag")
    for r in rows: r["is_sympathizer"] = (r["label"]=="sympathizer")
    return {"predictions":rows,"count":len(rows)}

@app.get("/history/review")
def get_review(limit:int=50):
    rows = db.get_predictions(limit=min(limit,500), only_action="review")
    for r in rows: r["is_sympathizer"] = (r["label"]=="sympathizer")
    return {"predictions":rows,"count":len(rows)}

@app.get("/dataset/info")
def dataset_info():
    """Dataset usage metadata for the supervisor's requirement."""
    info = db.get_dataset_info()
    if not info:
        return {"message":"No dataset metadata recorded yet. Run record_dataset_info.py after training."}
    return info

@app.post("/admin/clear")
def clear_log():
    """Wipe the prediction log (for clean demos)."""
    db.clear_predictions()
    history.clear()
    return {"message":"Prediction log cleared."}

if __name__=="__main__":
    import uvicorn
    log.info("Active model: %s", active_model["name"])
    log.info("Confidence tiers: flag>=%.0f%%  review %.0f-%.0f%%  ignore<%.0f%%",
             CONFIG["high_threshold"],CONFIG["low_threshold"],CONFIG["high_threshold"],CONFIG["low_threshold"])
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)
