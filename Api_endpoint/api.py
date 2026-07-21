"""
LLM Guardian API
================
Multi-pipeline prompt-injection / malicious-content classifier.

Pipelines:
  - repo_file        -> LightGBM (PCA+features) AND Cross-Encoder   [dual model]
  - indirect_context  -> LightGBM (PCA+features) AND DeBERTa-LoRA    [dual model]
  - trajectory         -> Qwen2.5-1.5B LoRA (generative classifier)

NOTE on CROSS_PIPELINE_MODE:
  By default (True), Cross-Encoder is applied to repo_file text and
  DeBERTa-LoRA is applied to indirect_context text, per explicit request.
  This is an out-of-distribution use of both models (they were originally
  trained on the other pipeline's data shape). Set CROSS_PIPELINE_MODE=False
  to restore the original, matched pairing:
      repo_file        -> LightGBM + DeBERTa-LoRA (fallback ensemble)
      indirect_context  -> LightGBM + Cross-Encoder (fallback ensemble)
"""

import os

# --- Thread/segfault safety fixes (keep before heavy imports) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import re
import math
import time
import joblib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

MODEL_PATHS = {
    # repo_file pipeline
    "repo_lightgbm": MODELS_DIR / "saved_repo_file_model" / "lightgbm.pkl",
    "repo_pca": MODELS_DIR / "saved_repo_file_model" / "pca.pkl",
    "repo_feature_columns": MODELS_DIR / "saved_repo_file_model" / "feature_columns.pkl",
    "repo_sentence_transformer": MODELS_DIR / "saved_repo_file_model" / "sentence_transformer",

    # indirect_context pipeline
    "indirect_lightgbm": MODELS_DIR / "saved_indirect_context_model" / "lightgbm.pkl",
    "indirect_pca_text": MODELS_DIR / "saved_indirect_context_model" / "pca_text.pkl",
    "indirect_pca_pair": MODELS_DIR / "saved_indirect_context_model" / "pca_paired.pkl",
    "indirect_feature_columns": MODELS_DIR / "saved_indirect_context_model" / "feature_columns.pkl",
    "indirect_sentence_transformer": MODELS_DIR / "saved_indirect_context_model" / "sentence_transformer",

    # shared transformer models
    "cross_encoder": MODELS_DIR / "cross_encoder_weights_bipa(indirect_context)",
    "deberta_lora": MODELS_DIR / "deberta_lora_weights_chunked_repo_prodnull",
    "qwen_lora": MODELS_DIR / "atbench_lora_final_v3_qwen",
}

# Toggle this to control which secondary model pairs with which pipeline.
# True  = repo_file->Cross-Encoder, indirect_context->DeBERTa-LoRA  (as requested)
# False = repo_file->DeBERTa-LoRA (original), indirect_context->Cross-Encoder (original)
CROSS_PIPELINE_MODE = False

# Generic instruction used to pair repo_file text with the Cross-Encoder,
# since Cross-Encoder expects a (user_intent, context) pair and repo_file
# only has a single text field.
GENERIC_REPO_QUERY = (
    "Evaluate this repository file content for hidden prompt injection, "
    "privilege escalation, or malicious instructions."
)

CONFIDENCE_THRESHOLD = 0.85
ENSEMBLE_WEIGHTS = {
    "repo_file": {"lightgbm": 0.6, "secondary": 0.4},        # secondary = cross_encoder or deberta
    "indirect_context": {"lightgbm": 0.35, "secondary": 0.65},
}

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE_LLM = torch.float16
    DEVICE_MAP = "auto"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE_LLM = torch.float32
    DEVICE_MAP = None
else:
    DEVICE = "cpu"
    DTYPE_LLM = torch.float32
    DEVICE_MAP = None

# ============================================================
# FEATURE EXTRACTION
# ============================================================

ROLEPLAY_OVERRIDE_PATTERNS = re.compile(
    r"(pretend you are|you are now|act as|dan mode|ignore (previous|all) instructions|disregard your rules|override)",
    re.IGNORECASE,
)
PERSISTENCE_POISONING_PATTERNS = re.compile(
    r"(write to|save this rule|remember|store this|update configuration|persist|memory/)",
    re.IGNORECASE,
)
EXFILTRATION_TRIGGER_PATTERNS = re.compile(
    r"(print the system|leak api|send contents|webhook|exfiltrate|unauth_disclosure)",
    re.IGNORECASE,
)
OBFUSCATION_PATTERNS = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){2,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?|\\u[0-9a-fA-F]{4}",
    re.IGNORECASE,
)


def calc_features(text: str) -> Dict[str, Any]:
    t = str(text) if not pd.isna(text) else ""
    t_len = max(len(t), 1)
    words = re.findall(r"\b\w+\b", t.lower())
    counts = Counter(t)
    freqs = [c / t_len for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in freqs) if freqs else 0.0
    ttr = len(set(words)) / len(words) if words else 0.0
    return {
        "text_length": len(t),
        "shannon_entropy": entropy,
        "payload_ttr": ttr,
        "caps_ratio": sum(1 for c in t if c.isupper()) / t_len,
        "punct_density": sum(1 for c in t if c in "!?.,;:#@*[]{}") / t_len,
        "obfuscation_flag": int(bool(OBFUSCATION_PATTERNS.search(t))),
        "roleplay_override_flag": int(bool(ROLEPLAY_OVERRIDE_PATTERNS.search(t))),
        "persistence_poisoning_flag": int(bool(PERSISTENCE_POISONING_PATTERNS.search(t))),
        "exfiltration_trigger_flag": int(bool(EXFILTRATION_TRIGGER_PATTERNS.search(t))),
    }


def _label(x: int) -> str:
    return "Malicious" if int(x) == 1 else "Benign"


# ============================================================
# GUARDIAN CORE
# ============================================================

class LLMGuardian:
    def __init__(self, model_paths: Dict[str, Path] = MODEL_PATHS, device: str = DEVICE):
        self.device = device
        self.load_status: Dict[str, bool] = {}
        print(f"[*] Initializing LLMGuardian on device: {self.device}")

        # ---------- Shared sentence embedder(s) ----------
        # Prefer pipeline-specific saved encoders if present, else fall back
        # to the base pretrained model.
        self.encoder_repo = self._safe_load(
            "repo_sentence_transformer",
            lambda p: SentenceTransformer(str(p) if p.exists() else "all-MiniLM-L6-v2", device=self.device),
        )
        self.encoder_indirect = self._safe_load(
            "indirect_sentence_transformer",
            lambda p: SentenceTransformer(str(p) if p.exists() else "all-MiniLM-L6-v2", device=self.device),
        )

        # ---------- repo_file: LightGBM + PCA ----------
        self.repo_lgbm = self._safe_load("repo_lightgbm", joblib.load)
        self.repo_pca = self._safe_load("repo_pca", joblib.load)
        self.repo_feature_cols = self._safe_load("repo_feature_columns", joblib.load)

        # ---------- indirect_context: LightGBM + PCA ----------
        self.indirect_lgbm = self._safe_load("indirect_lightgbm", joblib.load)
        self.indirect_pca_text = self._safe_load("indirect_pca_text", joblib.load)
        self.indirect_pca_pair = self._safe_load("indirect_pca_pair", joblib.load)
        self.indirect_feature_cols = self._safe_load("indirect_feature_columns", joblib.load)

        # ---------- Cross-Encoder ----------
        self.tokenizer_ce = self._safe_load(
            "cross_encoder", lambda p: AutoTokenizer.from_pretrained(p)
        )
        self.cross_encoder = self._safe_load(
            "cross_encoder",
            lambda p: AutoModelForSequenceClassification.from_pretrained(
                p, torch_dtype=torch.float32
            ).to(self.device).eval(),
        )

        # ---------- DeBERTa-LoRA ----------
        self.tokenizer_deb = self._safe_load(
            "deberta_lora", lambda p: AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
        )
        self.deberta_lora = self._safe_load(
            "deberta_lora",
            lambda p: PeftModel.from_pretrained(
                AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/deberta-v3-small", num_labels=2, torch_dtype=torch.float32
                ),
                p,
            ).to(self.device).eval(),
        )

        # ---------- Qwen-LoRA (trajectory) ----------
        self.tokenizer_qwen = self._safe_load(
            "qwen_lora", lambda p: AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        )
        if self.tokenizer_qwen is not None and self.tokenizer_qwen.pad_token is None:
            self.tokenizer_qwen.pad_token = self.tokenizer_qwen.eos_token

        def _load_qwen(p):
            kwargs = {"torch_dtype": DTYPE_LLM}
            if DEVICE_MAP:
                kwargs["device_map"] = DEVICE_MAP
            base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", **kwargs)
            if DEVICE_MAP is None:
                base = base.to(self.device)
            return PeftModel.from_pretrained(base, p).eval()

        self.qwen_lora = self._safe_load("qwen_lora", _load_qwen)

        print(f"[*] Load status: {self.load_status}")

    def _safe_load(self, key: str, loader):
        path = MODEL_PATHS[key]
        try:
            obj = loader(path)
            self.load_status[key] = True
            return obj
        except Exception as e:
            print(f"[!] Failed to load '{key}' from {path}: {e}")
            self.load_status[key] = False
            return None

    def health(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "cross_pipeline_mode": CROSS_PIPELINE_MODE,
            "models_loaded": self.load_status,
            "all_ready": all(self.load_status.values()) if self.load_status else False,
        }

    # ------------------------------------------------------
    # LOW-LEVEL MODEL RUNNERS (each returns a standardized dict)
    # ------------------------------------------------------

    def _lgbm_result(self, model_name: str, proba: np.ndarray, latency_ms: float) -> Dict[str, Any]:
        pred_class = int(np.argmax(proba))
        return {
            "model_name": model_name,
            "prediction": _label(pred_class),
            "confidence": float(proba[pred_class]),
            "raw_probabilities": {"Benign": float(proba[0]), "Malicious": float(proba[1])},
            "latency_ms": round(latency_ms, 2),
        }

    def _run_lightgbm_repo(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        feats = calc_features(text)
        feats_df = pd.DataFrame([feats])
        feats_df = feats_df.reindex(columns=self.repo_feature_cols[: len(feats_df.columns)], fill_value=0)

        emb = self.encoder_repo.encode([text], show_progress_bar=False)
        pca_emb = self.repo_pca.transform(emb)
        pca_df = pd.DataFrame(pca_emb, columns=[f"pca_text_{i}" for i in range(50)])
        X = pd.concat([feats_df, pca_df], axis=1)

        proba = self.repo_lgbm.predict_proba(X)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        return self._lgbm_result("lightgbm_repo_file", proba, latency_ms)

    def _run_lightgbm_indirect(self, context: str, user_intent: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        feats = calc_features(context)
        feats_df = pd.DataFrame([feats])
        feats_df = feats_df.reindex(columns=self.indirect_feature_cols[: len(feats_df.columns)], fill_value=0)

        text_emb = self.encoder_indirect.encode([context], show_progress_bar=False)
        pair_emb = self.encoder_indirect.encode([user_intent], show_progress_bar=False)
        pca_text = self.indirect_pca_text.transform(text_emb)
        pca_pair = self.indirect_pca_pair.transform(pair_emb)
        pca_text_df = pd.DataFrame(pca_text, columns=[f"pca_txt_{i}" for i in range(50)])
        pca_pair_df = pd.DataFrame(pca_pair, columns=[f"pca_pair_{i}" for i in range(50)])

        X = pd.concat([feats_df, pca_text_df, pca_pair_df], axis=1)
        proba = self.indirect_lgbm.predict_proba(X)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        return self._lgbm_result("lightgbm_indirect_context", proba, latency_ms)

    def _run_cross_encoder(self, sequence_a: str, sequence_b: str, model_name: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        inputs = self.tokenizer_ce(
            sequence_a, sequence_b,
            return_tensors="pt", truncation=True, padding=True, max_length=384,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.cross_encoder(**inputs)
        proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        pred_class = int(np.argmax(proba))
        return {
            "model_name": model_name,
            "prediction": _label(pred_class),
            "confidence": float(proba[pred_class]),
            "raw_probabilities": {"Benign": float(proba[0]), "Malicious": float(proba[1])},
            "latency_ms": round(latency_ms, 2),
        }

    def _run_deberta_lora(self, text: str, model_name: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        inputs = self.tokenizer_deb(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.deberta_lora(**inputs)
        proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        pred_class = int(np.argmax(proba))
        return {
            "model_name": model_name,
            "prediction": _label(pred_class),
            "confidence": float(proba[pred_class]),
            "raw_probabilities": {"Benign": float(proba[0]), "Malicious": float(proba[1])},
            "latency_ms": round(latency_ms, 2),
        }

    def _run_qwen_trajectory(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt = self.tokenizer_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(prompt) > 6000:
            prompt = "...[TRUNCATED FOR LENGTH]...\n" + prompt[-6000:]

        inputs = self.tokenizer_qwen(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
            self.qwen_lora.device
        )
        with torch.no_grad():
            output = self.qwen_lora.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer_qwen.eos_token_id,
            )
        input_len = inputs.input_ids.shape[-1]
        generated = output[0][input_len:]
        prediction_text = self.tokenizer_qwen.decode(generated, skip_special_tokens=True).strip().lower()

        if "malicious" in prediction_text:
            pred = 1
        elif "benign" in prediction_text:
            pred = 0
        else:
            pred = 1 if "mal" in prediction_text else 0

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "model_name": "qwen_lora_trajectory",
            "prediction": _label(pred),
            "confidence": 1.0,  # generative model: no direct logit confidence available
            "raw_probabilities": None,
            "raw_generated_text": prediction_text,
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------
    # ENSEMBLE COMBINATION
    # ------------------------------------------------------

    def _ensemble(self, pipeline: str, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = ENSEMBLE_WEIGHTS.get(pipeline, {"lightgbm": 0.5, "secondary": 0.5})
        w_map = {}
        for r in model_results:
            if "lightgbm" in r["model_name"]:
                w_map[r["model_name"]] = weights["lightgbm"]
            else:
                w_map[r["model_name"]] = weights["secondary"]

        combined_benign, combined_malicious = 0.0, 0.0
        total_weight = 0.0
        for r in model_results:
            if r.get("raw_probabilities") is None:
                continue
            w = w_map.get(r["model_name"], 1.0 / len(model_results))
            combined_benign += w * r["raw_probabilities"]["Benign"]
            combined_malicious += w * r["raw_probabilities"]["Malicious"]
            total_weight += w

        if total_weight > 0:
            combined_benign /= total_weight
            combined_malicious /= total_weight
        else:
            combined_benign, combined_malicious = 0.5, 0.5

        final_pred = "Malicious" if combined_malicious >= combined_benign else "Benign"
        final_conf = max(combined_benign, combined_malicious)

        preds = [r["prediction"] for r in model_results]
        agreement = len(set(preds)) == 1

        return {
            "prediction": final_pred,
            "confidence": round(float(final_conf), 4),
            "agreement": agreement,
            "method": f"weighted_average({weights})",
        }

    # ------------------------------------------------------
    # PUBLIC PIPELINE ENTRYPOINTS
    # ------------------------------------------------------

    def predict_repo_file(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = []

        lgbm_result = self._run_lightgbm_repo(text)
        results.append(lgbm_result)

        note = None
        if CROSS_PIPELINE_MODE:
            ce_result = self._run_cross_encoder(GENERIC_REPO_QUERY, text, "cross_encoder_on_repo_file")
            note = (
                "Cross-Encoder was trained on (user_intent, context) indirect_context data. "
                "Here it is applied out-of-distribution to repo_file text, paired with a "
                "generic instruction query. Treat its standalone score with caution."
            )
        else:
            ce_result = self._run_deberta_lora(text, "deberta_lora_repo_file")
        results.append(ce_result)

        ensemble = self._ensemble("repo_file", results)
        features = calc_features(text)

        return {
            "pipeline": "repo_file",
            "final_prediction": ensemble["prediction"],
            "final_confidence": ensemble["confidence"],
            "ensemble": ensemble,
            "models": results,
            "features": features,
            "note": note,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict_indirect_context(self, context: str, user_intent: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = []

        lgbm_result = self._run_lightgbm_indirect(context, user_intent)
        results.append(lgbm_result)

        note = None
        if CROSS_PIPELINE_MODE:
            combined_text = f"User Intent: {user_intent}\nContext: {context}"
            deb_result = self._run_deberta_lora(combined_text, "deberta_lora_on_indirect_context")
            note = (
                "DeBERTa-LoRA was trained on single-sequence repo_file text. Here it is "
                "applied out-of-distribution to a concatenated (user_intent + context) "
                "string. Treat its standalone score with caution."
            )
        else:
            deb_result = self._run_cross_encoder(user_intent, context, "cross_encoder_indirect_context")
        results.append(deb_result)

        ensemble = self._ensemble("indirect_context", results)
        features = calc_features(context)

        return {
            "pipeline": "indirect_context",
            "final_prediction": ensemble["prediction"],
            "final_confidence": ensemble["confidence"],
            "ensemble": ensemble,
            "models": results,
            "features": features,
            "note": note,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict_trajectory(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self._run_qwen_trajectory(messages)
        return {
            "pipeline": "trajectory",
            "final_prediction": result["prediction"],
            "final_confidence": result["confidence"],
            "ensemble": {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "agreement": True,
                "method": "single_model",
            },
            "models": [result],
            "features": None,
            "note": "Trajectory pipeline uses a single generative model (Qwen-LoRA); no ensemble available.",
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" in request and isinstance(request["messages"], list):
            return self.predict_trajectory(request["messages"])
        elif "context" in request and "user_intent" in request:
            return self.predict_indirect_context(request["context"], request["user_intent"])
        elif "text" in request:
            return self.predict_repo_file(request["text"])
        else:
            raise ValueError(
                "Invalid request format. Provide one of: "
                "'text' (repo_file), 'context'+'user_intent' (indirect_context), "
                "or 'messages' (trajectory)."
            )


# ============================================================
# FASTAPI APP
# ============================================================

guardian: Optional[LLMGuardian] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global guardian
    print("[*] Loading LLMGuardian models...")
    guardian = LLMGuardian()
    print("[*] LLMGuardian ready.")
    yield
    print("[*] Shutting down.")


app = FastAPI(title="LLM Guardian API", version="2.0.0", lifespan=lifespan)


# ---------------- Pydantic Schemas ----------------

class RepoFileRequest(BaseModel):
    text: str = Field(..., description="Raw repository file content to classify.")


class IndirectContextRequest(BaseModel):
    context: str = Field(..., description="External/untrusted content the agent processed.")
    user_intent: str = Field(..., description="The legitimate task the user requested.")


class Message(BaseModel):
    role: str
    content: str


class TrajectoryRequest(BaseModel):
    messages: List[Message]


class GenericRequest(BaseModel):
    text: Optional[str] = None
    context: Optional[str] = None
    user_intent: Optional[str] = None
    messages: Optional[List[Message]] = None


class ModelPrediction(BaseModel):
    model_name: str
    prediction: str
    confidence: float
    raw_probabilities: Optional[Dict[str, float]] = None
    raw_generated_text: Optional[str] = None
    latency_ms: float


class EnsembleResult(BaseModel):
    prediction: str
    confidence: float
    agreement: bool
    method: str


class DetailedPredictionResponse(BaseModel):
    pipeline: str
    final_prediction: str
    final_confidence: float
    ensemble: EnsembleResult
    models: List[ModelPrediction]
    features: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    total_latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    device: str
    cross_pipeline_mode: bool
    models_loaded: Dict[str, bool]
    all_ready: bool


# ---------------- Endpoints ----------------

@app.get("/health", response_model=HealthResponse)
async def health():
    if guardian is None:
        raise HTTPException(status_code=503, detail="Guardian not initialized yet.")
    h = guardian.health()
    return HealthResponse(
        status="ok" if h["all_ready"] else "degraded",
        device=h["device"],
        cross_pipeline_mode=h["cross_pipeline_mode"],
        models_loaded=h["models_loaded"],
        all_ready=h["all_ready"],
    )


@app.post("/guard/repo_file", response_model=DetailedPredictionResponse)
async def guard_repo_file(request: RepoFileRequest):
    try:
        return guardian.predict_repo_file(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/guard/indirect_context", response_model=DetailedPredictionResponse)
async def guard_indirect_context(request: IndirectContextRequest):
    try:
        return guardian.predict_indirect_context(request.context, request.user_intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/guard/trajectory", response_model=DetailedPredictionResponse)
async def guard_trajectory(request: TrajectoryRequest):
    try:
        messages = [m.model_dump() for m in request.messages]
        return guardian.predict_trajectory(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/guard", response_model=DetailedPredictionResponse)
async def guard_generic(request: GenericRequest):
    """
    Generic dispatch endpoint. Auto-detects pipeline based on payload shape:
      - {"text": "..."}                         -> repo_file
      - {"context": "...", "user_intent": "..."} -> indirect_context
      - {"messages": [...]}                      -> trajectory
    """
    try:
        payload = request.model_dump(exclude_none=True)
        if "messages" in payload:
            payload["messages"] = [dict(m) for m in payload["messages"]]
        return guardian.predict(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)