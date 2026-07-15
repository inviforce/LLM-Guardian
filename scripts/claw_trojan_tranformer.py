"""
ClawTrojan extraction pipeline for the Transformer (sequence-tagging) stream.

Changes from the previous version:
  1. Same border/negative harvesting as the LightGBM extractor.
  2. NO trajectory-level label aggregation (no more `.max()`). Every record
     keeps its own step-level label. This is what lets the model later output
     a running per-step risk score instead of one verdict per trajectory.
  3. NO deduplication by historical_step_depth. Both the user-prompt record
     and the context/tool-output record for each step are kept as separate
     sequence positions, in chronological order. This is what gives you
     "exact prompt where it went wrong" resolution instead of "somewhere in
     this trajectory."
  4. Computes the same 9 engineered features (entropy, regex flags, etc.) on
     each record's own text, to be concatenated onto that record's embedding
     at training time (embedding: 1152 dims -> +9 engineered = 1161 dims).
"""

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ==========================================
# Same feature functions as the LightGBM extractor -- kept duplicated here
# so this file can run standalone. If you prefer, delete this block and
# `from claw_trojan_lightgbm_extract import compute_text_metrics_only` instead.
# ==========================================

def calculate_shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    frequencies = [count / len(text) for count in counts.values()]
    return -sum(p * math.log2(p) for p in frequencies)


def calculate_ttr(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    return len(set(words)) / len(words) if words else 0.0


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


def compute_engineered_features(text: str) -> dict:
    t_str = str(text) if text else ""
    t_len = max(len(t_str), 1)
    return {
        "text_length": len(t_str),
        "shannon_entropy": calculate_shannon_entropy(t_str),
        "payload_ttr": calculate_ttr(t_str),
        "caps_ratio": sum(1 for c in t_str if c.isupper()) / t_len,
        "punct_density": sum(1 for c in t_str if c in "!?.,;:#@*[]{}") / t_len,
        "obfuscation_flag": int(bool(OBFUSCATION_PATTERNS.search(t_str))),
        "roleplay_override_flag": int(bool(ROLEPLAY_OVERRIDE_PATTERNS.search(t_str))),
        "persistence_poisoning_flag": int(bool(PERSISTENCE_POISONING_PATTERNS.search(t_str))),
        "exfiltration_trigger_flag": int(bool(EXFILTRATION_TRIGGER_PATTERNS.search(t_str))),
    }


def harvest_border_negative_ids(repo_path: Path) -> set:
    samples_dir = repo_path / "samples"
    border_ids = set()
    if not samples_dir.exists():
        print(f"[!] No samples/ directory found at {samples_dir} -- "
              f"falling back to per-step outcome_category/attack_type only.")
        return border_ids

    json_files = list(samples_dir.rglob("*.json")) + list(samples_dir.rglob("*.jsonl"))
    for f in json_files:
        try:
            if f.suffix == ".jsonl":
                entries = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                data = json.loads(f.read_text(encoding="utf-8"))
                entries = data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"[!] Could not parse {f}: {e}")
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sample_id = entry.get("sample_id")
            goal_text = str(entry.get("goal", "") or entry.get("attack_goal", "")).strip().lower()
            outcome = str(entry.get("outcome_category", "")).strip().lower()
            attack_type = str(entry.get("attack_type", "")).strip().lower()
            if sample_id and (
                goal_text.startswith("border sample:")
                or outcome == "none"
                or attack_type == "none"
            ):
                border_ids.add(sample_id)

    print(f"[+] Harvested {len(border_ids)} border/negative sample_ids from samples/")
    return border_ids


# ==========================================
# FEATURE EXTRACTION (per-record, sequence-preserving)
# ==========================================

def extract_transformer_features(sample_dict: dict, envs_root: Path, border_ids: set) -> list:
    sample_id = sample_dict.get("sample_id", "unknown")
    step_idx = int(sample_dict.get("step_idx", 0))
    step_dir = envs_root / sample_id / f"step_{step_idx}"

    is_border_negative = (
        sample_id in border_ids
        or str(sample_dict.get("outcome_category", "")).lower() == "none"
        or str(sample_dict.get("attack_type", "")).lower() == "none"
    )

    # 1. Conversation state
    session_path = step_dir / "session.jsonl"
    conversation_texts = []
    if session_path.exists():
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                for line in f:
                    log_entry = json.loads(line)
                    role = log_entry.get("role", "")
                    content = log_entry.get("content", "")
                    conversation_texts.append(f"{role.upper()}: {content}")
        except Exception:
            pass
    conversation_state = "\n".join(conversation_texts)

    # 2. Memory state
    memory_texts = []
    memory_main = step_dir / "workspace" / "MEMORY.md"
    if memory_main.exists():
        memory_texts.append(memory_main.read_text(encoding="utf-8", errors="ignore"))
    memory_dir = step_dir / "workspace" / "memory"
    if memory_dir.exists() and memory_dir.is_dir():
        for mem_file in memory_dir.glob("*.md"):
            memory_texts.append(mem_file.read_text(encoding="utf-8", errors="ignore"))
    memory_state = "\n---\n".join(memory_texts)

    # 3. Workspace state
    workspace_texts = []
    for ws_file in ["AGENTS.md", "TOOLS.md", "USER.md"]:
        ws_path = step_dir / "workspace" / ws_file
        if ws_path.exists():
            workspace_texts.append(ws_path.read_text(encoding="utf-8", errors="ignore"))
    workspace_state = "\n---\n".join(workspace_texts)

    base_meta = {
        "sample_id": sample_id,
        "historical_step_depth": step_idx,
        "raw_conversation_desc": conversation_state,
        "raw_memory_desc": memory_state,
        "raw_workspace_desc": workspace_state,
    }

    records = []

    # Record A: User Prompt -- kept as its own sequence position
    user_text = sample_dict.get("user_input", "")
    if user_text:
        meta_a = base_meta.copy()
        meta_a.update({"text": user_text, "label": 0, "is_direct_user_input": 1})
        meta_a.update(compute_engineered_features(user_text))
        records.append(meta_a)

    # Record B: Context/Tool Payload -- kept as its own sequence position
    is_malicious = bool(sample_dict.get("is_malicious", False)) and not is_border_negative
    if is_malicious:
        context_text = sample_dict.get("injection_text", "")
        label = 1
    else:
        context_text = sample_dict.get("tool_output", "") or sample_dict.get("clean_output", "")
        label = 0

    if context_text:
        meta_b = base_meta.copy()
        meta_b.update({"text": context_text, "label": label, "is_direct_user_input": 0})
        meta_b.update(compute_engineered_features(context_text))
        records.append(meta_b)

    return records


def run_extraction_pipeline(repo_dir_path: str, output_csv_path: str):
    repo_path = Path(repo_dir_path)
    sys.path.insert(0, str(repo_path))

    try:
        from loader import load_all_trojan_envs
    except ImportError:
        raise ImportError(f"Could not find loader.py inside {repo_path}.")

    envs_root = repo_path / "envs"
    print(f"[*] Loading deep step environments from {envs_root}...")
    steps = load_all_trojan_envs(str(envs_root))
    print(f"[+] Loaded {len(steps)} raw step objects.")

    border_ids = harvest_border_negative_ids(repo_path)

    all_features = []
    for step_obj in steps:
        records = extract_transformer_features(step_obj.to_dict(), envs_root, border_ids)
        all_features.extend(records)

    df = pd.DataFrame(all_features)

    print("\n[*] Per-record (step-level) class balance:")
    print(df["label"].value_counts())
    print(f"[*] Unique trajectories: {df['sample_id'].nunique()}")
    print(f"[*] Total sequence positions (records): {len(df)}")

    # Trajectory-grouped split, synchronized with the LightGBM extractor
    unique_samples = list(df["sample_id"].unique())
    random.seed(42)  # MUST match the LightGBM extraction script
    random.shuffle(unique_samples)

    split_index = int(len(unique_samples) * 0.8)
    train_samples = set(unique_samples[:split_index])
    df["dataset_split"] = df["sample_id"].apply(lambda x: "train" if x in train_samples else "test")

    assert (
        set(df[df["dataset_split"] == "train"]["sample_id"])
        & set(df[df["dataset_split"] == "test"]["sample_id"])
        == set()
    ), "Trajectory leakage detected between train and test splits!"

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[+] Extraction complete. Sequence data saved to: {output_csv_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "ClawTrojan"
    output_path = script_dir.parent / "data" / "processed_prod_null" / "transformer_sequence_data_v2.csv"
    run_extraction_pipeline(str(dataset_path), str(output_path))