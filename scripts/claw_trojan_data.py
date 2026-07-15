"""
ClawTrojan extraction pipeline for the LightGBM (agent_session) stream.

Changes from the previous "sanitized" version:
  1. Explicitly harvests border/negative trajectories (outcome_category == "none",
     attack_type == "none", or a sample-level goal starting with "border sample:")
     and force-labels every record from those trajectories as benign.
  2. Prints a diagnostic breakdown so you can SEE how many negative/border
     trajectories were found before training on them -- if this number is still
     tiny, the fix isn't in this script, it's that the raw dataset genuinely
     doesn't have many negatives and you need to look elsewhere (or synthesize).
  3. Keeps the same "no leaky metadata" discipline as before: only text-derived
     structural features go into the model. Provenance flags (is_border_negative)
     are used ONLY to build/verify labels, never fed in as a model feature.
"""

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ==========================================
# 1. LINGUISTIC & NLP UTILITIES (unchanged)
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


def compute_text_metrics(text: str, base_meta: dict) -> dict:
    """The 9 engineered features. Pure function of the text -- no leakage risk."""
    t_str = str(text) if text else ""
    t_len = max(len(t_str), 1)

    row = base_meta.copy()
    row.update(
        {
            "text": t_str,
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
    )
    return row


# ==========================================
# 2. BORDER / NEGATIVE SAMPLE HARVESTING (new)
# ==========================================

def harvest_border_negative_ids(repo_path: Path) -> set:
    """
    Scans samples/ for trajectory-level metadata identifying negative/border
    samples via the 'goal' field (README: 'goal' for negative/border samples,
    vs. 'attack_goal' for positive/attack samples).

    This is a SUPPLEMENT to the per-step outcome_category/attack_type check
    already available from loader.py -- both signals get merged in the main
    pipeline below. If your samples/ layout differs from the assumption here
    (one JSON per trajectory, or a JSON-lines file, or nested subfolders),
    adjust the glob pattern / parsing below accordingly.
    """
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
# 3. FEATURE EXTRACTION ENGINE
# ==========================================

def extract_step_features(sample_dict: dict, border_ids: set) -> list:
    records = []

    sample_id = sample_dict.get("sample_id", "unknown")
    is_border_negative = (
        sample_id in border_ids
        or str(sample_dict.get("outcome_category", "")).lower() == "none"
        or str(sample_dict.get("attack_type", "")).lower() == "none"
    )

    base_meta = {
        "sample_id": sample_id,
        "step_idx": int(sample_dict.get("step_idx", 0)),
        "stream": "agent_session",
    }

    # Record A: User Prompt (always benign in this dataset by construction)
    user_text = sample_dict.get("user_input", "")
    if user_text:
        meta_a = base_meta.copy()
        meta_a["label"] = 0
        records.append(compute_text_metrics(user_text, meta_a))

    # Record B: Context/Tool Payload
    is_malicious = bool(sample_dict.get("is_malicious", False)) and not is_border_negative
    if is_malicious:
        context_text = sample_dict.get("injection_text", "")
        label = 1
    else:
        context_text = (
            sample_dict.get("tool_output", "") or sample_dict.get("clean_output", "")
        )
        label = 0

    if context_text:
        meta_b = base_meta.copy()
        meta_b["label"] = label
        records.append(compute_text_metrics(context_text, meta_b))

    return records


# ==========================================
# 4. PIPELINE EXECUTION & TRAJECTORY SPLIT
# ==========================================

def run_extraction_pipeline(repo_dir_path: str, output_csv_path: str):
    import sys

    repo_path = Path(repo_dir_path)
    sys.path.insert(0, str(repo_path))

    try:
        from loader import load_all_trojan_envs
    except ImportError:
        raise ImportError(f"Could not find loader.py inside {repo_path}.")

    envs_root = repo_path / "envs"
    print(f"[*] Loading step environments from {envs_root}...")
    steps = load_all_trojan_envs(str(envs_root))
    print(f"[+] Loaded {len(steps)} raw step objects across all trajectories.")

    border_ids = harvest_border_negative_ids(repo_path)

    all_features = []
    for step_obj in steps:
        records = extract_step_features(step_obj.to_dict(), border_ids)
        all_features.extend(records)

    df = pd.DataFrame(all_features)

    # ---------------------------------------------------------
    # Diagnostics -- confirm the harvest actually changed the balance
    # ---------------------------------------------------------
    print("\n[*] Class balance after harvesting:")
    print(df["label"].value_counts())
    unique_traj = df["sample_id"].nunique()
    print(f"[*] Unique trajectories represented: {unique_traj}")

    # ---------------------------------------------------------
    # Strict trajectory-level split (grouped by sample_id, never by row)
    # ---------------------------------------------------------
    unique_samples = list(df["sample_id"].unique())
    random.seed(42)
    random.shuffle(unique_samples)

    split_index = int(len(unique_samples) * 0.8)
    train_samples = set(unique_samples[:split_index])
    df["dataset_split"] = df["sample_id"].apply(lambda x: "train" if x in train_samples else "test")

    print("\n[*] Split sizes (rows):")
    print(df["dataset_split"].value_counts())
    print("\n[*] Label balance within TRAIN split:")
    print(df[df["dataset_split"] == "train"]["label"].value_counts())
    print("\n[*] Label balance within TEST split:")
    print(df[df["dataset_split"] == "test"]["label"].value_counts())

    assert (
        set(df[df["dataset_split"] == "train"]["sample_id"])
        & set(df[df["dataset_split"] == "test"]["sample_id"])
        == set()
    ), "Trajectory leakage detected between train and test splits!"

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[+] Extraction complete. Matrix saved to: {output_csv_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "ClawTrojan"
    output_path = script_dir.parent / "data" / "processed_prod_null" / "claw_shield_features_v2.csv"
    run_extraction_pipeline(str(dataset_path), str(output_path))