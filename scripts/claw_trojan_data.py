import os
import re
import math
import json
import random
from collections import Counter
from pathlib import Path
import pandas as pd

# ==========================================
# 1. LINGUISTIC & NLP UTILITIES
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
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def calculate_markdown_density(text: str) -> float:
    if not text:
        return 0.0
    structural_tokens = len(re.findall(r'(```|\{|}|\bjson\b|###)', text, re.IGNORECASE))
    return (structural_tokens / max(len(text), 1)) * 100

ROLEPLAY_OVERRIDE_PATTERNS = re.compile(
    r"(pretend you are|you are now|act as|dan mode|ignore (previous|all) instructions|disregard your rules|override)", 
    re.IGNORECASE
)

PERSISTENCE_POISONING_PATTERNS = re.compile(
    r"(write to|save this rule|remember|store this|update configuration|persist|memory/)", 
    re.IGNORECASE
)

EXFILTRATION_TRIGGER_PATTERNS = re.compile(
    r"(print the system|leak api|send contents|webhook|exfiltrate|unauth_disclosure)", 
    re.IGNORECASE
)

OBFUSCATION_PATTERNS = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){2,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?|\\u[0-9a-fA-F]{4}", 
    re.IGNORECASE
)

MEMORY_DRIFT_PATTERNS = re.compile(
    r"(new rule|override active|persist|updated memory|changed preference|bypassed constraints)",
    re.IGNORECASE
)

# ==========================================
# 2. FEATURE EXTRACTION ENGINE
# ==========================================

def compute_text_metrics(text: str, base_features: dict) -> dict:
    text_len = len(text)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(text_len, 1)
    punct_density = sum(1 for c in text if c in "!?.,;:#@*[]{}") / max(text_len, 1)
    
    features = base_features.copy()
    features.update({
        "text": text,
        "text_length": text_len,
        "shannon_entropy": calculate_shannon_entropy(text),
        "payload_ttr": calculate_ttr(text),
        "caps_ratio": caps_ratio,
        "punct_density": punct_density,
        "obfuscation_flag": int(bool(OBFUSCATION_PATTERNS.search(text))),
        
        "roleplay_override_flag": int(bool(ROLEPLAY_OVERRIDE_PATTERNS.search(text))),
        "persistence_poisoning_flag": int(bool(PERSISTENCE_POISONING_PATTERNS.search(text))),
        "exfiltration_trigger_flag": int(bool(EXFILTRATION_TRIGGER_PATTERNS.search(text))),
    })
    return features

def extract_step_features(sample_dict: dict, envs_root: Path) -> list:
    extracted_records = []
    
    sample_id = sample_dict.get("sample_id", "unknown")
    step_idx = sample_dict.get("step_idx", 0)
    is_last_chance = int(sample_dict.get("is_last_chance", False))
    step_dir = envs_root / sample_id / f"step_{step_idx}"
    
    # ---------------------------------------------------------
    # NEW: Extract REAL data directly from the sandbox files
    # ---------------------------------------------------------
    
    # 1. True Conversation State (Compile session.jsonl)
    session_path = step_dir / "session.jsonl"
    conversation_texts = []
    has_tool_run = 0
    if session_path.exists():
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                for line in f:
                    log_entry = json.loads(line)
                    role = log_entry.get("role", "")
                    content = log_entry.get("content", "")
                    if role == "tool" or "tool" in content.lower():
                        has_tool_run = 1
                    conversation_texts.append(f"{role.upper()}: {content}")
        except Exception:
            pass
    conversation_state = "\n".join(conversation_texts)

    # 2. True Memory State (Read workspace/MEMORY.md and workspace/memory/*)
    memory_texts = []
    memory_main = step_dir / "workspace" / "MEMORY.md"
    if memory_main.exists():
        memory_texts.append(memory_main.read_text(encoding="utf-8", errors="ignore"))
        
    memory_dir = step_dir / "workspace" / "memory"
    if memory_dir.exists() and memory_dir.is_dir():
        for mem_file in memory_dir.glob("*.md"):
            memory_texts.append(mem_file.read_text(encoding="utf-8", errors="ignore"))
    memory_state = "\n---\n".join(memory_texts)

    # 3. True Workspace State (Read AGENTS.md, TOOLS.md, USER.md)
    workspace_texts = []
    for ws_file in ["AGENTS.md", "TOOLS.md", "USER.md"]:
        ws_path = step_dir / "workspace" / ws_file
        if ws_path.exists():
            workspace_texts.append(ws_path.read_text(encoding="utf-8", errors="ignore"))
    workspace_state = "\n---\n".join(workspace_texts)

    # ---------------------------------------------------------

    base_meta = {
        "sample_id": sample_id,
        "historical_step_depth": int(step_idx),
        "is_last_chance_window": is_last_chance,
        "history_contains_tool_execution": has_tool_run,
        "layer1_similarity_score": 0.0, 
        
        # NLP Metrics based on REAL file data
        "conversation_length": len(conversation_state),
        "conversation_ttr": calculate_ttr(conversation_state),
        "memory_drift_flag": int(bool(MEMORY_DRIFT_PATTERNS.search(memory_state))),
        "workspace_markdown_density": calculate_markdown_density(workspace_state),
        
        # Storing the raw extracted text into the CSV columns
        "raw_conversation_desc": conversation_state,
        "raw_memory_desc": memory_state,
        "raw_workspace_desc": workspace_state
    }

    # Record A: User Prompt
    user_text = sample_dict.get("user_input", "")
    if user_text:
        user_meta = base_meta.copy()
        user_meta.update({"label": 0, "is_direct_user_input": 1, "source_tier": 0})
        extracted_records.append(compute_text_metrics(user_text, user_meta))

    # Record B: Context/Tool
    is_malicious = sample_dict.get("is_malicious", False)
    metadata_fields = sample_dict.get("metadata", {})
    injection_src = metadata_fields.get("injection_src", "none")
    
    source_mapping = {"none": 0, "downloaded_file": 1, "memory": 2, "tool_return": 2, "mixed": 3}
    source_tier = source_mapping.get(injection_src, 1 if is_malicious else 0)

    if is_malicious:
        context_text = sample_dict.get("injection_text", "")
        label = 1
    else:
        context_text = sample_dict.get("tool_output", "") or sample_dict.get("clean_output", "")
        label = 0

    if context_text:
        context_meta = base_meta.copy()
        context_meta.update({"label": label, "is_direct_user_input": 0, "source_tier": source_tier})
        extracted_records.append(compute_text_metrics(context_text, context_meta))

    return extracted_records

# ==========================================
# 3. PIPELINE EXECUTION & SPLIT
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
    print(f"[*] Loading enriched step environments from {envs_root}...")
    steps = load_all_trojan_envs(str(envs_root))
    
    all_features = []
    for step_obj in steps:
        step_data = step_obj.to_dict()
        records = extract_step_features(step_data, envs_root)
        all_features.extend(records)
        
    df = pd.DataFrame(all_features)
    
    unique_samples = list(df["sample_id"].unique())
    random.seed(42)
    random.shuffle(unique_samples)
    
    split_index = int(len(unique_samples) * 0.8)
    train_samples = set(unique_samples[:split_index])
    df["dataset_split"] = df["sample_id"].apply(lambda x: "train" if x in train_samples else "test")
    
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n[+] Raw File Scrape successful. Matrix saved to: {output_csv_path}")
    print(f"[+] Total Matrix Shape: {df.shape}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "ClawTrojan"
    output_path = script_dir.parent / "data" / "processed" / "claw_shield_features.csv"
    run_extraction_pipeline(str(dataset_path), str(output_path))