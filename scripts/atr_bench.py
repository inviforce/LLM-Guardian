import json
import re
import math
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Linguistic & NLP Feature Functions  (unchanged from your original)
# ----------------------------------------------------------------------------
def calculate_shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    text = str(text)
    counts = Counter(text)
    length = len(text)
    if length == 0:
        return 0.0
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

def calculate_ttr(text: str) -> float:
    text = str(text) if text else ""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def calculate_caps_ratio(text: str) -> float:
    text = str(text) if text else ""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def calculate_punct_density(text: str) -> float:
    text = str(text) if text else ""
    if not text:
        return 0.0
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return punct / len(text)

ROLEPLAY_OVERRIDE_PATTERNS = re.compile(
    r"(ignore (all )?(previous|prior|above) (instructions|rules)|override safety protocols|act as (system|admin))",
    re.IGNORECASE)
PERSISTENCE_POISONING_PATTERNS = re.compile(
    r"(store (this|the following) (in )?memory|persist (this|the following) across sessions)", re.IGNORECASE)
EXFILTRATION_TRIGGER_PATTERNS = re.compile(r"(curl\s+http|wget\s+http|exfiltrat)", re.IGNORECASE)
OBFUSCATION_PATTERNS = re.compile(r"(base64|%[0-9a-fA-F]{2}%[0-9a-fA-F]{2})")

# New: Codex-specific / executable-risk regexes (these are the signals the
# Codex benchmark is actually built around: shell execution, destructive file
# ops, dependency/MCP supply-chain tampering)
SHELL_EXEC_PATTERNS = re.compile(
    r"\b(rm\s+-rf|chmod\s+777|curl\s+.*\|\s*sh|wget\s+.*\|\s*sh|sudo\s|:\(\)\{)", re.IGNORECASE)
DESTRUCTIVE_FS_PATTERNS = re.compile(
    r"\b(rm\s+-rf|git\s+reset\s+--hard|git\s+push\s+--force|truncate\s|shutil\.rmtree)", re.IGNORECASE)
DEP_SUPPLY_CHAIN_PATTERNS = re.compile(
    r"\b(pip install .*(git\+|http)|npm install .*(http|github\.com)|unverified package|typosquat)",
    re.IGNORECASE)

def extract_flags(text: str) -> dict:
    text = str(text) if text else ""
    return {
        "obfuscation_flag": int(bool(OBFUSCATION_PATTERNS.search(text))),
        "roleplay_override_flag": int(bool(ROLEPLAY_OVERRIDE_PATTERNS.search(text))),
        "persistence_poisoning_flag": int(bool(PERSISTENCE_POISONING_PATTERNS.search(text))),
        "exfiltration_trigger_flag": int(bool(EXFILTRATION_TRIGGER_PATTERNS.search(text))),
        "shell_exec_flag": int(bool(SHELL_EXEC_PATTERNS.search(text))),
        "destructive_fs_flag": int(bool(DESTRUCTIVE_FS_PATTERNS.search(text))),
        "dep_supply_chain_flag": int(bool(DEP_SUPPLY_CHAIN_PATTERNS.search(text))),
    }

def text_stats(text: str) -> dict:
    text = str(text) if text else ""
    return {
        "text_length": len(text),
        "shannon_entropy": calculate_shannon_entropy(text),
        "payload_ttr": calculate_ttr(text),
        "caps_ratio": calculate_caps_ratio(text),
        "punct_density": calculate_punct_density(text),
    }

def safe_string(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val)
    except Exception:
        return str(val)

# ----------------------------------------------------------------------------
# Claw parser
# ----------------------------------------------------------------------------
# Real schema: {"trajectory": {"events": [...]}, "labels": {...}, "reason": ...}
# Each element of events is: {"type": "message", "message": "<JSON-encoded str>"}
# The inner JSON string decodes to {"role": ..., "content": [ {type: text/toolCall/...} ]}
# Your original code read event.get("role") directly on the OUTER event dict,
# which does not have a "role" key -> always fell back to "unknown" and dumped
# the raw event JSON as text. Fixed here by decoding "message" first.

def parse_claw_event(event: dict) -> dict:
    if not isinstance(event, dict):
        return {"role": "unknown", "text": safe_string(event), "tool_names": []}

    msg_raw = event.get("message")
    msg = {}
    if isinstance(msg_raw, str):
        try:
            msg = json.loads(msg_raw)
        except Exception:
            msg = {}
    elif isinstance(msg_raw, dict):
        msg = msg_raw

    role = msg.get("role") or event.get("type") or "unknown"
    content = msg.get("content", "")
    text_parts = []
    tool_names = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(safe_string(block))
                continue
            btype = str(block.get("type", "")).lower()
            if btype == "text":
                text_parts.append(safe_string(block.get("text", "")))
            elif "tool" in btype or "call" in btype:
                name = block.get("name") or block.get("toolName") or ""
                args = block.get("input") or block.get("arguments") or block.get("args") or ""
                if name:
                    tool_names.append(str(name))
                text_parts.append(f"[TOOLCALL {name}] {safe_string(args)}")
            else:
                text_parts.append(safe_string(block.get("text", block)))
    elif isinstance(content, str):
        text_parts.append(content)
    else:
        text_parts.append(safe_string(content))

    return {"role": str(role), "text": " ".join(t for t in text_parts if t), "tool_names": tool_names}

def process_claw(data: list, source_tag: str) -> pd.DataFrame:
    all_rows = []
    for idx, example in enumerate(data):
        sample_id = f"{source_tag}_{idx}"
        labels = example.get("labels", {}) if isinstance(example.get("labels"), dict) else {}
        label = 0 if labels.get("is_safe", True) else 1
        risk_source = labels.get("risk_source", "")
        failure_mode = labels.get("failure_mode", "")
        harm_type = labels.get("harm_type", "")
        defense_type = labels.get("defense_type", "")

        events = []
        traj = example.get("trajectory", {})
        if isinstance(traj, dict):
            events = traj.get("events", []) or []

        rolling_conversation = []
        for step_idx, event in enumerate(events):
            parsed = parse_claw_event(event)
            role, step_text = parsed["role"], parsed["text"]
            rolling_conversation.append(f"[{role.upper()}] {step_text}")
            conv_desc = "\n".join(rolling_conversation)

            row = {
                "source": source_tag,
                "sample_id": sample_id,
                "step_index": step_idx,
                "n_steps": len(events),
                "role": role,
                "text": step_text,
                "raw_conversation_desc": conv_desc,
                "label": label,
                "risk_source": risk_source,
                "failure_mode": failure_mode,
                "harm_type": harm_type,
                "defense_type": defense_type,
                "rollout_shell_command_count": 0,
                "rollout_apply_patch_count": 0,
                "rollout_function_call_count": 0,
            }
            row.update(text_stats(step_text))
            row.update(extract_flags(step_text))
            all_rows.append(row)
    return pd.DataFrame(all_rows)

# ----------------------------------------------------------------------------
# Codex parser
# ----------------------------------------------------------------------------
# Real schema: {"id":..., "conversation":[{"role":..,"content":..}], "codex_rollout":
# [{"timestamp":..,"type": session_meta/turn_context/response_item/event_msg,"payload":{...}}],
# "risk_source":..,"failure_mode":..,"is_safe":..,"reason":..,"tool_used":[...]}
#
# codex_rollout is the executable-risk trace (shell exec, apply_patch, function
# calls) that your original script dropped entirely. We keep per-step text
# from `conversation`, and additionally compute trajectory-level rollout
# features (attached to every row) since rollout events aren't 1:1 aligned
# with conversation turns.

def codex_rollout_features(rollout: list) -> dict:
    shell_count = 0
    patch_count = 0
    func_call_count = 0
    names = []
    if isinstance(rollout, list):
        for ev in rollout:
            if not isinstance(ev, dict):
                continue
            payload = ev.get("payload", {})
            if not isinstance(payload, dict):
                continue
            ptype = str(payload.get("type", "")).lower()
            name = str(payload.get("name", "") or "")
            args = safe_string(payload.get("arguments", payload.get("command", "")))
            if ptype in ("function_call", "custom_tool_call") or ev.get("type") == "response_item" and ptype == "function_call":
                func_call_count += 1
                if name:
                    names.append(name)
            if "shell" in ptype or "exec" in ptype or SHELL_EXEC_PATTERNS.search(args):
                shell_count += 1
            if "patch" in ptype or "apply_patch" in name.lower():
                patch_count += 1
    return {
        "rollout_shell_command_count": shell_count,
        "rollout_apply_patch_count": patch_count,
        "rollout_function_call_count": func_call_count,
        "rollout_event_count": len(rollout) if isinstance(rollout, list) else 0,
    }

def process_codex(data: list, source_tag: str) -> pd.DataFrame:
    all_rows = []
    for idx, example in enumerate(data):
        sample_id = f"{source_tag}_{idx}"
        label = 0 if example.get("is_safe", True) else 1
        risk_source = example.get("risk_source", "")
        failure_mode = example.get("failure_mode", "")
        harm_type = example.get("harm_type", "")
        defense_type = example.get("defense_type", "")

        turns = example.get("conversation", []) or []
        rollout = example.get("codex_rollout", []) or []
        rfeat = codex_rollout_features(rollout)

        rolling_conversation = []
        for step_idx, turn in enumerate(turns):
            role = turn.get("role", "unknown") if isinstance(turn, dict) else "unknown"
            step_text = safe_string(turn.get("content", "")) if isinstance(turn, dict) else safe_string(turn)
            rolling_conversation.append(f"[{str(role).upper()}] {step_text}")
            conv_desc = "\n".join(rolling_conversation)

            row = {
                "source": source_tag,
                "sample_id": sample_id,
                "step_index": step_idx,
                "n_steps": len(turns),
                "role": role,
                "text": step_text,
                "raw_conversation_desc": conv_desc,
                "label": label,
                "risk_source": risk_source,
                "failure_mode": failure_mode,
                "harm_type": harm_type,
                "defense_type": defense_type,
            }
            row.update(rfeat)
            row.update(text_stats(step_text))
            row.update(extract_flags(step_text))
            all_rows.append(row)
    return pd.DataFrame(all_rows)

# ----------------------------------------------------------------------------
# Split & save
# ----------------------------------------------------------------------------
def split_and_save(df: pd.DataFrame, out_dir: Path, test_size: float):
    if df.empty:
        print("[!] Combined dataset was empty. Skipping write.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_labels = df.groupby("sample_id")["label"].first()

    try:
        train_ids, test_ids = train_test_split(
            traj_labels.index.tolist(), test_size=test_size, random_state=42, stratify=traj_labels.values)
    except Exception:
        train_ids, test_ids = train_test_split(
            traj_labels.index.tolist(), test_size=test_size, random_state=42)

    train_df = df[df["sample_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["sample_id"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    n_traj = traj_labels.shape[0]
    n_safe = int((traj_labels == 0).sum())
    n_unsafe = int((traj_labels == 1).sum())
    print(f"[+] Combined trajectories: {n_traj}  (safe={n_safe}, unsafe={n_unsafe})")
    print(f"[+] Train trajectories: {len(train_ids)}  Test trajectories: {len(test_ids)}")
    print(f"[+] Wrote train.csv / test.csv to: {out_dir}")

# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claw-json", type=str, default=None, help="Path to ATBench-Claw json")
    parser.add_argument("--codex-json", type=str, default=None, help="Path to ATBench-Codex json")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not args.claw_json and not args.codex_json:
        raise SystemExit("Provide at least one of --claw-json / --codex-json")

    frames = []
    if args.claw_json:
        with open(args.claw_json, "r", encoding="utf-8") as f:
            claw_data = json.load(f)
        claw_df = process_claw(claw_data, "claw")
        print(f"[*] ATBench-Claw: {claw_df['sample_id'].nunique()} trajectories parsed "
              f"from {len(claw_data)} raw records.")
        frames.append(claw_df)

    if args.codex_json:
        with open(args.codex_json, "r", encoding="utf-8") as f:
            codex_data = json.load(f)
        codex_df = process_codex(codex_data, "codex")
        print(f"[*] ATBench-Codex: {codex_df['sample_id'].nunique()} trajectories parsed "
              f"from {len(codex_data)} raw records.")
        frames.append(codex_df)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    split_and_save(combined, Path(args.out_dir), args.test_size)