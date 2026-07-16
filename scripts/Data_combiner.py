import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Target Phase 1 Schema
SCHEMA_COLS = ["stream", "text", "paired_text", "label", "source", "sample_id", "step_index"]

def process_prodnull(train_path: str, test_path: str, out_dir: str):
    """Formats ProdNull (repo_file) train and test sets."""
    os.makedirs(out_dir, exist_ok=True)
    
    for split_path, split_name in [(train_path, "train.csv"), (test_path, "test.csv")]:
        df = pd.read_csv(split_path)
        df["stream"] = "repo_file"
        df["paired_text"] = None
        df["sample_id"] = None
        df["step_index"] = None
        
        # Enforce strict schema and save
        df = df[SCHEMA_COLS]
        df.to_csv(os.path.join(out_dir, split_name), index=False)
        print(f"[+] Saved repo_file {split_name} | Shape: {df.shape}")

def process_bipia(raw_path: str, out_dir: str):
    """Formats BIPIA (indirect_context) and creates an 80/20 stratified split."""
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"context": "text", "user_intent": "paired_text"})
    
    df["stream"] = "indirect_context"
    df["sample_id"] = None
    df["step_index"] = None
    
    df = df[SCHEMA_COLS]
    
    # 80/20 split stratifying by the label to maintain the 50/50 balance
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"[+] Saved indirect_context train.csv | Shape: {train_df.shape}")
    print(f"[+] Saved indirect_context test.csv | Shape: {test_df.shape}")

def process_clawshield(raw_path: str, out_dir: str):
    """Formats ClawShield (agent_session) using its pre-computed trajectory split."""
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"historical_step_depth": "step_index"})
    
    df["stream"] = "agent_session"
    df["paired_text"] = None
    df["source"] = "ClawShield"
    
    # Separate the data safely by the trajectory dataset_split
    train_full = df[df["dataset_split"] == "train"].copy()
    test_full = df[df["dataset_split"] == "test"].copy()
    
    # 1. Save Text Contract (Core schema for Transformers/Cross-Encoders)
    train_full[SCHEMA_COLS].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_full[SCHEMA_COLS].to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"[+] Saved agent_session text train.csv | Shape: {train_full[SCHEMA_COLS].shape}")
    print(f"[+] Saved agent_session text test.csv | Shape: {test_full[SCHEMA_COLS].shape}")
    
    # 2. Save Tabular Features (For LightGBM Layer 2a)
    tabular_cols = [c for c in df.columns if c not in ["stream", "text", "paired_text", "source", "dataset_split"]]
    tabular_df = df[tabular_cols]
    tabular_df.to_csv(os.path.join(out_dir, "tabular_features.csv"), index=False)
    print(f"[+] Saved agent_session tabular_features.csv | Shape: {tabular_df.shape}")


if __name__ == "__main__":
    # ==========================================
    # FILE PATH PLACEHOLDERS
    # ==========================================
    
    # Inputs
    PRODNULL_TRAIN_CSV = "data/processed_prod_null/train.csv"
    PRODNULL_TEST_CSV  = "data/processed_prod_null/test.csv"
    BIPIA_RAW_CSV      = "data/processed_prod_null/bipia_raw.csv"
    CLAWSHIELD_RAW_CSV = "data/processed_prod_null/claw_shield_features.csv"
    
    # Output Directories (Following Phase 0 architecture)
    PRODNULL_OUT_DIR   = "m_data/processed/repo_file"
    BIPIA_OUT_DIR      = "m_data/processed/indirect_context"
    CLAWSHIELD_OUT_DIR = "m_data/processed/agent_session"
    
    # Execution
    print("--- Processing ProdNull ---")
    process_prodnull(PRODNULL_TRAIN_CSV, PRODNULL_TEST_CSV, PRODNULL_OUT_DIR)
    
    print("\n--- Processing BIPIA ---")
    process_bipia(BIPIA_RAW_CSV, BIPIA_OUT_DIR)
    
    print("\n--- Processing ClawShield ---")
    process_clawshield(CLAWSHIELD_RAW_CSV, CLAWSHIELD_OUT_DIR)