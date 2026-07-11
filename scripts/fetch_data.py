#prodnull/prompt-injection-repo-dataset 
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def fetch_and_split_data():
    # Create the pipeline storage architecture
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Downloading 'prodnull/prompt-injection-repo-dataset' from Hugging Face...")
    # Load the training data stream directly from the hub
    dataset = load_dataset("prodnull/prompt-injection-repo-dataset", split="train")
    
    # Convert data into a standard pandas workspace
    df = pd.DataFrame(dataset)
    
    # Map and normalize into the single strict pipeline schema
    df = df.rename(columns={"text": "text", "label": "label"})
    df["source"] = "prodnull/prompt-injection-repo-dataset"
    
    # Keep only the target schema columns
    df = df[["text", "label", "source"]]
    
    # Deduplicate exact text entries to protect against train/test data leakage
    initial_count = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"Removed {initial_count - len(df)} duplicate text rows.")
    
    print(f"Processing complete: {len(df)} distinct samples found.")
    print("Generating reproducible 80/20 stratified data split...")
    
    # Execute split ensuring equal class ratios across both targets using fixed state seed
    train_data, test_data = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df["label"], 
        random_state=42
    )
    
    # Save the splits safely to disk
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Success! Saved train partition to: {train_path} ({len(train_data)} rows)")
    print(f"Success! Saved test partition to: {test_path} ({len(test_data)} rows)")

if __name__ == "__main__":
    fetch_and_split_data()