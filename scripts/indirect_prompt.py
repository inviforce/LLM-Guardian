import pandas as pd
from datasets import load_dataset

# Load the raw dataset
print("Downloading BIPIA dataset...")
dataset = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT")

# Extract all splits exactly as they are
frames = []
for split in dataset.keys():
    df = dataset[split].to_pandas()
    frames.append(df)

# Combine and save to CSV
raw_df = pd.concat(frames, ignore_index=True)
output_file = "bipia_raw.csv"
raw_df.to_csv(output_file, index=False)

print(f"Done. Saved {len(raw_df)} rows to {output_file}.")