#!/usr/bin/env python3
"""Upload kalshi filtered dataset to HuggingFace"""

import subprocess
from pathlib import Path
from datasets import Dataset
import pandas as pd

# Get script directory
script_dir = Path(__file__).parent
kalshi_dir = script_dir / "kalshi"

# First, regenerate data with updated script
print("Regenerating data with price_at_cutoff field...")
prepare_script = kalshi_dir / "prepare_futurex_training_data.py"
subprocess.run(["python3", str(prepare_script)], check=True, cwd=kalshi_dir)

# Load the parquet file
data_path = kalshi_dir / "data" / "futurex_training_data.parquet"
print(f"\nLoading data from {data_path}...")
df = pd.read_parquet(data_path)

print(f"Loaded {len(df)} examples")
print(f"✓ Has price_at_cutoff: {'price_at_cutoff' in df.columns}")
print(f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist()[:5])}...")

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Push to hub
repo_name = "SubconsciousDev/kalshi-filtered"
print(f"\nUploading to HuggingFace as '{repo_name}'...")

dataset.push_to_hub(
    repo_name,
    private=False,
    commit_message="FutureX-style Kalshi training data: 1,433 examples, volume>1000, price 20-80¢, includes price_at_cutoff"
)

print(f"\n✓ Successfully uploaded!")
