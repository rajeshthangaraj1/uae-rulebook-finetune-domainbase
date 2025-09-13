import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Path to the JSONL file
jsonl_path = 'data/qa_merged_50.jsonl'

# Read the JSONL file into a DataFrame efficiently
# (Assume each line is a valid JSON object as per the example)
df = pd.read_json(jsonl_path, lines=True)

# Convert the DataFrame to a HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df)

# Push to HuggingFace Hub
hf_dataset.push_to_hub('rajeshthangaraj1/uae-banking-rulebook-qa', token=os.getenv('HF_TOKEN'))

print('Dataset pushed to HuggingFace Hub successfully.')
