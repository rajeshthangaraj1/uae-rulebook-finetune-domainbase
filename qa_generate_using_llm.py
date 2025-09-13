import os
import time
import html
import re
import logging
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from datasets import Dataset, concatenate_datasets

# Setup logging
logging.basicConfig(
    filename="gemini_qa_generator.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def clean_context(text):
    text = text.replace("\\n", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\\/", "/")
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_run_agent(agent, prompt):
    try:
        return agent.run(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "rate limit" in error_msg or "permission" in error_msg:
            logging.critical(f"🚫 Quota or rate limit error: {e}")
        else:
            logging.error(f"❌ Error from LLM agent: {e}")
        raise e  # Raise to break and avoid retry


def generate_prompt(text_chunk):
    return f"""
You are a helpful assistant. Based on the following content, generate **3 question-answer pairs** that a reader might ask after reading it.

Text:
\"\"\"{text_chunk}\"\"\"

Return the output in this JSON format (no explanations or extra text):

[
  {{
    "question": "...",
    "answer": "..."
  }},
  ...
]
"""


def append_to_csv(data, file_path):
    import pandas as pd
    df = pd.DataFrame(data)
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df.to_csv(file_path, index=False)


def append_to_jsonl(data, file_path):
    import json
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    load_dotenv()

    agent = Agent(
        model=Gemini(id=os.getenv("GEMINI_MODEL")),
        markdown=True,
    )

    with open("scraped_section.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = [full_text[i:i + 500] for i in range(0, len(full_text), 500)]

    checkpoint_file = "checkpoint.txt"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0

    logging.info(f"⏳ Resuming from chunk index: {start_index}")

    BATCH_SIZE = 5
    failure_count = 0

    for batch_start in range(start_index, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        batch_qa_pairs = []
        batch_success = True

        for idx, chunk in enumerate(batch):
            chunk_idx = batch_start + idx
            logging.info(f"⏳ Processing chunk {chunk_idx + 1}/{len(chunks)}")
            prompt = generate_prompt(chunk)

            try:
                response = safe_run_agent(agent, prompt)
                content = response.content.strip().strip("`").replace("```json", "").replace("```", "").strip()

                if content.startswith("json\n"):
                    content = content[5:].strip()

                pairs = eval(content)

                for pair in pairs:
                    batch_qa_pairs.append({
                        "question": pair.get("question", ""),
                        "answer": pair.get("answer", ""),
                        "context": clean_context(chunk)
                    })

                time.sleep(60)

            except Exception as e:
                logging.error(f"❌ Error in chunk {chunk_idx + 1}: {e}")
                batch_success = False
                break  # No retry: break immediately

        if batch_success:
            # ✅ Save checkpoint
            with open(checkpoint_file, "w") as f:
                f.write(str(batch_start + BATCH_SIZE))
            logging.info(f"✅ Batch {batch_start // BATCH_SIZE + 1} complete. Checkpoint saved.")

            if batch_qa_pairs:
                # ✅ Convert to dataset
                batch_dataset = Dataset.from_dict({
                    "question": [item["question"] for item in batch_qa_pairs],
                    "answer": [item["answer"] for item in batch_qa_pairs],
                    "context": [item["context"] for item in batch_qa_pairs]
                })

                # ✅ Append to disk
                batch_dataset.save_to_disk("gemini_qa_dataset")
                append_to_csv(batch_qa_pairs, "qa_pairs.csv")
                append_to_jsonl(batch_qa_pairs, "qa_pairs.jsonl")

                # ✅ Push to HF
                batch_dataset.push_to_hub("rajeshthangaraj1/uae-banking-rulebook-qa", token=os.getenv("HF_TOKEN"))

                logging.info(f"📤 Hugging Face push complete. Added: {len(batch_qa_pairs)}")
            else:
                logging.warning("⚠️ No QA pairs generated in this batch.")
        else:
            logging.warning("⚠️ Skipping checkpoint due to batch failure.")
            break  # Don't continue if any batch failed

        time.sleep(1800)


if __name__ == "__main__":
    main()
