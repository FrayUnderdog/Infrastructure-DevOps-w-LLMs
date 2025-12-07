"""
data_collection_preprocessing.py

    Data collection + preprocessing pipeline for LLM pretraining.

    - Primary source:  OpenWebText parquet mirror `dylanebert/openwebtext`.
      (OpenWebText: ~8M documents, ~38GB text, far > 1GB overall.)
    - Fallback source: CC-News `vblagoje/cc_news`, a 1.12GB news dataset
      derived from Common Crawl, with 708,241 English news articles.
      Size of downloaded dataset files: 1.12 GB, number of rows: 708,241. [HF card]

    - In FULL mode, we conceptually load about 250,000 documents (~1.1–1.2GB raw text).
"""

from __future__ import annotations

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
from typing import List, Dict, Any, Optional


# ==============================
# 1. Load Data：OpenWebText & CC-News
# ==============================

def load_openwebtext_texts(max_examples: int, dev_mode: bool = True) -> List[str]:
    """
        Load a subset of the OpenWebText dataset from the parquet-based
        mirror `dylanebert/openwebtext`.
    """
    if dev_mode:
        print(f"[DEV MODE] Loading dylanebert/openwebtext with at most {max_examples} examples.")
    else:
        print(f"[FULL MODE] Loading dylanebert/openwebtext with at most {max_examples} examples.")

    split = f"train[:{max_examples}]"
    ds = load_dataset("dylanebert/openwebtext", split=split)

    texts: List[str] = []
    n = len(ds)
    for i in range(n):
        t = ds[i]["text"]
        if t is not None:
            texts.append(t)

    print(f"Loaded {len(texts)} raw documents from dylanebert/openwebtext.")
    return texts


def load_cc_news_texts(max_examples: int, dev_mode: bool = True) -> List[str]:
    """
        Load a subset of CC-News dataset `vblagoje/cc_news` as a fallback.
        HF card: ~708k rows, 1.12GB downloaded dataset files.
    """
    if dev_mode:
        print(f"[DEV MODE] Loading vblagoje/cc_news with at most {max_examples} examples.")
    else:
        print(f"[FULL MODE] Loading vblagoje/cc_news with at most {max_examples} examples.")

    split = f"train[:{max_examples}]"
    ds = load_dataset("vblagoje/cc_news", split=split)

    texts: List[str] = []
    n = len(ds)
    for i in range(n):
        t = ds[i]["text"]
        if t is not None:
            texts.append(t)

    print(f"Loaded {len(texts)} raw documents from vblagoje/cc_news.")
    return texts


def load_raw_texts(max_examples: int, dev_mode: bool = True) -> List[str]:
    """
        Try to load texts from OpenWebText first.
        If any exception happens during download/preparation,
        fall back to CC-News as an alternative source.
    """
    try:
        print("[INFO] Trying OpenWebText (dylanebert/openwebtext) first...")
        return load_openwebtext_texts(max_examples=max_examples, dev_mode=dev_mode)
    except Exception as e:
        print("[WARN] Failed to load OpenWebText due to an exception.")
        print(f"[WARN] Exception type: {type(e).__name__}, message: {e}")
        print("[INFO] Falling back to CC-News (vblagoje/cc_news) from Common Crawl...")
        return load_cc_news_texts(max_examples=max_examples, dev_mode=dev_mode)


# ==============================
# 2. Clean Data
# ==============================

def clean_text(text: str) -> Optional[str]:
    """
        Clean a single document:
        - Lowercase
        - Collapse whitespace
        - Filter out too short documents (num_words < 50)
    """
    # Lowercase
    text = text.lower()

    # Collapse multiple spaces / newlines into one space
    text = re.sub(r"\s+", " ", text).strip()

    # Filter too short documents
    num_words = len(text.split())
    if num_words < 50:
        return None

    return text


def clean_and_dedup(texts: List[str]) -> List[str]:
    """
        Clean a list of documents and remove exact duplicates.
    """
    cleaned: List[str] = []
    seen = set()

    for t in tqdm(texts, desc="Cleaning & deduplicating"):
        c = clean_text(t)
        if c is None:
            continue
        if c in seen:
            continue
        seen.add(c)
        cleaned.append(c)

    return cleaned


# ==============================
# 3. tokenization + chunking
# ==============================

def tokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    block_size: int = 512,
    min_chunk_len: int = 32,
    max_chunks: Optional[int] = None,
) -> List[List[int]]:
    """
        Tokenize cleaned texts and split into fixed-length chunks.
    """
    all_token_ids: List[List[int]] = []

    for t in tqdm(texts, desc="Tokenizing"):
        if max_chunks is not None and len(all_token_ids) >= max_chunks:
            break

        ids = tokenizer.encode(t, add_special_tokens=True)

        # Chunk into blocks
        for i in range(0, len(ids), block_size):
            if max_chunks is not None and len(all_token_ids) >= max_chunks:
                break

            chunk = ids[i: i + block_size]
            if len(chunk) < min_chunk_len:
                continue
            all_token_ids.append(chunk)

    return all_token_ids


# ==============================
# 4. define Dataset & collate_fn
# ==============================

class PretrainDataset(Dataset):
    """
        Simple Dataset wrapper around a list of token ID sequences.
    """

    def __init__(self, tokenized_chunks: List[List[int]]):
        self.data = tokenized_chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long)


def build_collate_fn(tokenizer: AutoTokenizer):
    """
        Build a collate_fn that:
        - Pads sequences in a batch to the same length using pad_token_id
        - Creates attention_mask (1 for real tokens, 0 for padding)
    """
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Find max length in this batch
        max_len = max(x.size(0) for x in batch)

        input_ids = []
        attention_masks = []

        for x in batch:
            pad_len = max_len - x.size(0)
            pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long)
            padded = torch.cat([x, pad_tensor], dim=0)

            mask = torch.cat(
                [torch.ones_like(x, dtype=torch.long),
                 torch.zeros(pad_len, dtype=torch.long)],
                dim=0,
            )

            input_ids.append(padded)
            attention_masks.append(mask)

        input_ids_t = torch.stack(input_ids, dim=0)         # [B, L]
        attention_masks_t = torch.stack(attention_masks, 0) # [B, L]

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_masks_t,
        }

    return collate_fn


# ==============================
# 5. Save sample batch
# ==============================

def save_sample_batches(
    loader: DataLoader,
    num_batches: int = 5,
    save_path: str = "sample_dataset.pt"
) -> str:
    """
        Take the first `num_batches` batches from DataLoader
        and save them as a list of dicts using torch.save.
    """
    samples: List[Dict[str, torch.Tensor]] = []
    it = iter(loader)

    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        samples.append(batch)

    torch.save(samples, save_path)
    return save_path


# ==============================
# 6. main()
# ==============================

def main(
    dev_mode: bool = True,
) -> None:
    """
        Main entry for the data pipeline.

        dev_mode=True:
            - Use a small subset (e.g., 5,000 docs) to debug the pipeline quickly.
        dev_mode=False:
            - Use a much larger subset (e.g., 250,000 docs),
              which corresponds to ~1.1–1.2 GB of raw text on average.
    """

    # ---------------------------------
    # 1) how many samples to take
    # ---------------------------------
    if dev_mode:
        max_examples = 5_000      # for test
        max_chunks = 10_000       # 
        print("[INFO] Running in DEV mode.")
    else:
        max_examples = 250_000    # 
        max_chunks = 100_000      # 
        print("[INFO] Running in FULL mode (approx >1GB raw text).")

    # ---------------------------------
    # 2) load raw text（for firstly OpenWebText，backup: CC-News）
    # ---------------------------------
    texts_raw = load_raw_texts(max_examples=max_examples, dev_mode=dev_mode)

    # ---------------------------------
    # 3) clean text + deduplicate
    # ---------------------------------
    texts_clean = clean_and_dedup(texts_raw)
    print(f"After cleaning & deduplication: {len(texts_clean)} documents remain.")

    # ---------------------------------
    # 4) prepare tokenizer
    # ---------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Using tokenizer: gpt2 (pad_token_id={tokenizer.pad_token_id})")

    # ---------------------------------
    # 5) tokenize + chunk
    # ---------------------------------
    tokenized_chunks = tokenize_texts(
        texts_clean,
        tokenizer,
        block_size=512,
        min_chunk_len=32,
        max_chunks=max_chunks,
    )
    print(f"Total tokenized chunks: {len(tokenized_chunks)}")

    # ---------------------------------
    # 6)  Dataset & DataLoader
    # ---------------------------------
    dataset = PretrainDataset(tokenized_chunks)
    collate_fn = build_collate_fn(tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print("DataLoader is ready.")

    # ---------------------------------
    # 7) save 5 batch 
    # ---------------------------------
    save_path = save_sample_batches(loader, num_batches=5, save_path="sample_dataset.pt")
    print(f"Saved sample batches to {save_path}")


if __name__ == "__main__":
    # for test, dev_mode=True with smaller data
    main(dev_mode=True)
    # for 1GB text data
    # main(dev_mode=False)
