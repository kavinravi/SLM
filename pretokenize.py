"""Pre-tokenize FineWeb to a local numpy memmap for fast training."""
import os
import time
import yaml
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

with open('config.yaml') as f:
    config = yaml.safe_load(f)

data_cfg = config['data']
tok_path = os.path.join(config['tokenizer']['save_dir'], 'tokenizer.json')
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id('<eos>')
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer loaded: {vocab_size} tokens")

TARGET_TOKENS = 15_000_000_000  # 15B tokens (~28 GB on disk as uint16)
OUT_PATH = "data/train_tokens.npy"
CHUNK_SIZE = 10_000_000  # flush to disk every 10M tokens

os.makedirs("data", exist_ok=True)

print(f"Target: {TARGET_TOKENS/1e9:.1f}B tokens -> {OUT_PATH}")
print(f"Estimated disk: ~{TARGET_TOKENS * 2 / 1e9:.1f} GB")
print("Streaming and tokenizing FineWeb...")

ds = load_dataset(
    data_cfg['dataset'],
    name=data_cfg.get('name'),
    split=data_cfg['split'],
    streaming=True,
)

fp = np.memmap(OUT_PATH, dtype=np.uint16, mode='w+', shape=(TARGET_TOKENS,))

written = 0
buffer = []
t0 = time.time()
docs = 0

for doc in ds:
    text = doc[data_cfg['text_field']]
    if len(text) < data_cfg.get('min_doc_length', 100):
        continue

    ids = tokenizer.encode(text).ids + [eos_id]
    buffer.extend(ids)
    docs += 1

    while len(buffer) >= CHUNK_SIZE:
        chunk = buffer[:CHUNK_SIZE]
        buffer = buffer[CHUNK_SIZE:]
        remaining = TARGET_TOKENS - written
        if remaining <= 0:
            break
        n = min(len(chunk), remaining)
        fp[written:written + n] = np.array(chunk[:n], dtype=np.uint16)
        written += n

        elapsed = time.time() - t0
        tps = written / elapsed
        eta = (TARGET_TOKENS - written) / tps if tps > 0 else 0
        print(f"  {written/1e9:.2f}B / {TARGET_TOKENS/1e9:.1f}B tokens | "
              f"{tps/1e6:.2f}M tok/s | {docs:,} docs | "
              f"ETA {eta/60:.0f}min")

    if written >= TARGET_TOKENS:
        break

fp.flush()
elapsed = time.time() - t0
print(f"\nDone: {written/1e9:.2f}B tokens in {elapsed/60:.1f} min")
print(f"File: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e9:.1f} GB)")
print(f"Average speed: {written/elapsed/1e6:.2f}M tok/s")

meta = {'num_tokens': written, 'vocab_size': vocab_size, 'dtype': 'uint16'}
np.save("data/meta.npy", meta)
print(f"Metadata saved to data/meta.npy")
