"""Upload checkpoints to Hugging Face Hub."""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

REPO_ID = "kavin-ravi/slm-checkpoints"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

api = HfApi(token=os.environ["HF_TOKEN"])

create_repo(REPO_ID, repo_type="model", exist_ok=True)

files = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
print(f"Uploading {len(files)} files to {REPO_ID}:")
for f in files:
    print(f"  {f}")

for f in files:
    path = os.path.join(CHECKPOINT_DIR, f)
    size_gb = os.path.getsize(path) / 1e9
    print(f"\nUploading {f} ({size_gb:.1f} GB)...")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  Done: https://huggingface.co/{REPO_ID}/blob/main/{f}")

print(f"\nAll uploads complete: https://huggingface.co/{REPO_ID}")
