# SLM -- Small Language Model

Training a GPT-style transformer from scratch on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) using a single RTX 5090 (32GB VRAM).

## Model

| | |
| --- | --- |
| Architecture | Decoder-only transformer (GPT-style) |
| Parameters | ~538M |
| Hidden dim | 1280 |
| Attention heads | 20 |
| Layers | 24 |
| FFN dim | 5120 |
| Context length | 2048 tokens |
| Precision | bf16 (via `torch.amp.autocast`) |
| Attention | PyTorch SDPA with flash backend (`is_causal=True`) |
| Position encoding | Learned absolute |
| Weight tying | Embedding and LM head share weights |
| Initialization | N(0, 0.02) with scaled residual projections |

## Training

- **Dataset:** FineWeb (streamed, since the actual dataset is something insane like 50TB)
- **Tokenizer:** Custom 50k-vocab BPE trained on ~2GB of FineWeb via the `tokenizers` library (UNK rate should end up around 0? at least it did for me)
- **Optimizer:** AdamW (fused, weight decay on 2D+ params only)
- **Schedule:** Cosine decay from 3e-4 to 3e-5 with 2000-step linear warmup
- **Effective batch size:** ~524k tokens (micro-batch 8 x grad accum 32 x 2048 seq len)
- **Gradient checkpointing:** Enabled
- **`torch.compile`:** Enabled
- **Tracking:** Weights & Biases

## File Structure

```
.
├── train.ipynb        # Single notebook: setup, tokenizer, data pipeline, model, training, eval
├── config.yaml        # All hyperparameters
├── requirements.txt   # Dependencies (pinned to PyTorch cu128 for RTX 5090)
├── tokenizer/         # Saved tokenizer artifacts (generated)
├── checkpoints/       # Model checkpoints (generated)
└── wandb/             # W&B run logs (generated)
```

## Requirements

- NVIDIA RTX 5090 (or any Blackwell GPU with CUDA 12.8)
- Python 3.10+
- PyTorch >= 2.7.0 with CUDA 12.8

```bash
pip install -r requirements.txt
```
