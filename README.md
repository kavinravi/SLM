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
├── inference.ipynb    # Notebook for running inference w/ the most recent checkpoint (enables testing during training)
├── milestones.md      # Markdown file tracking how the model behaved for me at various step milestones - not important, just fun to see at times
├── config.yaml        # All hyperparameters which you can adjust based on your hardware/time constraints
├── requirements.txt   # Dependencies (pinned to PyTorch cu128 for RTX 5090)
├── tokenizer/         # Saved tokenizer artifacts (generated, in gitignore)
├── checkpoints/       # Model checkpoints (generated, in gitignore)
└── wandb/             # W&B run logs (generated, in gitignore)
```

## Requirements

- NVIDIA RTX 5090 (or any Blackwell GPU with CUDA 12.8)
- Python 3.10+ (I used 3.12.3)
- PyTorch >= 2.7.0 with CUDA 12.8
- WandB is very helpful for tracking model training over long periods of time, so you'll need to make an account and obtain an API key. This is free and takes less than 5 minutes, but is still notable. The benefits are mainly that you're able to track training remotely from anywhere on any device. The notebook cell under "Training" will prompt you for this if you haven't done it already, so no worries about doing anything in advance. 

```bash
pip install -r requirements.txt
```

## Customization
- Obviously config.yaml is customizable (max_steps, logging granularity, batch size, etc.)
- Checkpoint used in inference.ipynb is changeable to see how the model gains coherence across steps over time

## Known Issues
- Logging updates pile up causing OOM crashes on my IDE (current sol'n is to decrease logging granularity by a factor of 10x, but this isn't a fix)
- Even the tiniest of power or internet outages will interrupt training, the former because it shuts off power to the computer, the latter because it interrupts the data streaming of the dataset which stops training
    - Checkpointing somewhat mitigates the severity of these issues since I start from the most recent checkpoint, but this does require manual restarting - so as of now, no "start training and go on vacation" sadly

## Work for the Future
- Retrain the model using second order optimizers if possible (e.g. Gauss-Newton, SOAP, SHAMPOO, etc.) if I can find ones compatible with the particular pytorch version I need for CUDA 12.8
- Add an argument in train.ipynb that automatically clears the output in training cells every once in awhile so I don't have to worry about training crashes
- Add some form of QK-norm, since I think that was the problem that was causing gradient & loss explosions and leading to having to restart training 6 times before 20k steps (:skull:)