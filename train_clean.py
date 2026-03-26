import os
import math
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

# ── Setup ──────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
device = torch.device('cuda')

print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}')

with open('config.yaml') as f:
    config = yaml.safe_load(f)

config['training']['gradient_checkpointing'] = False
print(f"gradient_checkpointing: {config['training']['gradient_checkpointing']}")

torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tok_cfg = config['tokenizer']
data_cfg = config['data']
os.makedirs(tok_cfg['save_dir'], exist_ok=True)

tok_path = os.path.join(tok_cfg['save_dir'], 'tokenizer.json')

if os.path.exists(tok_path):
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"Loaded tokenizer: {tokenizer.get_vocab_size()} tokens")
else:
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer_obj = trainers.BpeTrainer(
        vocab_size=tok_cfg['vocab_size'],
        min_frequency=tok_cfg['min_frequency'],
        special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
    )

    sample_bytes = tok_cfg['sample_size_gb'] * 1024 ** 3

    def text_iter():
        ds = load_dataset(data_cfg['dataset'], name=data_cfg.get('name'), split=data_cfg['split'], streaming=True)
        total = 0
        for doc in ds:
            text = doc[data_cfg['text_field']]
            if len(text) < data_cfg['min_doc_length']:
                continue
            yield text
            total += len(text.encode('utf-8'))
            if total >= sample_bytes:
                break

    print(f"Training tokenizer on ~{tok_cfg['sample_size_gb']}GB of FineWeb...")
    tokenizer.train_from_iterator(text_iter(), trainer=trainer_obj)
    tokenizer.save(tok_path)
    print(f"Saved tokenizer: {tokenizer.get_vocab_size()} tokens")

print("Tokenizer ready.")

# ── Data Pipeline ──────────────────────────────────────────────────────────────
class PackedDataset(IterableDataset):
    def __init__(self, tokenizer, config, seed=42):
        self.tokenizer = tokenizer
        self.seq_len = config['model']['context_length']
        self.data_cfg = config['data']
        self.eos_id = tokenizer.token_to_id('<eos>')
        self.seed = seed

    def __iter__(self):
        epoch = 0
        while True:
            ds = load_dataset(
                self.data_cfg['dataset'],
                name=self.data_cfg.get('name'),
                split=self.data_cfg['split'],
                streaming=True,
            )
            ds = ds.shuffle(seed=self.seed + epoch, buffer_size=10_000)
            epoch += 1
            buffer = []
            for doc in ds:
                text = doc[self.data_cfg['text_field']]
                if len(text) < self.data_cfg['min_doc_length']:
                    continue
                buffer.extend(self.tokenizer.encode(text).ids + [self.eos_id])
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len + 1:]
                    yield (
                        torch.tensor(chunk[:-1], dtype=torch.long),
                        torch.tensor(chunk[1:], dtype=torch.long),
                    )

# ── Model ──────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, use_checkpoint):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, context_length, dropout, use_checkpoint):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, use_checkpoint)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / (2 * n_layers) ** 0.5)
            nn.init.normal_(block.ffn.down.weight, std=0.02 / (2 * n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

# ── Model init ─────────────────────────────────────────────────────────────────
print("Creating model...")
mcfg = config['model']
tcfg = config['training']
vocab_size = tokenizer.get_vocab_size()

raw_model = GPT(
    vocab_size=vocab_size,
    d_model=mcfg['d_model'],
    n_heads=mcfg['n_heads'],
    n_layers=mcfg['n_layers'],
    d_ff=mcfg['d_ff'],
    context_length=mcfg['context_length'],
    dropout=mcfg['dropout'],
    use_checkpoint=tcfg['gradient_checkpointing'],
).to(device)

n_params = sum(p.numel() for p in raw_model.parameters())
print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')
print(f'VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

# ── Optimizer + compile ────────────────────────────────────────────────────────
print("Setting up optimizer...")
decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
nodecay_params = [p for p in raw_model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {'params': decay_params, 'weight_decay': tcfg['weight_decay']},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ],
    lr=tcfg['peak_lr'],
    betas=(tcfg['beta1'], tcfg['beta2']),
    fused=True,
)

if tcfg['compile']:
    print("Compiling model (this takes a few minutes on first run)...")
model = torch.compile(raw_model) if tcfg['compile'] else raw_model


def get_lr(step):
    if step < tcfg['warmup_steps']:
        return tcfg['peak_lr'] * step / tcfg['warmup_steps']
    ratio = min((step - tcfg['warmup_steps']) / (tcfg['max_steps'] - tcfg['warmup_steps']), 1.0)
    return tcfg['min_lr'] + 0.5 * (1 + math.cos(math.pi * ratio)) * (tcfg['peak_lr'] - tcfg['min_lr'])


# ── Resume / state ─────────────────────────────────────────────────────────────
start_step, tokens_seen, best_val_loss = 0, 0, float('inf')
train_log = {'step': [], 'loss': [], 'lr': []}
val_log = {'step': [], 'loss': []}
os.makedirs(tcfg['checkpoint_dir'], exist_ok=True)

if tcfg.get('resume_from'):
    print(f"Resuming from {tcfg['resume_from']}...")
    ckpt = torch.load(tcfg['resume_from'], weights_only=False, map_location=device)
    raw_model.load_state_dict(ckpt['model'])
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt.get('step', 0) + 1
    tokens_seen = ckpt.get('tokens_seen', 0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    train_log = ckpt.get('train_log', train_log)
    val_log = ckpt.get('val_log', val_log)
    print(f'Resumed from step {start_step} (optimizer: {"loaded" if "optimizer" in ckpt else "reset"})')

# ── Data loaders ───────────────────────────────────────────────────────────────
print("Setting up data loaders...")
eval_batches = None

print("Creating train DataLoader...")
train_ds = PackedDataset(tokenizer, config, seed=config['seed'] + start_step)
train_loader = DataLoader(train_ds, batch_size=tcfg['micro_batch_size'], drop_last=True)
data_iter = iter(train_loader)

# ── W&B ────────────────────────────────────────────────────────────────────────
if config['wandb']['enabled']:
    import wandb
    wandb.init(project=config['wandb']['project'], config=config, resume='allow')
    print("W&B initialized.")


@torch.no_grad()
def evaluate():
    global eval_batches
    if eval_batches is None:
        print('Prefetching eval batches (one-time)...')
        eval_ds = PackedDataset(tokenizer, config, seed=config['seed'])
        eval_loader = DataLoader(eval_ds, batch_size=tcfg['micro_batch_size'], drop_last=True)
        eval_iter = iter(eval_loader)
        eval_batches = [next(eval_iter) for _ in range(tcfg['eval_steps'])]
        del eval_ds, eval_loader
        print(f'Eval batches ready: {len(eval_batches)}')
    model.eval()
    losses = []
    for x, y in eval_batches:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x.to(device)).view(-1, vocab_size), y.to(device).view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ── Training loop ──────────────────────────────────────────────────────────────
tokens_per_step = tcfg['micro_batch_size'] * tcfg['gradient_accumulation_steps'] * mcfg['context_length']
loss_ema = None
spike_factor = 1.5
max_consecutive_skips = 50
consecutive_skips = 0
model.train()

print(f"Starting training from step {start_step} to {tcfg['max_steps']}...")
print(f"Tokens per step: {tokens_per_step:,}")

for step in range(start_step, tcfg['max_steps']):
    t0 = time.time()
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad()
    accum_loss = 0.0

    for _ in range(tcfg['gradient_accumulation_steps']):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
            (loss / tcfg['gradient_accumulation_steps']).backward()
        accum_loss += loss.item() / tcfg['gradient_accumulation_steps']

    if loss_ema is None:
        loss_ema = accum_loss
    if accum_loss > spike_factor * loss_ema:
        consecutive_skips += 1
        if consecutive_skips <= max_consecutive_skips:
            optimizer.zero_grad()
            print(f'step {step} | SPIKE SKIPPED ({consecutive_skips}/{max_consecutive_skips}) | loss {accum_loss:.4f} (ema {loss_ema:.4f})')
            if config['wandb']['enabled']:
                wandb.log({'train/loss': accum_loss, 'train/spike_skipped': 1}, step=step)
            continue
        print(f'step {step} | EMA RESET | skip limit reached, resetting ema {loss_ema:.4f} -> {accum_loss:.4f}')
        loss_ema = accum_loss
        consecutive_skips = 0
    else:
        consecutive_skips = 0
    loss_ema = 0.99 * loss_ema + 0.01 * accum_loss

    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tcfg['gradient_clip'])
    optimizer.step()
    tokens_seen += tokens_per_step
    dt = time.time() - t0

    if step % tcfg['log_interval'] == 0:
        tps = tokens_per_step / dt
        print(f'step {step} | loss {accum_loss:.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f} | {tps:.0f} tok/s | {dt*1000:.0f}ms')
        train_log['step'].append(step)
        train_log['loss'].append(accum_loss)
        train_log['lr'].append(lr)
        if config['wandb']['enabled']:
            wandb.log({'train/loss': accum_loss, 'train/lr': lr, 'train/grad_norm': grad_norm.item(), 'train/tok_per_sec': tps, 'tokens_seen': tokens_seen}, step=step)

    if step > 0 and step % tcfg['eval_interval'] == 0:
        vl = evaluate()
        ppl = math.exp(vl)
        print(f'step {step} | val_loss {vl:.4f} | ppl {ppl:.2f}')
        val_log['step'].append(step)
        val_log['loss'].append(vl)
        if config['wandb']['enabled']:
            wandb.log({'val/loss': vl, 'val/ppl': ppl}, step=step)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(
                {'model': raw_model.state_dict(), 'step': step, 'val_loss': vl},
                os.path.join(tcfg['checkpoint_dir'], 'best.pt'),
            )

    if step > 0 and step % tcfg['save_interval'] == 0:
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'tokens_seen': tokens_seen,
            'best_val_loss': best_val_loss,
            'train_log': train_log,
            'val_log': val_log,
        }, os.path.join(tcfg['checkpoint_dir'], f'step_{step}.pt'))

print(f'Done. Tokens: {tokens_seen:,}')
if config['wandb']['enabled']:
    wandb.finish()
