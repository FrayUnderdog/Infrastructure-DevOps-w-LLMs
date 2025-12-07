"""
mini_gpt_train_full.py
"""

import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


# =========================
# 1. Config
# =========================

DATA_PATH = "train_tokens_128.pt"          # 128-length token dataset
CHECKPOINT_PATH = "mini_gpt_full.pt"       # where to save the final model

NUM_EPOCHS = 2                             # moderate training time
BATCH_SIZE = 32                            # typical batch size
LEARNING_RATE = 1e-3                       # baseline LR

MAX_SEQS = 20000                           # only use first 20,000 sequences for saving time


# =========================
# 2. Dataset & collate_fn
# =========================

class TokenDataset(Dataset):
    """
    Wrap a list of token ID sequences.

    Each item is a 1D LongTensor: [seq_len]
    """
    def __init__(self, token_sequences):
        self.data = token_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long)


def make_collate_fn(pad_token_id: int):
    """
    Return a collate_fn that:

    - Pads each sequence in the batch to the same length with pad_token_id.
    - Builds input_ids and labels for next-token prediction:
        * input_ids = padded[:, :-1]
        * labels    = padded[:,  1:]
    - Builds attention_mask for input_ids (1 = real token, 0 = padding).
    """
    def collate_fn(batch):
        # batch: List[1D LongTensor]
        lengths = [x.size(0) for x in batch]
        max_len = max(lengths)

        padded = []
        for x in batch:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), pad_token_id, dtype=torch.long)
                x = torch.cat([x, pad_tensor], dim=0)
            padded.append(x)

        # [B, max_len]
        padded = torch.stack(padded, dim=0)

        # next-token prediction
        # input_ids: [B, max_len - 1]
        # labels:    [B, max_len - 1]
        input_ids = padded[:, :-1]
        labels = padded[:, 1:].clone()

        # attention mask for input_ids
        attention_mask = (input_ids != pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate_fn


# =========================
# 3. Mini-GPT Model
# =========================

class MiniGPT(nn.Module):
    """
    A small GPT-style Transformer for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Token & positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder (batch_first=True makes input [B, L, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1 for real token, 0 for padding)
        returns: logits [B, L, vocab_size]
        """
        B, L = input_ids.size()
        device = input_ids.device

        # positions: [0, 1, 2, ..., L-1]
        pos = torch.arange(L, device=device).unsqueeze(0)  # [1, L]

        # embeddings
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # src_key_padding_mask: True for PAD positions
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for pad
            src_key_padding_mask = (attention_mask == 0)  # [B, L], bool
        else:
            src_key_padding_mask = None

        # pass through Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # project to vocab
        logits = self.lm_head(x)  # [B, L, vocab]
        return logits


# =========================
# 4. Training function
# =========================

def train_mini_gpt_full():
    # 4.1 Load token sequences
    print(f"[TRAIN] Loading token sequences from: {DATA_PATH}")
    token_sequences = torch.load(DATA_PATH)  # List[List[int]]
    print(f"[TRAIN] Total sequences in file: {len(token_sequences)}")

    # Use only the first MAX_SEQS sequences to keep training time reasonable
    if len(token_sequences) > MAX_SEQS:
        token_sequences = token_sequences[:MAX_SEQS]
        print(f"[TRAIN] Using first {MAX_SEQS} sequences for training.")
    else:
        print(f"[TRAIN] Using all {len(token_sequences)} sequences for training.")

    dataset = TokenDataset(token_sequences)
    print(f"[TRAIN] Dataset size: {len(dataset)} sequences")

    # 4.2 Load tokenizer (for pad_token_id & vocab_size)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    print(f"[TRAIN] pad_token_id = {pad_token_id}, vocab_size = {vocab_size}")

    # 4.3 DataLoader
    collate_fn = make_collate_fn(pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"[TRAIN] Batch size: {BATCH_SIZE}")

    # 4.4 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # 4.5 Model
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        dropout=0.1,
        pad_token_id=pad_token_id,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TRAIN] Model parameters: {num_params}")

    # 4.6 Loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4.7 Training loop
    loss_history = []
    ppl_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        step = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)      # [B, L]
            labels = batch["labels"].to(device)            # [B, L]
            attention_mask = batch["attention_mask"].to(device)  # [B, L]

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask=attention_mask)  # [B, L, vocab]
            # reshape for CrossEntropyLoss: [B*L, vocab], [B*L]
            B, L, V = logits.size()
            loss = criterion(
                logits.view(B * L, V),
                labels.view(B * L),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                print(f"[TRAIN] Epoch {epoch} Step {step}: loss = {loss.item():.4f}")

        avg_loss = total_loss / step
        ppl = math.exp(avg_loss)

        loss_history.append(avg_loss)
        ppl_history.append(ppl)

        print(f"[EPOCH {epoch}] avg_loss = {avg_loss:.4f}, perplexity = {ppl:.2f}")

    # 4.8 Save checkpoint
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[TRAIN] Saved model checkpoint to: {CHECKPOINT_PATH}")

    # 4.9 Plot loss & perplexity
    epochs = list(range(1, NUM_EPOCHS + 1))

    # Loss curve
    plt.figure()
    plt.plot(epochs, loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss vs Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve_full.png")
    plt.close()
    print("[PLOT] Saved loss curve to loss_curve_full.png")

    # Perplexity curve
    plt.figure()
    plt.plot(epochs, ppl_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexity vs Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("perplexity_curve_full.png")
    plt.close()
    print("[PLOT] Saved perplexity curve to perplexity_curve_full.png")

    return loss_history, ppl_history


if __name__ == "__main__":
    train_mini_gpt_full()
