import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass
from new_prepare import (
    get_shakespeare_data,
    get_shakespeare_dataloaders,
    ShakespeareTokenizer,
    ShakespeareDataset,
)


@dataclass
class Config:
    n_embd: int
    n_head: int
    n_layer: int = 12
    block_size: int = 8
    vocab_size: int = 65
    bias: bool = False
    dropout: float = 0.1
    lr: float = 0.001


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.model(x)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(
            normalized_shape=config.n_embd, elementwise_affine=config.bias
        )
        self.ln2 = nn.LayerNorm(
            normalized_shape=config.n_embd, elementwise_affine=config.bias
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )

        self.mlp = MLP(config)

    def forward(self, x):
        # pre-norm architecture
        x_norm = self.ln1(x)
        # mha with residual
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        causal_mask = causal_mask.to(x.device)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=causal_mask,
            is_causal=True,
        )  # Causal Self Attention when using the same 3 values
        # attn_out [batch_size, seq_length, embed_dim]
        # attn_weights [batch_size, num_heads, seq_length]
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(config.vocab_size, config.n_embd),
                position_embedding=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(
                    normalized_shape=config.n_embd, elementwise_affine=config.bias
                ),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.token_embedding.weight = self.lm_head.weight

        # init weights?

    def forward(self, idx: torch.Tensor):
        device = idx.device
        b, t = idx.shape

        tok_emb = self.transformer.token_embedding(idx)  # (b, t, n_embed)

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.position_embedding(pos)  # (t, n_embed)
        x = self.transformer.drop(tok_emb + pos_emb)  # convert tokens to embeddings
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)  # final layer norm
        logits = self.lm_head(x)  # convert back to vocab size
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temp

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    device: str,
    tokenizer: ShakespeareTokenizer,
    gradient_accumulation_steps: int = 8,
):
    model.train()
    best_val_loss = 0
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for i, (data, targets) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            targets = targets.to(device)

            if device == "cuda":
                with torch.amp.autocast(device):
                    logits = model(data)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]), targets.view(-1)
                    )
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(data)
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]), targets.view(-1)
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                if device == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 2000 == 0:
                print("Saving checkpoint...")
                torch.save(model.state_dict(), "checkpoint.pth")
                print(
                    f"Batch {i + 1}: Loss: {loss.item() * gradient_accumulation_steps:.4f}"
                )
                print("\n")
                ids = model.generate(
                    torch.tensor([tokenizer.encode("\n")], device=device),
                    max_new_tokens=100,
                    temp=0.8,
                )
                print(f"{tokenizer.decode(ids.tolist())}\n")
                val_loss = evaluate(model, val_dataloader, device)
                print(f"{val_loss=}\n")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("Saving best model...")
                    torch.save(model.state_dict(), "best_model.pth")
                scheduler.step(val_loss)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    val_loss = 0
    for data, targets in tqdm(loader):
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        val_loss += loss.item()

    model.train()
    return val_loss / len(loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_shakespeare_data()
    tokenizer = ShakespeareTokenizer(data=data)
    dataset = ShakespeareDataset(data=data, tokenizer=tokenizer, block_size=256)
    train_loader, val_loader = get_shakespeare_dataloaders(
        dataset=dataset, tokenizer=tokenizer, batch_size=64
    )

    # config = Config(
    #     n_embd=512,
    #     n_head=8,
    #     block_size=1024,
    #     bias=False,
    #     dropout=0.1,
    # )
    config = Config(
        n_embd=384,
        n_head=6,
        n_layer=6,
        block_size=256,
        vocab_size=65,
        dropout=0.2,
        lr=0.001,
    )

    model = GPT(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=100,
        threshold=1e-6,
    )
    train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=10,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
