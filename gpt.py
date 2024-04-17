import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


# hyperparameters
max_iters = 20000
lr = 1e-3
eval_iters = 500
batch_size = 32
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_dim = 32
dropout = 0.4

# Read the input corpus
with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("Length of text: ", len(text))

# Create characters as vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary: ", ''.join(chars))
print("Vocabulary size: ", vocab_size)

# Encoder and decoder function for idx to char and back
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert text to torch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split data into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Get single batch of data for training
def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out['train'], out['val']

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # Attention scores
        weights = q @ k.transpose(-2, -1) * C**-0.5 # # (B, T, C) @ # (B, C, T) --> # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        # weighted aggregation of values
        v = self.value(x) # (B, T, C)
        context = weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return context
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.ma = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.ma(self.ln1(x)) # Communication layer
        x = x + self.ffn(self.ln2(x)) # Computation layer
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            Block(embed_dim, n_head=4),
            Block(embed_dim, n_head=4),
            Block(embed_dim, n_head=4),
            nn.LayerNorm(embed_dim),
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embed = self.token_embedding_table(idx) # (B, T, embed_dim)
        pos_embed = self.positional_embedding_table(torch.arange(T, device=device)) # (T, embed_dim)
        x = token_embed + pos_embed # (B, T, embed_dim)
        x = self.blocks(x) # (B, T, embed_dim)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    

# Create the model and optimizer
model = LanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

# Train the model
for step in range(max_iters):
    xb, yb = get_batch(split='train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if ((step + 1) % 1000 == 0):
        train_loss, val_loss = estimate_loss()
        print(f'Step {step + 1}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}')
        
print(f'Final training loss:  {loss.item():.4f}')

# generate text from the model
print("Sample generation after training: ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 500)[0].tolist()))
