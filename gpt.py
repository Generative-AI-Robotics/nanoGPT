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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding_table = nn.Embedding(block_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embed = self.token_embedding_table(idx) # (B, T, embed_dim)
        pos_embed = self.positional_embedding_table(torch.arange(T, device=device)) # (T, embed_dim)
        x = token_embed + pos_embed # (B, T, embed_dim)
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    

# Create the model and optimizer
model = BigramLanguageModel()
m = model.to(device)
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
        
print("Final training loss: ", loss.item())

# generate text from the model
print("Sample generation after training: ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 500)[0].tolist()))
