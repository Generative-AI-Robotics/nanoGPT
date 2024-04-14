import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


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
def get_batch(split='train', block_size=8, batch_size=4):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C) (4, 8, 65)
        
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
m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32

# Train the model
for steps in range(20000):
    xb, yb = get_batch(split='train', batch_size=batch_size)
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if (steps % 1000 == 0):
        print("Steps: ", steps, "Loss: ", loss.item())
    
print("Final training loss: ", loss.item())

# Test inference of the trained model
print("Sample generation after training: ")
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), 500)[0].tolist()))
