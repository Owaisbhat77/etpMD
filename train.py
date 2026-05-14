import torch
import torch.nn as nn
from model import TinyLLM

# --- Our training text ---
text = """
the quick brown fox jumps over the lazy dog
machine learning is fun and powerful
deploying models is an important skill
"""

# --- Build vocabulary ---
chars = sorted(set(text))          # All unique characters
vocab_size = len(chars)

# Maps: character <--> number
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# --- Convert text to numbers ---
data = [char_to_idx[c] for c in text]

# --- Hyperparameters ---
embed_size  = 32
hidden_size = 64
seq_length  = 20
epochs      = 200
lr          = 0.01

# --- Create model ---
model = TinyLLM(vocab_size, embed_size, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn   = nn.CrossEntropyLoss()

# --- Training loop ---
print("Training started...")

for epoch in range(epochs):
    total_loss = 0

    for i in range(0, len(data) - seq_length - 1, seq_length):

        # Input: sequence of characters
        x = torch.tensor(data[i : i+seq_length]).unsqueeze(0)

        # Target: next character for each position
        y = torch.tensor(data[i+1 : i+seq_length+1])

        # Forward pass
        output, _ = model(x)
        loss = loss_fn(output.squeeze(0), y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# --- Save model + vocab ---
torch.save({
    'model_state': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size':  vocab_size,
    'embed_size':  embed_size,
    'hidden_size': hidden_size
}, 'llm_model.pth')

print("✅ Model saved as llm_model.pth")
