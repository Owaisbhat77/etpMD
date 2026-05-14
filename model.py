import torch
import torch.nn as nn

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TinyLLM, self).__init__()
        
        # Step 1: Convert each character to a vector
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Step 2: LSTM learns patterns in text
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Step 3: Predict next character
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
