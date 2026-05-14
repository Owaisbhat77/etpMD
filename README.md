# Building & Deploying an LLM from Scratch on Docker 🚀

---

## 🗺️ Big Picture — What We're Building

```
Train tiny LLM → Save model → Build API → Pack in Docker → Run anywhere
```

---

## 📋 What You Need First

- Python installed
- Docker installed
- Basic Python knowledge

---

## STEP 1: Build a Tiny LLM from Scratch 🧠

We'll build a **simple character-level language model** (like a baby GPT) using PyTorch.

### Install requirements first:
```bash
pip install torch flask
```

### Create file: `model.py`

```python
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
```

> **Simple Explanation:**
> - **Embedding** = converts letters to numbers the model understands
> - **LSTM** = remembers previous characters, learns patterns
> - **Linear layer** = predicts what comes next

---

## STEP 2: Train the Model 🏋️

### Create file: `train.py`

```python
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
```

### Run it:
```bash
python train.py
```

You'll see:
```
Training started...
Epoch 50/200  | Loss: 45.2
Epoch 100/200 | Loss: 30.1
Epoch 150/200 | Loss: 18.4
Epoch 200/200 | Loss: 10.2
✅ Model saved as llm_model.pth
```

---

## STEP 3: Create an API to Use the Model 🌐

### Create file: `app.py`

```python
from flask import Flask, request, jsonify
import torch
from model import TinyLLM

app = Flask(__name__)

# --- Load saved model ---
checkpoint = torch.load('llm_model.pth', map_location='cpu')

char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']
vocab_size  = checkpoint['vocab_size']
embed_size  = checkpoint['embed_size']
hidden_size = checkpoint['hidden_size']

model = TinyLLM(vocab_size, embed_size, hidden_size)
model.load_state_dict(checkpoint['model_state'])
model.eval()  # Set to prediction mode

print("✅ Model loaded successfully!")

# --- Text generation function ---
def generate_text(seed_text, num_chars=100):
    result = seed_text
    hidden = None

    for _ in range(num_chars):
        # Convert current text to numbers
        input_ids = [char_to_idx.get(c, 0) for c in result[-20:]]
        x = torch.tensor(input_ids).unsqueeze(0)

        # Predict next character
        with torch.no_grad():
            output, hidden = model(x, hidden)

        # Pick the most likely next character
        next_idx  = output[0, -1].argmax().item()
        next_char = idx_to_char[next_idx]
        result   += next_char

    return result

# --- API Endpoints ---
@app.route('/')
def home():
    return jsonify({
        "message": "🤖 Tiny LLM is running!",
        "usage": "POST /generate with {'prompt': 'your text'}"
    })

@app.route('/generate', methods=['POST'])
def generate():
    data        = request.get_json()
    prompt      = data.get('prompt', 'the')
    num_chars   = data.get('num_chars', 100)

    generated = generate_text(prompt, num_chars)

    return jsonify({
        "prompt":    prompt,
        "generated": generated
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Test it locally first:
```bash
python app.py
```

Then open a new terminal:
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "the quick", "num_chars": 50}'
```

---

## STEP 4: Create requirements.txt 📄

```bash
# Create this file manually
```

### `requirements.txt`:
```
torch==2.0.0
flask==2.3.0
```

---

## STEP 5: Write the Dockerfile 🐳

### Create file: `Dockerfile` (no extension!)

```dockerfile
# Step 1: Start with Python base image
FROM python:3.10-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy requirements first (for caching)
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project files
COPY . .

# Step 6: Tell Docker which port to open
EXPOSE 5000

# Step 7: Command to run when container starts
CMD ["python", "app.py"]
```

> **Line by line explanation:**
> - `FROM` = which base OS/image to start from
> - `WORKDIR` = like `cd` into a folder
> - `COPY` = copy files into container
> - `RUN` = run a command while building
> - `EXPOSE` = open this port
> - `CMD` = run this when container starts

---

## STEP 6: Your Folder Structure 📁

Make sure all files are like this:
```
my-llm/
│
├── model.py          ← Model architecture
├── train.py          ← Training script
├── app.py            ← Flask API
├── llm_model.pth     ← Saved model (after training)
├── requirements.txt  ← Dependencies
└── Dockerfile        ← Docker instructions
```

---

## STEP 7: Build the Docker Image 🔨

```bash
# Go into your project folder
cd my-llm

# Build the image (name it "tiny-llm")
docker build -t tiny-llm .
```

You'll see Docker executing each step:
```
Step 1/7: FROM python:3.10-slim ✅
Step 2/7: WORKDIR /app ✅
Step 3/7: COPY requirements.txt ✅
Step 4/7: RUN pip install... ✅
Step 5/7: COPY . . ✅
Step 6/7: EXPOSE 5000 ✅
Step 7/7: CMD python app.py ✅

Successfully built tiny-llm ✅
```

---

## STEP 8: Run the Docker Container 🚀

```bash
docker run -p 5000:5000 tiny-llm
```

> `-p 5000:5000` means:
> **Your computer's port 5000 → Container's port 5000**

You'll see:
```
✅ Model loaded successfully!
* Running on http://0.0.0.0:5000
```

---

## STEP 9: Test Your Deployed LLM 🎉

### Option A — Using curl:
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "machine", "num_chars": 80}'
```

### Option B — Using Python:
```python
import requests

response = requests.post(
    "http://localhost:5000/generate",
    json={"prompt": "the quick", "num_chars": 100}
)
print(response.json())
```

### You'll get back:
```json
{
  "prompt": "the quick",
  "generated": "the quick brown fox jumps over the lazy..."
}
```

---

## 🗺️ Full Flow Summary

```
1. model.py      → Define LLM architecture
      ↓
2. train.py      → Train on text data
      ↓
3. llm_model.pth → Saved trained model
      ↓
4. app.py        → Flask API wraps the model
      ↓
5. Dockerfile    → Package everything
      ↓
6. docker build  → Create image
      ↓
7. docker run    → Start container
      ↓
8. localhost:5000 → Your LLM is LIVE! 🎉
```

---

## 🎯 Viva Questions on This

| Question | Answer |
|---|---|
| What is a Dockerfile? | Instructions to build a Docker image |
| What does `docker build` do? | Creates an image from Dockerfile |
| What does `docker run` do? | Starts a container from an image |
| What is -p 5000:5000? | Maps host port to container port |
| Why Flask? | To expose model as a web API |
| What is LSTM? | Learns sequential patterns in text |
| What is model.pth? | PyTorch file storing trained weights |
| Why `WORKDIR /app`? | Sets working directory in container |

---

**You're building a real LLM + deploying it — that's impressive for a viva! 💪🔥**
