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
