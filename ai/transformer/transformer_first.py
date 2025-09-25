import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Set device automatically
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.scale = math.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Test dengan device yang tepat
def test_attention_mac():
    d_k = 64
    batch_size, seq_len = 2, 4
    
    # Pindahkan tensors ke device yang tepat
    Q = torch.randn(batch_size, seq_len, d_k).to(device)
    K = torch.randn(batch_size, seq_len, d_k).to(device)
    V = torch.randn(batch_size, seq_len, d_k).to(device)
    
    attention = ScaledDotProductAttention(d_k).to(device)
    output, weights = attention(Q, K, V)
    
    print(f"Running on: {output.device}")
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    
    return output.cpu(), weights.cpu()  # Pindah ke CPU untuk visualisasi

# Eksperimen sederhana yang optimized untuk Mac
def mac_friendly_experiment():
    """Eksperimen yang ringan untuk Mac"""
    
    vocab_size = 20  # Lebih kecil untuk menghemat memory
    d_model = 128    # Dimensionality lebih kecil
    seq_len = 6      # Sequence length lebih pendek
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, d_ff):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, num_heads, d_ff),
                num_layers=2
            )
            self.output_layer = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            x = self.output_layer(x)
            return x
    
    model = SimpleTransformer(vocab_size, d_model, num_heads=4, d_ff=256).to(device)
    
    # Data kecil
    inputs = torch.randint(0, vocab_size, (16, seq_len)).to(device)  # Batch kecil
    targets = inputs.clone()
    
    # Forward pass test
    with torch.no_grad():
        outputs = model(inputs)
        print(f"Model test - Input: {inputs.shape}, Output: {outputs.shape}")
        print("✅ Model berjalan di Mac dengan baik!")
    
    return model

# Jalankan testing
if __name__ == "__main__":
    print("🔧 Testing Transformer on Mac...")
    
    # Test 1: Basic attention
    output, weights = test_attention_mac()
    
    # Test 2: Simple model
    model = mac_friendly_experiment()
    
    # Test 3: Visualisasi (akan jalan di CPU)
    print("\n📊 Testing visualization...")
    sentence = ["I", "love", "transformer", "models"]
    attention_weights = torch.randn(1, 4, len(sentence), len(sentence))  # Mock data
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights[0, 0].detach().numpy(), 
                xticklabels=sentence, yticklabels=sentence,
                cmap='viridis', annot=True, fmt='.2f')
    plt.title('Attention Weights Visualization')
    plt.show()
    
    print("🎉 Semua test berhasil! Transformer kompatibel dengan Mac.")