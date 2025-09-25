import torch
import torch.nn as nn

print("🔥 Testing Transformer Components...")

# Test basic tensor operations
x = torch.randn(2, 5, 64)  # [batch, sequence, features]
print(f"Tensor shape: {x.shape}")

# Test simple linear layer
linear = nn.Linear(64, 32)
output = linear(x)
print(f"After linear layer: {output.shape}")

print("✅ Semua bekerja dengan baik! Anda siap untuk eksperimen Transformer.")