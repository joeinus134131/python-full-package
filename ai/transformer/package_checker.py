import torch
import matplotlib
import seaborn
import numpy as np

print("=== Cek Versi Package ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {seaborn.__version__}")
print(f"NumPy: {np.__version__}")

# Test basic functionality
print("\n=== Test Basic Operations ===")
x = torch.tensor([1, 2, 3])
print(f"Tensor test: {x}")
print("✅ Semua package berfungsi dengan baik!")