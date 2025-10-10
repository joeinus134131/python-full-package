import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

transform = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST(
    root="./data",
    train=True, 
    transform=transform, 
    download=True
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        # ENCODER: 784 → 128 → 64 → 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),    # 784 pixels input (28x28)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 32),      # Bottleneck - compressed representation
            nn.ReLU()
        )
        
        # DECODER: 32 → 64 → 128 → 784  
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # Output antara 0-1 (pixel values)
            # nn.Tanh()
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Flatten image: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(-1, 784)
        
        # Encode → Decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleAutoencoder().to(device)
criterion = nn.MSELoss()  # Mean Squared Error

# model type of optimizernya
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_autoencoder(model, train_loader, epochs=7):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data.view(-1, 784))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

# RUN TRAINING!
train_autoencoder(model, train_loader, epochs=7)

def visualize_reconstruction(model, test_loader):
    model.eval()
    with torch.no_grad():
        # Get one batch
        data_iter = iter(test_loader)
        images, _ = next(data_iter)
        images = images.to(device)
        
        # Get reconstructions
        outputs = model(images)
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(2, 8, figsize=(12, 4))
        for i in range(8):
            # Original
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed  
            axes[1, i].imshow(outputs[i].cpu().view(28, 28), cmap='Oranges')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Load test data
test_data = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)

# Visualize!
visualize_reconstruction(model, test_loader)

# Lihat compressed representation
def see_latent_space(model, test_loader):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        
        # Get encoded representations
        encoded = model.encoder(images.view(-1, 784))
        
        print("Original image shape:", images[0].shape)      # [1, 28, 28]
        print("Encoded shape:", encoded[0].shape)           # [32] 
        print("Compression ratio:", 784/32)                 # 24.5x compression!
        
        # Show some encoded values
        print("First 10 values in latent space:", encoded[0][:10])

see_latent_space(model, test_loader)# Lihat compressed representation