import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from autoencoder import ConvAutoencoder
from pathlib import Path
import matplotlib.pyplot as plt

# Adjustable parameter
batch_size = 64
test_ratio = 0.02

# Load and transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Configure paths
current_file = Path(__file__).resolve()
project_path = current_file.parents[2]
noisy_cells = Path(project_path, "Data", "Output_files", "HTB5-170122", "Dataset")

# Load dataset
full_dataset = datasets.ImageFolder(root=noisy_cells, transform=transform)
test_size = int(len(full_dataset) * test_ratio)
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = ConvAutoencoder()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("Starting training...")
# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# 🔍 Visualize test reconstructions
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)
    recon = model(images)

# Bring images to CPU
images = images.cpu()
recon = recon.cpu()

# Plot
n = min(8, len(images))
fig, axs = plt.subplots(2, n, figsize=(n*2, 4))
for i in range(n):
    axs[0, i].imshow(images[i].squeeze(), cmap='gray')
    axs[0, i].set_title("Original")
    axs[0, i].axis('off')
    axs[1, i].imshow(recon[i].squeeze(), cmap='gray')
    axs[1, i].set_title("Reconstructed")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
