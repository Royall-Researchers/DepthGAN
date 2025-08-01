import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --------------------------
# Device
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------
# Dataset for Grayscale Stereo Images
# --------------------------
class StereoDataset(Dataset):
    def __init__(self, left_folder, right_folder):
        self.left_paths = sorted(os.listdir(left_folder))
        self.right_paths = sorted(os.listdir(right_folder))
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = cv2.imread(os.path.join(self.left_folder, self.left_paths[idx]), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(os.path.join(self.right_folder, self.right_paths[idx]), cv2.IMREAD_GRAYSCALE)
        left_tensor = self.transform(left_img.astype(np.float32) / 255.0)
        right_tensor = self.transform(right_img.astype(np.float32) / 255.0)
        return left_tensor, right_tensor

# --------------------------
# Generator Network
# --------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, left, right):
        x = torch.cat([left, right], dim=1)
        return self.decoder(self.encoder(x))

# --------------------------
# Discriminator Network
# --------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Linear(256 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(self.fc(torch.flatten(x, 1)))

# --------------------------
# Load Data
# --------------------------
left_dir = '/content/drive/MyDrive/Colab Notebooks/image_0'
right_dir = '/content/drive/MyDrive/Colab Notebooks/image_1'
dataset = StereoDataset(left_dir, right_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# --------------------------
# Models, Loss, Optimizer
# --------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()

# --------------------------
# Training Loop
# --------------------------
epochs = 50
for epoch in range(epochs):
    for left, right in dataloader:
        left, right = left.to(device), right.to(device)
        real_disp = torch.rand_like(left).to(device)
        fake_disp = generator(left, right)

        # Train Discriminator
        optimizer_D.zero_grad()
        d_real = discriminator(real_disp)
        d_fake = discriminator(fake_disp.detach())
        d_loss = criterion_bce(d_real, torch.ones_like(d_real)) + \
                 criterion_bce(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        d_fake = discriminator(fake_disp)
        g_loss = criterion_bce(d_fake, torch.ones_like(d_fake)) + \
                 10 * criterion_l1(fake_disp, real_disp)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

# --------------------------
# Export to ONNX
# --------------------------
generator.eval()
discriminator.eval()
dummy_left = torch.randn(1, 1, 256, 256, device=device)
dummy_right = torch.randn(1, 1, 256, 256, device=device)
torch.onnx.export(generator, (dummy_left, dummy_right), "/content/drive/MyDrive/Colab Notebooks/depthgan_generator.onnx", opset_version=11)
torch.onnx.export(discriminator, torch.randn(1, 1, 256, 256, device=device), "/content/drive/MyDrive/Colab Notebooks/depthgan_discriminator.onnx", opset_version=11)
print("✅ ONNX export complete.")

# --------------------------
# Visualization
# --------------------------
def visualize_depth(left_path, right_path, model):
    model.eval()
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    left_tensor = transforms.ToTensor()(left).unsqueeze(0).to(device)
    right_tensor = transforms.ToTensor()(right).unsqueeze(0).to(device)

    with torch.no_grad():
        disparity = model(left_tensor, right_tensor).cpu().squeeze().numpy()
        depth = 1.0 / (disparity + 1e-6)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(disparity, cmap='magma')
    plt.title("Disparity Map")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap='inferno')
    plt.title("Estimated Depth")
    plt.axis('off')
    plt.show()

# Call like:
visualize_depth("/content/drive/MyDrive/Colab Notebooks/image_0/001999.png", "/content/drive/MyDrive/Colab Notebooks/image_1/001999.png", generator)
