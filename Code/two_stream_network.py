import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import TwoStreamStatic

IMAGE_SIZE = 64        # Resize images
FEATURE_DIM = 256      # CNN output features

# --------------------------
# Dataset for static images
# --------------------------
class StaticImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# --------------------------
# Transforms and DataLoaders
# --------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

train_dataset = StaticImageDataset('train_data/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = StaticImageDataset('test_data/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = len(train_dataset.class_to_idx)
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, Classes: {num_classes}")


# --------------------------
# Model, Loss, Optimizer
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoStreamStatic(feature_dim=FEATURE_DIM, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --------------------------
# Training and Testing Loops
# --------------------------
def train_model(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return running_loss / len(dataloader), acc

def test_model(model, criterion, dataloader, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return running_loss / len(dataloader), acc


# --------------------------
# Training Loop
# --------------------------
train_loss_curve = []
test_loss_curve = []
train_acc_curve = []
test_acc_curve = []

num_epochs = 50
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, criterion, optimizer, train_loader, device)
    test_loss, test_acc = test_model(model, criterion, test_loader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    train_loss_curve.append(train_loss)
    test_loss_curve.append(test_loss)
    train_acc_curve.append(train_acc)
    test_acc_curve.append(test_acc)

torch.save(model.state_dict(), 'pre trained weigths/two_stream_static_weights.pth')

import matplotlib.pyplot as plt

# Data
epochs = list(range(1, num_epochs+1))

# Plot Loss
plt.figure()
plt.plot(epochs, train_loss_curve, label='Train Loss')
plt.plot(epochs, test_loss_curve, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epochs, train_acc_curve, label='Train Accuracy')
plt.plot(epochs, test_acc_curve, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train vs Test Accuracy')
plt.legend()
plt.show()