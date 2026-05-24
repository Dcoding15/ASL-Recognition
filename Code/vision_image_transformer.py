import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from models import ViT

MAX_FRAMES = 10  # Fixed number of frames per sequence

class SequenceImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Only include valid directories as classes
        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            # Only include valid sequence folders
            if any(img.lower().endswith(('.jpg', '.jpeg', '.png')) for img in os.listdir(cls_dir)):
                self.samples.append((cls_dir, self.class_to_idx[cls]))
            else:
                for seq in os.listdir(cls_dir):
                    seq_path = os.path.join(cls_dir, seq)
                    if os.path.isdir(seq_path):
                        self.samples.append((seq_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq_path, label = self.samples[index]
        
        # Load images (sorted)
        images = sorted([
            img for img in os.listdir(seq_path)
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        frames = []
        for img_name in images:
            img_path = os.path.join(seq_path, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Ensure fixed sequence length
        if len(frames) > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]
        elif len(frames) < MAX_FRAMES:
            pad_frames = [frames[-1].clone() for _ in range(MAX_FRAMES - len(frames))]
            frames.extend(pad_frames)

        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        return frames, torch.tensor(label, dtype=torch.long)


# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Train and Test Data Loaders
train_dataset = SequenceImageDataset('train_data/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = SequenceImageDataset('test_data/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


def train_model(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for frames, labels in dataloader:
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
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
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return running_loss / len(dataloader), acc

device = torch.device("cpu")
num_classes = len(train_dataset.class_to_idx)

model = ViT(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

torch.save(model.state_dict(), 'pre trained weigths/vit_weights.pth')

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