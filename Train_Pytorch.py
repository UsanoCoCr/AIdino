import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

jump_path = os.path.join('pics', 'jump')
none_path = os.path.join('pics', 'none')

jump_files = [os.path.join(jump_path, jump) for jump in os.listdir(jump_path)]
none_files = [os.path.join(none_path, none) for none in os.listdir(none_path)]

all_files = jump_files + none_files
random.shuffle(all_files)

X_pytorch_shuffled = [torch.tensor(cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (960, 540)), dtype=torch.float32) for file in all_files]
X_tensor_shuffled = torch.stack(X_pytorch_shuffled).unsqueeze(1).to(device)
labels_shuffled = [0 if 'jump' in file else 1 for file in all_files]
y_tensor_shuffled = torch.tensor(labels_shuffled, dtype=torch.long).to(device)

train_size = int(0.8 * len(X_tensor_shuffled))
val_size = len(X_tensor_shuffled) - train_size
train_dataset = TensorDataset(X_tensor_shuffled[:train_size], y_tensor_shuffled[:train_size])
val_dataset = TensorDataset(X_tensor_shuffled[train_size:], y_tensor_shuffled[train_size:])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class DinoCNN(nn.Module):
    def __init__(self):
        super(DinoCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(16*240*135, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 16*240*135)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


model = DinoCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print("predicted: ", predicted)
        print("labels: ", labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total}%")

# Save the model
torch.save(model.state_dict(), 'dino_model_pytorch.pth')