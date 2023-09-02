import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F

# 定义模型
class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)  # 960x540 -> 480x270
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) # 480x270 -> 240x135
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2) # 240x135 -> 120x68
        self.fc1 = nn.Linear(120 * 68 * 128, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# 定义数据集
class DinoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels = ['jump', 'none']
        self.data = []
        for label in self.labels:
            for img_name in os.listdir(os.path.join(root_dir, label)):
                self.data.append((os.path.join(root_dir, label, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0  # Ensure the channel dimension is the first dimension
        label_idx = self.labels.index(label)
        return img, label_idx


# 训练模型
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dataset = DinoDataset('./pics')
    train_dataset, test_dataset = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DinoNet().to(device)  # 输入图片大小为960x540
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01) 

    epochs = 5
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} Train Loss: {loss.item()} Test Loss: {test_loss/len(test_loader)} Accuracy: {100.*correct/total}%")

    torch.save(model.state_dict(), 'dino_model.pth')

if __name__ == "__main__":
    train()