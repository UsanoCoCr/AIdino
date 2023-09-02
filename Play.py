import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageGrab
import numpy as np
import pyautogui
import time

# 定义模型结构（与train.py中的结构相同）
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

# 加载模型
model = DinoNet()
model.load_state_dict(torch.load('dino_model.pth'))
model.eval()

time.sleep(3) # 等待3秒

pyautogui.press('space') # 按空格键开始游戏

while True:
    print("Predicting...")
    # 截图
    img = ImageGrab.grab()
    img = img.resize((960, 540))
    img = torch.tensor(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # 使用模型预测
    with torch.no_grad():
        prediction = model(img)
    action = torch.argmax(prediction, dim=1).item()

    # 根据预测结果模拟按键操作
    if action == 0:
        pyautogui.press('up')
        print("Jump!")
    elif action == 1:
        pass