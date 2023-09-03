from PIL import ImageGrab, ImageOps
import numpy as np
import pyautogui
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

pyautogui.FAILSAFE = False

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = DinoCNN().to(device)
model.load_state_dict(torch.load('dino_model_pytorch.pth'))
model.eval()

specific_region_coordinates = (170, 260, 290, 400)
specific_region_coordinates_late = (250, 260, 330, 400)
threshold = 252.2
threshold_large_cactus = 228
large_cactus_delay = 0.1

def delay_based_on_time(start_time):
    elapsed_time = time.time() - start_time
    if elapsed_time > 20:
        return False
    elif elapsed_time > 12:
        if is_cactus_present_late(img):
            return False
        return True
    else:
        time.sleep(0.05)
        isCactus, isLarge = is_cactus_present(img)
        if isCactus:
            if isLarge:
                time.sleep(large_cactus_delay)
            return False
        return True

def is_cactus_present(img):
    judge_large_cactus = False
    region = img.crop((specific_region_coordinates)) 
    region.save('region.png')
    avg_pixel_intensity = np.mean(np.array(region))
    #print(avg_pixel_intensity)
    if avg_pixel_intensity < threshold_large_cactus:
        judge_large_cactus = True
    if avg_pixel_intensity > threshold:
        return False, judge_large_cactus
    return True, judge_large_cactus

def is_cactus_present_late(img):
    region = img.crop((specific_region_coordinates_late)) 
    region.save('region.png')
    avg_pixel_intensity = np.mean(np.array(region))
    #print(avg_pixel_intensity)
    if avg_pixel_intensity > threshold:
        return False
    return True

time.sleep(3)  # Wait for 3 seconds
pyautogui.press('space')  # Press space to start the game
start_time = time.time()

while True:
    #print("Predicting...")
    img = ImageGrab.grab()
    img = img.resize((960, 540))
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    if predicted.item() == 0:
        if delay_based_on_time(start_time):
            #print("Delaying...")
            continue
        pyautogui.press('up')
        #print("Jump!")
    elif predicted.item() == 1:
        pass