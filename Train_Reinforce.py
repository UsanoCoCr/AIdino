import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageGrab, ImageOps
import numpy as np
import pyautogui
import random
import time

pyautogui.FAILSAFE = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
reward_live = 1
reward_die = -10

# 从chrome://dino导入环境
class DinoEnv:
    def __init__(self):
        self.action_space = [0, 1] # 0: do nothing, 1: jump
        self.reset()
    
    def reset(self):
        pyautogui.press('space')
        time.sleep(0.1)
        return self.get_state()
    
    def step(self, action):
        # if two images are the same, then the dino is dead
        # return next_state, reward, done
        img1 = self.get_state()
        time.sleep(0.1)
        img2 = self.get_state()
        if torch.equal(img1, img2):
            return img2, reward_die, True
        if action == 1:
            pyautogui.press('space')
        time.sleep(0.1)
        return self.get_state(), reward_live, False

    def get_state(self):
        img = ImageGrab.grab()
        img = img.resize((960, 540))
        img = ImageOps.grayscale(img)  
        state = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return state

# 定义模型
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

class DinoAgent:
    def __init__(self):
        self.env = DinoEnv()
        self.model = DinoCNN().to(device)
        self.model.load_state_dict(torch.load('./models/model_100.pth'))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = [] # 存储格式(state, action, reward, next_state)
        self.gamma = 0.99 # 折扣因子

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space[random.randint(0, 1)]
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                step_epsilon = 0.6
                action = self.choose_action(state, step_epsilon)
                next_state, reward, done = self.env.step(action)
                self.memory.append((state, action, reward, next_state))
                self.optimize_model()
                state = next_state
            print(f"Episode {episode} finished.")
            if episode % 100 == 0:
                torch.save(self.model.state_dict(), f"./models/model_{episode}.pth")    

    def optimize_model(self):
        if len(self.memory) < 32:  # Wait until there are enough experiences in memory
            return
        
        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states)
        
        # Compute the Q-values for the current states and next states
        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1))
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values)
        
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = DinoAgent()
time.sleep(2)
agent.train(1000)