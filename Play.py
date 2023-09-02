from PIL import ImageGrab, ImageOps
import numpy as np
import pyautogui
import time
import joblib

# 加载模型
model = joblib.load('dino_model.pth')

def delay_based_on_time(start_time, initial_delay=0.2, increase_rate=0.008, min_delay=0):
    elapsed_time = time.time() - start_time
    delay = max(initial_delay - (increase_rate * elapsed_time), min_delay)
    return delay

time.sleep(3) # 等待3秒
pyautogui.press('space') # 按空格键开始游戏
start_time = time.time()

while True:
    print("Predicting...")
    # 截图并处理
    img = ImageGrab.grab()
    img = img.resize((960, 540))
    img = ImageOps.grayscale(img)  # 转换为灰度图像
    img_np = np.array(img).reshape(1, -1)

    prediction = model.predict(img_np)

    # 根据预测结果模拟按键操作
    if prediction[0] == 0:
        time.sleep(delay_based_on_time(start_time))  # 根据游戏时间增加延迟
        pyautogui.press('up')
        print("Jump!")
    elif prediction[0] == 1:
        pass