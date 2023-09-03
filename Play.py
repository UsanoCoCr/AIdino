from PIL import ImageGrab, ImageOps
import numpy as np
import pyautogui
import time
import joblib

# 加载模型
model = joblib.load('dino_model.pth')

additional_delay_for_cactus = 0.08
specific_region_coordinates = (250, 260, 500, 400)
threshold = 242

def delay_based_on_time(start_time, initial_delay=0.2, increase_rate=0.008, min_delay=0):
    elapsed_time = time.time() - start_time
    delay = max(initial_delay - (increase_rate * elapsed_time), min_delay)
    if is_large_cactus_present(img) and elapsed_time < 30:
            delay += additional_delay_for_cactus
    return delay

def is_large_cactus_present(img):
    region = img.crop((specific_region_coordinates)) 
    avg_pixel_intensity = np.mean(np.array(region))

    region.save('region.png')
    print(avg_pixel_intensity)

    if avg_pixel_intensity < threshold:
        return True
    return False

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