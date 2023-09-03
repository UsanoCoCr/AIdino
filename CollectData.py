import time
import os
from PIL import ImageGrab
import keyboard

# 创建存储图片的文件夹
if not os.path.exists('./pics/jump_temp'):
    os.makedirs('./pics/jump_temp')
if not os.path.exists('./pics/none_temp'):
    os.makedirs('./pics/none_temp')

time.sleep(3)  # 等待3秒

while True:
    # 截图
    img = ImageGrab.grab()
    img = img.resize((960, 540))
    
    # 检查按键
    if keyboard.is_pressed('up arrow'):
        img.save(f"./pics/jump_temp/{int(time.time() * 1000)}.png")
        time.sleep(0.2)
    else:
        img.save(f"./pics/none_temp/{int(time.time() * 1000)}.png")
        time.sleep(0.2)
