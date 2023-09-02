import os
import random

def balance_dataset(root_dir):
    # 获取jump和none的图片路径列表
    jump_images = [os.path.join(root_dir, 'jump', img_name) for img_name in os.listdir(os.path.join(root_dir, 'jump'))]
    none_images = [os.path.join(root_dir, 'none', img_name) for img_name in os.listdir(os.path.join(root_dir, 'none'))]

    # 计算需要删除的none图片数量
    num_to_delete = len(none_images) - len(jump_images)

    # 如果none的图片数量多于jump的图片数量，则随机删除一些none的图片
    if num_to_delete > 0:
        images_to_delete = random.sample(none_images, num_to_delete)
        for img_path in images_to_delete:
            os.remove(img_path)
        print(f"Deleted {num_to_delete} images from 'none' category.")

    else:
        print("No images need to be deleted. 'jump' category has more or equal images than 'none' category.")

if __name__ == "__main__":
    root_dir = './pics'
    balance_dataset(root_dir)