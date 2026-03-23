import os
import numpy as np
from PIL import Image, ImageDraw
import math
import random

def create_heart_mosaic(image_folder='images', output_path='heart_mosaic.jpg', 
                       heart_size=400, tile_size=20, overlap=0.1):
    # 1. 获取所有图片
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        raise FileNotFoundError(f"请将图片放入 {os.path.abspath(image_folder)} 文件夹中")
    
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise FileNotFoundError(f"在 {image_folder} 中未找到任何图片")
    
    # 2. 创建心形蒙版
    def is_in_heart(x, y, scale=1.0):
        x = (x - 0.5) * 3 * scale
        y = (0.5 - y) * 3 * scale
        # 心形方程
        return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0
    
    # 3. 创建画布
    canvas = Image.new('RGB', (heart_size, heart_size), 'white')
    
    # 4. 计算图块间距和偏移量
    spacing = int(tile_size * (1 - overlap))
    offset_x = (heart_size % spacing) // 2
    offset_y = (heart_size % spacing) // 2
    
    # 5. 加载并处理所有图片
    images = []
    for img_file in image_files:
        try:
            img = Image.open(os.path.join(image_folder, img_file))
            img = img.convert('RGB')
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"警告: 无法加载图片 {img_file}: {e}")
    
    if not images:
        raise ValueError("没有可用的图片")
    
    # 6. 绘制马赛克
    total_tiles = ((heart_size - offset_y) // spacing + 1) * ((heart_size - offset_x) // spacing + 1)
    processed_tiles = 0
    
    for y in range(offset_y, heart_size, spacing):
        for x in range(offset_x, heart_size, spacing):
            nx = x / heart_size
            ny = y / heart_size
            if is_in_heart(nx, ny, scale=0.9):
                img = random.choice(images)
                canvas.paste(img, (x, y))
            
            processed_tiles += 1
            if processed_tiles % 100 == 0:
                progress = processed_tiles / total_tiles * 100
                print(f"\r处理进度: {min(100, progress):.1f}%", end="")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path)
    print(f"\n马赛克图片已保存至: {os.path.abspath(output_path)}")
    return canvas

if __name__ == "__main__":
    try:
        if not os.path.exists('output'):
            os.makedirs('output')
            
        # 生成马赛克
        print("开始生成心形马赛克...")
        mosaic = create_heart_mosaic(
            image_folder='images',
            output_path='output/heart_mosaic.jpg',
            heart_size=1400,
            tile_size=10,
            overlap=0
        )

        print("正在显示生成的心形马赛克...")
        mosaic.show()
        print("完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n使用说明:")
        print("1. 在程序所在目录下创建一个名为 'images' 的文件夹")
        print("2. 将你的图片放入该文件夹中")
        print("3. 运行此脚本")
        print("\n提示:")
        print("- 建议使用至少10张不同的图片以获得更好的效果")
        print("- 可以调整 heart_size 和 tile_size 参数来改变心形大小和细节程度")