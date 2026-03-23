import os
import numpy as np
from PIL import Image, ImageDraw
import math
import random

def create_mosaic_from_mask(mask_path, image_folder='images', output_path='mosaic.jpg', 
                             canvas_size=800, tile_size=20, overlap=0.1):
    # 1. 获取所有图片
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        raise FileNotFoundError(f"请将图片放入 {os.path.abspath(image_folder)} 文件夹中")
    
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise FileNotFoundError(f"在 {image_folder} 中未找到任何图片")

    # 2. 加载并处理蒙版图片
    try:
        mask_im = Image.open(mask_path).convert('L') # 转换为灰度图
        mask_im = mask_im.resize((canvas_size, canvas_size), Image.LANCZOS)
    except FileNotFoundError:
        raise FileNotFoundError(f"蒙版图片未找到: {os.path.abspath(mask_path)}")

    # 3. 创建画布
    canvas = Image.new('RGB', (canvas_size, canvas_size), 'white')
    
    # 4. 计算图块间距和偏移量
    spacing = int(tile_size * (1 - overlap))
    offset_x = (canvas_size % spacing) // 2
    offset_y = (canvas_size % spacing) // 2
    
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
    total_tiles = ((canvas_size - offset_y) // spacing + 1) * ((canvas_size - offset_x) // spacing + 1)
    processed_tiles = 0
    
    for y in range(offset_y, canvas_size, spacing):
        for x in range(offset_x, canvas_size, spacing):
            # 检查蒙版对应位置的像素值，深色 (<128) 则填充
            if mask_im.getpixel((x, y)) < 128:
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
        
        # --- 配置 --- #
        mask_image_path = 'hua.png'  # <-- 使用你的五角星图片
        output_image_path = 'output/hua_mosaic.jpg'

        # 生成马赛克
        print(f"开始使用 '{mask_image_path}' 作为蒙版生成马赛克...")
        mosaic = create_mosaic_from_mask(
            mask_path=mask_image_path,
            image_folder='images',  # 图片文件夹
            output_path=output_image_path,  # 输出路径
            canvas_size=1400,  # 画布大小
            tile_size=20,      # 每个小图块大小
            overlap=0.0        # 图块重叠比例
        )

        print(f"正在显示生成的马赛克...")
        mosaic.show()
        print("完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n使用说明:")
        print("1. 在程序所在目录下创建一个名为 'images' 的文件夹")
        print("2. 将你的图片放入 'images' 文件夹中")
        print(f"3. 确保蒙版图片 (例如: '{mask_image_path}') 存在于程序目录下")
        print("4. 运行此脚本")
        print("\n提示:")
        print("- 建议使用至少10张不同的图片以获得更好的效果")
        print("- 可以调整 canvas_size 和 tile_size 参数来改变输出大小和细节程度")
