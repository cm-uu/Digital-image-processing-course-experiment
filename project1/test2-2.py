import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import random

def create_photo_mosaic(original_image_path, image_folder='images', output_path='output/photo_mosaic.jpg',
                      canvas_size=800, tile_size=30, overlap=0, final_alpha=0.3):
    #1.加载并调整原始图片大小
    try:
        original = Image.open(original_image_path).convert('RGB')
        original = original.resize((canvas_size, canvas_size), Image.LANCZOS)
    except FileNotFoundError:
        raise FileNotFoundError(f"原始图片未找到: {os.path.abspath(original_image_path)}")

    #2.获取所有马赛克图块图片
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        raise FileNotFoundError(f"请将图片放入 {os.path.abspath(image_folder)} 文件夹中")
    
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise FileNotFoundError(f"在 {image_folder} 中未找到任何图片")

    #3.预处理所有马赛克图块
    print("正在预处理马赛克图块...")
    tile_images = []
    for img_file in image_files:
        try:
            img = Image.open(os.path.join(image_folder, img_file))
            img = img.convert('RGB')
            # 调整图块大小，保持宽高比
            img_ratio = img.width / img.height
            if img_ratio > 1:
                new_width = tile_size
                new_height = int(tile_size / img_ratio)
            else:
                new_height = tile_size
                new_width = int(tile_size * img_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)
            tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
            x = (tile_size - new_width) // 2
            y = (tile_size - new_height) // 2
            tile.paste(img, (x, y))

            tile_images.append(tile)
        except Exception as e:
            print(f"警告: 无法处理图片 {img_file}: {e}")
    
    if not tile_images:
        raise ValueError("没有可用的马赛克图块图片")

    # 4. 创建马赛克层
    print("正在生成马赛克...")
    mosaic_layer = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
    spacing = max(1, int(tile_size * (1 - overlap)))  # 确保至少为1
    
    # 计算马赛克图块的行列数
    rows = (canvas_size + spacing - 1) // spacing
    cols = (canvas_size + spacing - 1) // spacing

    for i in range(rows):
        for j in range(cols):
            y = i * spacing
            x = j * spacing
            if x >= canvas_size or y >= canvas_size:
                continue
            region = original.crop((x, y, min(x + tile_size, canvas_size), min(y + tile_size, canvas_size)))
            tile = random.choice(tile_images).copy()
            enhancer = ImageEnhance.Brightness(tile)
            tile = enhancer.enhance(0.8)
            mosaic_layer.paste(tile, (x, y))

        progress = (i + 1) / rows * 100
        print(f"\r处理进度: {progress:.1f}%", end="")
    
    print("\n正在合成最终图像...")
    final_image = Image.blend(original, mosaic_layer, final_alpha)
    
    # 7. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_image.save(output_path, quality=95)
    print(f"马赛克图片已保存至: {os.path.abspath(output_path)}")
    
    return final_image

if __name__ == "__main__":
    try:
        if not os.path.exists('output'):
            os.makedirs('output')
        
        # 配置参数
        original_image = 'people.png'  # 原始图片路径
        output_image = 'output/photo_mosaic.jpg'  # 输出图片路径
        
        print(f"开始生成照片马赛克效果...")
        result = create_photo_mosaic(
            original_image_path=original_image,
            image_folder='images',  # 马赛克图块图片所在文件夹
            output_path=output_image,
            canvas_size=800,       # 输出图片大小
            tile_size=10,          # 每个马赛克图块的大小
            overlap=0,             # 图块重叠比例（0-1）
            final_alpha=0.35      # 马赛克层的不透明度（0-1）
        )
        
        print("生成完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n使用说明:")
        print(f"1. 确保原始图片 '{original_image}' 存在于程序目录下")
        print("2. 在程序所在目录下创建一个名为 'images' 的文件夹")
        print("3. 将用于马赛克的图片（如人脸图片）放入 'images' 文件夹中")
        print("4. 运行此脚本")
        print("\n提示:")
        print("- 建议使用至少20-30张不同的图片以获得更好的效果")
        print("- 可以调整 canvas_size 和 tile_size 参数来改变输出大小和细节程度")
        print("- 调整 final_alpha 参数（0-1之间）可以改变马赛克效果的强度，值越小原图越清晰")
