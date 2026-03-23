import os
import numpy as np
from PIL import Image, ImageEnhance
import random
from collections import defaultdict

def get_average_color(image):
    """计算图像的平均颜色"""
    # 将图像缩小到1x1像素，然后获取其颜色
    small = image.resize((1, 1), resample=Image.BILINEAR)
    return small.getpixel((0, 0))

def color_distance(c1, c2):
    """计算两个颜色之间的欧氏距离"""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def find_best_match(target_color, image_dict):
    """
    在图片集中找到颜色最接近的图片
    
    参数:
        target_color: 目标颜色 (R, G, B)
        image_dict: 图片字典，格式为 {图片路径: 平均颜色}
        
    返回:
        最匹配的图片路径，如果没有可用的则返回None
    """
    min_distance = float('inf')
    best_image = None
    
    for img_path, img_color in image_dict.items():
        dist = color_distance(target_color, img_color)
        if dist < min_distance:
            min_distance = dist
            best_image = img_path
    
    return best_image

def create_photo_mosaic(original_image, image_folder, output_path, 
                       tile_size=30, 
                       enhance_match=True,
                       enhance_strength=0.5,
                       allow_reuse=True,
                       max_usage=3):
    # 1. 加载并调整原始图片大小
    print("正在加载原始图片...")
    try:
        original = Image.open(original_image).convert('RGB')
        width, height = original.size
        cols = width // tile_size
        rows = height // tile_size
        canvas_width = cols * tile_size
        canvas_height = rows * tile_size
        
        if (width, height) != (canvas_width, canvas_height):
            original = original.resize((canvas_width, canvas_height), Image.LANCZOS)
    except Exception as e:
        raise Exception(f"加载原始图片失败: {e}")

    # 2. 加载并预处理所有马赛克图块
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        raise FileNotFoundError(f"请将图片放入 {os.path.abspath(image_folder)} 文件夹中")
    
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise FileNotFoundError(f"在 {image_folder} 中未找到任何图片")
    print("正在预处理马赛克图块...")
    image_dict = {}
    for img_file in image_files:
        try:
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('RGB')
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
            avg_color = get_average_color(tile)
            image_dict[img_path] = (tile, avg_color)

        except Exception as e:
            print(f"警告: 无法处理图片 {img_file}: {e}")
    
    if not image_dict:
        raise ValueError("没有可用的马赛克图块图片")
    
    # 3. 创建马赛克画布
    print("正在生成马赛克...")
    mosaic = Image.new('RGB', (canvas_width, canvas_height))
    
    # 用于记录每张图片的使用次数
    usage_count = {img_path: 0 for img_path in image_dict.keys()}
    
    # 4. 为每个网格选择最佳匹配的图片
    for row in range(rows):
        for col in range(cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            region = original.crop((left, upper, right, lower))
            target_color = get_average_color(region)
            if allow_reuse:
                best_match = find_best_match(
                    target_color, 
                    {k: v[1] for k, v in image_dict.items()} )
            else:
                available_imgs = {
                    k: v[1] for k, v in image_dict.items() 
                    if usage_count[k] < max_usage }
                if not available_imgs:
                    print("\n警告：可用图片不足，部分图片将被重复使用")
                    available_imgs = {k: v[1] for k, v in image_dict.items()}
                best_match = find_best_match(target_color, available_imgs)

                
            if best_match is None:
                print("\n错误：无法找到合适的图片，请检查图片库")
                return
            tile_img = image_dict[best_match][0].copy()
            if enhance_match:
                strength = max(0, min(1, enhance_strength))
                hsv_img = tile_img.convert('HSV')
                h, s, v = hsv_img.split()
                target_hsv = Image.new('RGB', (1, 1), target_color).convert('HSV')
                target_h, target_s, _ = target_hsv.getpixel((0, 0))
                h = h.point(lambda p: int(p * (1 - strength) + target_h * strength))
                s = s.point(lambda p: min(255, int(p * (1 + 0.5 * strength))))
                tile_img = Image.merge('HSV', (h, s, v)).convert('RGB')
            mosaic.paste(tile_img, (left, upper))
            usage_count[best_match] += 1
            
        # 显示进度
        progress = (row + 1) / rows * 100
        print(f"\r处理进度: {progress:.1f}%", end="")
    
    print("\n正在合成最终图像...")
    
    # 5. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mosaic.save(output_path, quality=95)
    
    # 6. 输出统计信息
    print("\n马赛克生成完成！")
    print(f"使用的图片总数: {len(usage_count)} / {len(image_files)}")
    print(f"输出图片已保存至: {os.path.abspath(output_path)}")
    
    return mosaic

if __name__ == "__main__":
    try:
        # 创建输出目录
        if not os.path.exists('output'):
            os.makedirs('output')
        
        # 配置参数
        config = {
            'original_image': "pic1.jpg",    # 原始图片路径
            'image_folder': "images",       # 马赛克图块文件夹
            'output_path': "output.jpg",    # 输出图片路径
            'tile_size': 40,               # 马赛克图块大小（像素）
            'enhance_match': True,         # 是否增强颜色匹配
            'enhance_strength': 0.7,       # 颜色增强强度（0-1）
            'allow_reuse': False,           # 是否允许图片重复使用
            'max_usage': 3                  # 每张图片最多使用次数（当allow_reuse为False时有效）
        }
        
        print("=== 照片马赛克生成器 ===")
        print(f"原始图片: {config['original_image']}")
        print(f"图块大小: {config['tile_size']}px")
        print(f"图片重复使用: {'允许' if config['allow_reuse'] else '不允许'}")
        if not config['allow_reuse']:
            print(f"每张图片最多使用次数: {config['max_usage']}")
        print("=" * 30)
        
        # 生成马赛克
        result = create_photo_mosaic(**config)
        
        print("\n生成完成！")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n使用说明:")
        print("1. 确保原始图片存在于程序目录下")
        print("2. 在程序所在目录下创建一个名为 'images' 的文件夹")
        print("3. 将用于马赛克的图片放入 'images' 文件夹中")
        print("4. 运行此脚本")
        print("\n提示:")
        print("- 建议使用至少30-50张不同的图片以获得更好的效果")
        print("- 可以调整 tile_size 参数来改变马赛克图块的大小")
