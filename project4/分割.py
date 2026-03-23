import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_region_segmentation(input_path, output_dir="segmentation_results",
                              pore_thresh=(155, 165), oil_thresh=(130, 145),
                              stone_thresh=(160, 220), water_thresh=(0, 130)):
    """
    图像区域分割：针对包含水、油、石头、孔隙的混合材质图像，基于灰度阈值实现分割
    :param input_path: 输入图像路径（支持绝对路径/相对路径）
    :param output_dir: 结果保存目录（默认自动创建）
    :param pore_thresh: 孔隙灰度区间 (min, max)，默认(155, 165)
    :param oil_thresh: 油灰度区间 (min, max)，默认(130, 145)
    :param stone_thresh: 石头灰度区间 (min, max)，默认(160, 220)
    :param water_thresh: 水灰度区间 (min, max)，默认(0, 130)
    :return: 各物质分割掩码（字典形式）与可视化结果
    """
    # -------------------------- 1. 初始化与预处理 --------------------------
    # 创建结果保存目录
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取图像（OpenCV默认BGR格式，后续转灰度）
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径：{input_path}")

    # 彩色图转灰度图（实验核心预处理步骤）
    if len(img.shape) == 3:  # 若为3通道彩色图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # 若已为单通道灰度图，直接使用
        gray_img = img.copy()

    # 获取图像尺寸（用于创建掩码）
    height, width = gray_img.shape
    print(f"图像尺寸：{height}行 × {width}列")

    # -------------------------- 2. 基于灰度阈值生成分割掩码 --------------------------
    # 初始化4类物质的掩码（全0矩阵，uint8类型适配OpenCV）
    pore_mask = np.zeros((height, width), dtype=np.uint8)  # 孔隙掩码
    oil_mask = np.zeros((height, width), dtype=np.uint8)  # 油掩码
    stone_mask = np.zeros((height, width), dtype=np.uint8)  # 石头掩码
    water_mask = np.zeros((height, width), dtype=np.uint8)  # 水掩码

    # 依据灰度区间赋值：目标区域设为255（白色），非目标区域保持0（黑色）
    # 孔隙：灰度 > 155 且 < 165
    pore_mask[(gray_img > pore_thresh[0]) & (gray_img < pore_thresh[1])] = 255
    # 油：灰度 > 130 且 ≤ 145
    oil_mask[(gray_img > oil_thresh[0]) & (gray_img <= oil_thresh[1])] = 255
    # 石头：灰度 > 160 且 ≤ 220
    stone_mask[(gray_img > stone_thresh[0]) & (gray_img <= stone_thresh[1])] = 255
    # 水：灰度 ≤ 130（即 0-130）
    water_mask[(gray_img >= water_thresh[0]) & (gray_img <= water_thresh[1])] = 255

    # -------------------------- 3. 结果可视化（多子图对比） --------------------------
    # 转换原图为RGB格式（适配matplotlib显示，OpenCV默认BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else gray_img

    # 创建子图（2行3列：原图、灰度图、孔隙、油、水、石头）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Image Region Segmentation Results (Water/Oil/Stone/Pore)", fontsize=16)

    # 子图1：原始图像
    axes[0, 0].imshow(img_rgb, cmap=None if len(img.shape) == 3 else "gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")  # 隐藏坐标轴

    # 子图2：灰度图像（分割基础）
    axes[0, 1].imshow(gray_img, cmap="gray")
    axes[0, 1].set_title("Gray Image")
    axes[0, 1].axis("off")

    # 子图3：孔隙分割结果
    axes[0, 2].imshow(pore_mask, cmap="gray")
    axes[0, 2].set_title(f"Pore (Gray: {pore_thresh[0]}-{pore_thresh[1]})")
    axes[0, 2].axis("off")

    # 子图4：油分割结果
    axes[1, 0].imshow(oil_mask, cmap="gray")
    axes[1, 0].set_title(f"Oil (Gray: {oil_thresh[0]}-{oil_thresh[1]})")
    axes[1, 0].axis("off")

    # 子图5：水分割结果
    axes[1, 1].imshow(water_mask, cmap="gray")
    axes[1, 1].set_title(f"Water (Gray: {water_thresh[0]}-{water_thresh[1]})")
    axes[1, 1].axis("off")

    # 子图6：石头分割结果
    axes[1, 2].imshow(stone_mask, cmap="gray")
    axes[1, 2].set_title(f"Stone (Gray: {stone_thresh[0]}-{stone_thresh[1]})")
    axes[1, 2].axis("off")

    # 调整子图间距，避免标签重叠
    plt.tight_layout()
    # 保存可视化结果（高分辨率）
    plt.savefig(os.path.join(output_dir, "segmentation_visualization.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------- 4. 保存分割掩码（用于后续分析） --------------------------
    cv2.imwrite(os.path.join(output_dir, "pore_mask.png"), pore_mask)
    cv2.imwrite(os.path.join(output_dir, "oil_mask.png"), oil_mask)
    cv2.imwrite(os.path.join(output_dir, "water_mask.png"), water_mask)
    cv2.imwrite(os.path.join(output_dir, "stone_mask.png"), stone_mask)
    print(f"\n分割完成！结果已保存至：{os.path.abspath(output_dir)}")

    # -------------------------- 5. 返回结果（掩码字典） --------------------------
    return {
        "gray_image": gray_img,
        "pore_mask": pore_mask,
        "oil_mask": oil_mask,
        "water_mask": water_mask,
        "stone_mask": stone_mask
    }


# -------------------------- 主函数调用（实验入口） --------------------------
if __name__ == "__main__":
    # 1. 替换为你的实验图像路径（例如："mixed_material.jpg"）
    INPUT_IMAGE_PATH = "object.png"  # 关键：修改为实际图像路径

    # 2. （可选）调整灰度阈值（根据你的图像实际灰度分布）
    # 若不确定阈值，可先运行代码查看灰度图，再通过图像软件（如PS）获取各物质灰度区间
    CUSTOM_THRESHOLDS = {
        "pore_thresh": (155, 165),  # 孔隙灰度区间
        "oil_thresh": (130, 145),  # 油灰度区间
        "stone_thresh": (160, 220),  # 石头灰度区间
        "water_thresh": (0, 130)  # 水灰度区间
    }

    # 3. 执行分割
    try:
        segmentation_results = image_region_segmentation(
            input_path=INPUT_IMAGE_PATH,
            **CUSTOM_THRESHOLDS  # 传入自定义阈值
        )
    except Exception as e:
        print(f"分割过程出错：{str(e)}")