import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_region_segmentation(input_path, output_dir="segmentation_results",
                              pore_thresh=(155, 165),
                              stone_thresh=(160, 220),
                              water_thresh=(0, 130)):
    """
    图像区域分割：针对包含水、油、石头、孔隙的混合材质图像，油采用多颜色空间分割，其他物质基于灰度阈值
    :param input_path: 输入图像路径
    :param output_dir: 结果保存目录
    :param pore_thresh: 孔隙灰度区间 (min, max)
    :param stone_thresh: 石头灰度区间 (min, max)
    :param water_thresh: 水灰度区间 (min, max)
    :return: 各物质分割掩码与可视化结果
    """
    # -------------------------- 1. 初始化与预处理 --------------------------
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径：{input_path}")

    # 彩色图转灰度图（用于其他物质分割）
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    height, width = gray_img.shape
    print(f"图像尺寸：{height}行 × {width}列")

    # -------------------------- 2. 分割逻辑 --------------------------
    # 初始化掩码
    pore_mask = np.zeros((height, width), dtype=np.uint8)
    stone_mask = np.zeros((height, width), dtype=np.uint8)
    water_mask = np.zeros((height, width), dtype=np.uint8)

    # 基于灰度的分割（孔隙、石头、水）
    pore_mask[(gray_img > pore_thresh[0]) & (gray_img < pore_thresh[1])] = 255
    stone_mask[(gray_img > stone_thresh[0]) & (gray_img <= stone_thresh[1])] = 255
    water_mask[(gray_img >= water_thresh[0]) & (gray_img <= water_thresh[1])] = 255

    # 基于多颜色空间的油分割（采用第二个文件的方法）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # HSV黄色范围
    lower_hsv_yellow = np.array([8, 40, 40])
    upper_hsv_yellow = np.array([45, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_hsv_yellow, upper_hsv_yellow)

    # Lab黄色范围
    lower_lab_yellow = np.array([0, 0, 120])
    upper_lab_yellow = np.array([255, 40, 255])
    lab_mask = cv2.inRange(lab, lower_lab_yellow, upper_lab_yellow)

    # 融合掩码
    oil_mask = cv2.bitwise_or(hsv_mask, lab_mask)

    # 形态学操作
    kernel_tiny = np.ones((2, 2), np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)
    oil_mask = cv2.dilate(oil_mask, kernel_tiny, iterations=2)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 连通区域过滤
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(oil_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 2:
            oil_mask[labels == i] = 0

    # -------------------------- 3. 结果可视化 --------------------------
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else gray_img

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Image Region Segmentation Results", fontsize=16)

    axes[0, 0].imshow(img_rgb, cmap=None if len(img.shape) == 3 else "gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_img, cmap="gray")
    axes[0, 1].set_title("Gray Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pore_mask, cmap="gray")
    axes[0, 2].set_title(f"Pore (Gray: {pore_thresh[0]}-{pore_thresh[1]})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(oil_mask, cmap="gray")
    axes[1, 0].set_title("Oil (HSV+Lab Fusion)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(water_mask, cmap="gray")
    axes[1, 1].set_title(f"Water (Gray: {water_thresh[0]}-{water_thresh[1]})")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(stone_mask, cmap="gray")
    axes[1, 2].set_title(f"Stone (Gray: {stone_thresh[0]}-{stone_thresh[1]})")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_visualization.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------- 4. 保存结果 --------------------------
    cv2.imwrite(os.path.join(output_dir, "pore_mask.png"), pore_mask)
    cv2.imwrite(os.path.join(output_dir, "oil_mask.png"), oil_mask)
    cv2.imwrite(os.path.join(output_dir, "water_mask.png"), water_mask)
    cv2.imwrite(os.path.join(output_dir, "stone_mask.png"), stone_mask)
    print(f"\n分割完成！结果已保存至：{os.path.abspath(output_dir)}")

    return {
        "gray_image": gray_img,
        "pore_mask": pore_mask,
        "oil_mask": oil_mask,
        "water_mask": water_mask,
        "stone_mask": stone_mask
    }


if __name__ == "__main__":
    INPUT_IMAGE_PATH = "object.png"
    CUSTOM_THRESHOLDS = {
        "pore_thresh": (155, 165),
        "stone_thresh": (160, 220),
        "water_thresh": (0, 130)
    }

    try:
        segmentation_results = image_region_segmentation(
            input_path=INPUT_IMAGE_PATH,** CUSTOM_THRESHOLDS
        )
    except Exception as e:
        print(f"分割过程出错：{str(e)}")