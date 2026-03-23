import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def image_region_segmentation(input_path, output_dir="segmentation_results",
                              pore_thresh=(150, 220),
                              water_thresh=(0, 130)):
    # -------------------------- 1. 初始化与预处理 --------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径：{input_path}")

    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    height, width = gray_img.shape
    print(f"图像尺寸：{height} × {width}")

    # 在读取图像后、分割前，加入这段代码
    plt.figure(figsize=(10, 5))
    plt.hist(gray_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title("Gray Level Histogram")
    plt.xlabel("Gray Value (0-255)")
    plt.ylabel("Pixel Count")
    plt.grid(True)
    plt.show()


    # -------------------------- 2. 独立阈值分割 --------------------------
    water_mask = np.zeros((height, width), dtype=np.uint8)
    pore_mask = np.zeros((height, width), dtype=np.uint8)

    water_mask[(gray_img >= 0) & (gray_img <= 130)] = 255

    pore_mask[(gray_img > 150) & (gray_img < 170)] = 255

    # -------------------------- 3. 油分割（不变）--------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    lower_hsv_yellow = np.array([8, 40, 40])
    upper_hsv_yellow = np.array([45, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_hsv_yellow, upper_hsv_yellow)

    lower_lab_yellow = np.array([0, 0, 120])
    upper_lab_yellow = np.array([255, 40, 255])
    lab_mask = cv2.inRange(lab, lower_lab_yellow, upper_lab_yellow)

    oil_mask = cv2.bitwise_or(hsv_mask, lab_mask)

    # 形态学优化
    kernel_tiny = np.ones((2, 2), np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)
    oil_mask = cv2.dilate(oil_mask, kernel_tiny, iterations=2)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 小区域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(oil_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 2:
            oil_mask[labels == i] = 0

    # -------------------------- 4. 可视化 --------------------------
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Image Region Segmentation (Independent Thresholding)", fontsize=16)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_img, cmap="gray")
    axes[0, 1].set_title("Grayscale Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pore_mask, cmap="gray")
    axes[0, 2].set_title(f"Pore (Gray: {pore_thresh[0]}–{pore_thresh[1]})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(oil_mask, cmap="gray")
    axes[1, 0].set_title("Oil (HSV+Lab)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(water_mask, cmap="gray")
    axes[1, 1].set_title(f"Water (Gray: {water_thresh[0]}–{water_thresh[1]})")
    axes[1, 1].axis("off")


    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[oil_mask == 255] = [0, 255, 255]   # 油：青色 (BGR: [0,255,255] → 显示为黄色)
    overlay[water_mask == 255] = [255, 0, 0]   # 水：蓝色
    overlay[pore_mask == 255] = [0, 255, 0]    # 孔隙：绿色
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Overlay (Oil:Yellow, Water:Blue, Pore:Green)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_visualization.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------- 5. 保存结果 --------------------------
    cv2.imwrite(os.path.join(output_dir, "pore_mask.png"), pore_mask)
    cv2.imwrite(os.path.join(output_dir, "oil_mask.png"), oil_mask)
    cv2.imwrite(os.path.join(output_dir, "water_mask.png"), water_mask)
    print(f"\n分割完成！结果保存至：{os.path.abspath(output_dir)}")

    return {
        "gray_image": gray_img,
        "pore_mask": pore_mask,
        "oil_mask": oil_mask,
        "water_mask": water_mask
    }


if __name__ == "__main__":
    INPUT_IMAGE_PATH = "object.png"
    CUSTOM_THRESHOLDS = {
        "pore_thresh": (155, 165),
        "water_thresh": (0, 130)
    }

    try:
        segmentation_results = image_region_segmentation(
            input_path=INPUT_IMAGE_PATH, **CUSTOM_THRESHOLDS
        )
    except Exception as e:
        print(f"分割出错：{e}")