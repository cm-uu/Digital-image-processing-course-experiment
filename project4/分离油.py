import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def final_refine_oil_segmentation(input_path, output_dir="final_oil_results"):
    """
    终极优化油分割：覆盖所有小面积黄色油
    """
    # 1. 初始化与图像读取
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # 增加Lab颜色空间（黄色在Lab中特征更稳定）

    # 2. 多颜色空间融合识别黄色（HSV+Lab，覆盖所有黄色色调）
    # HSV黄色范围（进一步放宽）
    lower_hsv_yellow = np.array([8, 40, 40])
    upper_hsv_yellow = np.array([45, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_hsv_yellow, upper_hsv_yellow)

    # Lab黄色范围（a通道偏负，b通道偏正）
    lower_lab_yellow = np.array([0, 0, 120])
    upper_lab_yellow = np.array([255, 40, 255])
    lab_mask = cv2.inRange(lab, lower_lab_yellow, upper_lab_yellow)

    # 融合HSV+Lab掩码（取并集，覆盖更多黄色区域）
    oil_mask = cv2.bitwise_or(hsv_mask, lab_mask)

    # 3. 形态学操作（极致保留小区域）
    kernel_tiny = np.ones((2, 2), np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)
    # 先膨胀2次（最大化放大小油区域）→ 轻度开运算去噪 → 闭运算填充
    oil_mask = cv2.dilate(oil_mask, kernel_tiny, iterations=2)  # 放大微小小区域
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 4. 连通区域保留（面积阈值降至2，仅过滤像素级噪声）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(oil_mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2:  # 仅过滤面积<2的极端噪声
            oil_mask[labels == i] = 0

    # 5. 可视化最终结果
    plt.figure(figsize=(12, 6))
    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image (All Small Yellow Oil)")
    plt.axis("off")
    # 最终油分割结果
    oil_vis = img_rgb.copy()
    oil_vis[oil_mask > 0] = [255, 0, 0]  # 油区域标红
    plt.subplot(1, 2, 2)
    plt.imshow(oil_vis)
    plt.title("Final Oil Segmentation (All Small Areas Included)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_oil_vis.png"), dpi=300)
    plt.show()

    # 6. 保存最终掩码
    cv2.imwrite(os.path.join(output_dir, "final_oil_mask.png"), oil_mask)
    print(f"最终油分割完成！所有小油区域已覆盖")
    return oil_mask


# 主函数调用
if __name__ == "__main__":
    INPUT_IMAGE_PATH = "object.png"  # 替换为你的图像路径
    try:
        final_oil_mask = final_refine_oil_segmentation(INPUT_IMAGE_PATH)
    except Exception as e:
        print(f"处理出错：{str(e)}")