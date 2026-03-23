import cv2
import numpy as np


def compute_stats(lab_img):
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    l_mean = np.mean(l_channel)
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)

    l_std = np.std(l_channel, ddof=1)
    a_std = np.std(a_channel, ddof=1)
    b_std = np.std(b_channel, ddof=1)

    return {
        'l': {'mean': l_mean, 'std': l_std},
        'a': {'mean': a_mean, 'std': a_std},
        'b': {'mean': b_mean, 'std': b_std}
    }


def color_transfer(source_path, target_path, output_path):
    # 1. 读取图像（OpenCV 默认读取为 BGR 格式）
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

    # 3. 计算源图像和目标图像的统计特征
    source_stats = compute_stats(source_lab)
    target_stats = compute_stats(target_lab)

    #4.分离源图像的Lab通道
    t_l, t_a, t_b = cv2.split(target_lab)


    eps = 1e-6

    # 逐通道进行统计匹配
    #去中心化-缩放-平移
    normalized_l = (t_l - target_stats['l']['mean']) * (source_stats['l']['std'] / (target_stats['l']['std'] + eps)) + \
                   source_stats['l']['mean']
    normalized_a = (t_a - target_stats['a']['mean']) * (source_stats['a']['std'] / (target_stats['a']['std'] + eps)) + \
                   source_stats['a']['mean']
    normalized_b = (t_b - target_stats['b']['mean']) * (source_stats['b']['std'] / (target_stats['b']['std'] + eps)) + \
                   source_stats['b']['mean']


    # 6. 限制像素值范围在 [0, 255]（符合 8 位图像格式）
    normalized_l = np.clip(normalized_l, 0, 255)
    normalized_a = np.clip(normalized_a, 0, 255)
    normalized_b = np.clip(normalized_b, 0, 255)

    normalized_lab = cv2.merge([
        normalized_l.astype(np.uint8),
        normalized_a.astype(np.uint8),
        normalized_b.astype(np.uint8)
    ])

    result_img = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, result_img)
    print(f"颜色迁移完成，结果已保存至：{output_path}")

    cv2.imshow("Source Image", source_img)
    cv2.imshow("Target Image", target_img)
    cv2.imshow("Result Image", result_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ##d

    SOURCE_IMAGE_PATH = "c2.png"
    TARGET_IMAGE_PATH = "c1.png"
    OUTPUT_IMAGE_PATH = "result.jpg"

    try:
        color_transfer(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    except Exception as e:
        print(f"程序执行出错：{e}")