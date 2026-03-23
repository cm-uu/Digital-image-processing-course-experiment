import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
def spatial_enhancement_medical(image_path):

    pil_img = Image.open(image_path).convert('L')#转为灰度模式
    imgA = np.array(pil_img, dtype=np.float64)


    # 图B计算拉普拉斯边缘
    imgB = cv2.Laplacian(imgA, cv2.CV_64F)
    #归一化
    imgB = 255 * (imgB - imgB.min()) / (imgB.max() - imgB.min() + 1e-8)
    imgB = imgB.astype(np.float64)

    # C=A+B
    imgC = imgA + imgB
    imgC = 255 * (imgC - imgC.min()) / (imgC.max() - imgC.min() + 1e-8)
    imgC = imgC.astype(np.float64)


    sobel_x = cv2.Sobel(imgA, cv2.CV_64F, dx=1, dy=0)
    sobel_y = cv2.Sobel(imgA, cv2.CV_64F, dx=0, dy=1)
    imgD = cv2.magnitude(sobel_x, sobel_y)
    imgD = 255 * (imgD - imgD.min()) / (imgD.max() - imgD.min() + 1e-8)
    imgD = imgD.astype(np.float64)

    # 5.中值滤波(5*5 的核)
    imgE = cv2.medianBlur(np.uint8(imgD), ksize=5)
    imgE = imgE.astype(np.float64)


    imgF = imgC * imgE

    imgF = 255 * (imgF - imgF.min()) / (imgF.max() - imgF.min() + 1e-8)
    imgF = imgF.astype(np.float64)


    imgG = imgA + imgF
    # 归一化到[0,255]
    imgG = 255 * (imgG - imgG.min()) / (imgG.max() - imgG.min() + 1e-8)
    imgG = imgG.astype(np.float64)  # 图像G（A+F结果）


    imgH = np.power(imgG/ 255.0, 0.5)
    imgH = imgH * 255.0
    imgH = imgH.astype(np.uint8)


    results = {
        'A': imgA.astype(np.uint8),
        'B_Laplacian': imgB.astype(np.uint8),
        'C_A+B': imgC.astype(np.uint8),
        'D_Sobel': imgD.astype(np.uint8),
        'E_5x5': imgE.astype(np.uint8),
        'F_C*E': imgF.astype(np.uint8),
        'G_A+F': imgG.astype(np.uint8),
        'H_(γ=0.5)': imgH
    }
    return results

def show_and_save_results(results, save_path_prefix="medical_enhancement_"):
    plt.figure(figsize=(16, 10))
    for i, (name, img) in enumerate(results.items(), 1):
        plt.subplot(2, 4, i)
        plt.imshow(img, cmap='gray')
        plt.title(name, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 保存所有结果（便于对比报告）
    for name, img in results.items():
        save_name = save_path_prefix + name.replace(' ', '_').replace('(', '').replace(')', '').replace('*', '')
        cv2.imwrite(f"{save_name}.jpg", img)
    print("所有步骤结果已保存，前缀为：", save_path_prefix)

if __name__ == '__main__':

    image_path = r"object.png"
    try:
        # 执行空域增强
        enhancement_results = spatial_enhancement_medical(image_path)
        # 显示并保存结果
        show_and_save_results(enhancement_results)
        print("实验完成！所有步骤与报告一致，结果已保存和显示。")
    except Exception as e:
        print(f"出错：{e}")
        print("请检查：1. 图像路径是否正确；2. 图像是否为有效格式；3. 依赖库是否安装（opencv-python、pillow、matplotlib、numpy）")