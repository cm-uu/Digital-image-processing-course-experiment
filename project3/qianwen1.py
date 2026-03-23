import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
elephant = cv2.imread('t4.png')   # BGR 格式
cheetah = cv2.imread('t3.png')

# 调整到相同尺寸
h, w = min(elephant.shape[0], cheetah.shape[0]), min(elephant.shape[1], cheetah.shape[1])
elephant = cv2.resize(elephant, (w, h))
cheetah = cv2.resize(cheetah, (w, h))

# 1. 对大象做低通滤波（高斯模糊）
low_pass = cv2.GaussianBlur(elephant, (0, 0), sigmaX=16, sigmaY=16)

# 2. 对猎豹做高通滤波：原图 - 低通（注意：这里也用高斯模糊提取低频）
cheetah_low = cv2.GaussianBlur(cheetah, (0, 0), sigmaX=24, sigmaY=24)
high_pass = cheetah.astype(np.float32) - cheetah_low.astype(np.float32)

# 3. 合成：低频（大象） + 高频（猎豹）
result = low_pass.astype(np.float32) + high_pass

# 限制像素值到 [0, 255] 并转回 uint8
result = np.clip(result, 0, 255).astype(np.uint8)

# 显示（转换 BGR → RGB 用于 matplotlib）
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(elephant, cv2.COLOR_BGR2RGB))
plt.title("Elephant (Original)")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(low_pass, cv2.COLOR_BGR2RGB))
plt.title("Elephant (Low-pass)")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(cheetah, cv2.COLOR_BGR2RGB))
plt.title("Cheetah (Original)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor((high_pass + 128).clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Cheetah (High-pass, offset for visibility)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Fused Color Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# 可选：保存结果
cv2.imwrite('fused_color_result.jpg', result)