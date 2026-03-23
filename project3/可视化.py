import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_lowpass_filter_2d(shape, sigma):
    """生成频域高斯低通滤波器（2D）"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)
    D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
    H = np.exp(- (D ** 2) / (2 * (sigma ** 2)))
    return H

# 设置参数
shape = (100, 100)   # 滤波器尺寸
sigma = 20           # 高斯标准差

# 生成滤波器
H = gaussian_lowpass_filter_2d(shape, sigma)

# 创建坐标网格（用于3D绘图）
rows, cols = shape
u = np.arange(rows)
v = np.arange(cols)
U, V = np.meshgrid(v, u)  # 注意：meshgrid顺序是 (x, y) → (cols, rows)

# 绘制3D图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面
surf = ax.plot_surface(U, V, H, cmap='viridis', linewidth=0, antialiased=True)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=10, label='Filter Value')

# 设置标签
ax.set_xlabel('Frequency Column (v)')
ax.set_ylabel('Frequency Row (u)')
ax.set_zlabel('Filter Magnitude')
ax.set_title(f'3D Gaussian Low-pass Filter (σ = {sigma})')

plt.tight_layout()
plt.show()