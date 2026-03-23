import cv2
import numpy as np
import sys

def gaussian_lowpass_filter(shape, sigma):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)
    D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
    H = np.exp(- (D ** 2) / (2 * (sigma ** 2)))
    return H

def apply_frequency_filter(img, filter_type='low', sigma=5):

    #傅里叶变换
    f = np.fft.fft2(img.astype(np.float32))
    f_shift = np.fft.fftshift(f)

    H_low = gaussian_lowpass_filter(img.shape, sigma)
    if filter_type == 'low':
        H = H_low
    elif filter_type == 'high':
        H = 1 - H_low
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    #频域滤波
    filtered_f = f_shift * H

    #逆变换回空间域
    f_ishift = np.fft.ifftshift(filtered_f)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    return img_back, H

def blend_images_freq(low_freq_img_path, high_freq_img_path, sigma_low=5, sigma_high=5):
    low_img = cv2.imread(low_freq_img_path, cv2.IMREAD_GRAYSCALE)
    high_img = cv2.imread(high_freq_img_path, cv2.IMREAD_GRAYSCALE)

    if low_img is None or high_img is None:
        raise FileNotFoundError("请检查图像路径是否正确！")

    # 统一尺寸
    h = min(low_img.shape[0], high_img.shape[0])
    w = min(low_img.shape[1], high_img.shape[1])
    low_img = cv2.resize(low_img, (w, h), interpolation=cv2.INTER_AREA)
    high_img = cv2.resize(high_img, (w, h), interpolation=cv2.INTER_AREA)

    # === 低通滤波（频域）===
    low_pass, _ = apply_frequency_filter(low_img, filter_type='low', sigma=sigma_low)
    # === 高通滤波（频域）===
    high_pass, _ = apply_frequency_filter(high_img, filter_type='high', sigma=sigma_high)

    high_pass_vis = np.clip(high_pass + 128, 0, 255).astype(np.uint8)

    hybrid = low_pass + high_pass
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

    low_pass_uint8 = np.clip(low_pass, 0, 255).astype(np.uint8)

    return hybrid, low_pass_uint8, high_pass_vis



if __name__ == "__main__":
    low_path = 'p3.png'   #低频图像（如爱因斯坦）
    high_path = 'p4.png'  #高频图像（如玛丽莲）

    sigma_low = 30#数值越小越模糊
    sigma_high = 40#越大越平滑

    try:
        hybrid, low_pass, high_pass_vis = blend_images_freq(
            low_path, high_path,
            sigma_low=sigma_low,
            sigma_high=sigma_high
        )

        # 显示
        cv2.imshow('Low-pass (Freq Domain)', low_pass)
        cv2.imshow('High-pass (Freq Domain)', high_pass_vis)
        cv2.imshow('Hybrid Image (Freq Domain)', hybrid)

        cv2.imwrite('hybrid_freq.png', hybrid)
        cv2.imwrite('low_pass_freq.png', low_pass)
        cv2.imwrite('high_pass_freq_vis.png', high_pass_vis)

        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)