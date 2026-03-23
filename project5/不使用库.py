import numpy as np
from PIL import Image

def rgb_to_lab(rgb_img):
    rgb = rgb_img.astype(np.float64) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb2xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb.reshape(-1, 3), rgb2xyz.T).reshape(rgb.shape)
    xyz_ref = np.array([95.047, 100.0, 108.883])
    xyz = xyz / xyz_ref
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1 / 3)
    xyz[~mask] = 7.787 * xyz[~mask] + 16 / 116
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    L = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    a = np.clip(a, -128, 127)
    b = np.clip(b, -128, 127)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab_img):
    L, a, b = lab_img[..., 0], lab_img[..., 1], lab_img[..., 2]
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    def f(t):
        mask = t > 0.2068966
        t[mask] = t[mask] ** 3
        t[~mask] = (t[~mask] - 16 / 116) / 7.787
        return t
    x, y, z = f(x), f(y), f(z)
    xyz_ref = np.array([95.047, 100.0, 108.883])
    xyz = np.stack([x, y, z], axis=-1) * xyz_ref
    xyz2rgb = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    rgb = np.dot(xyz.reshape(-1, 3), xyz2rgb.T).reshape(xyz.shape)
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] = rgb[~mask] * 12.92
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def read_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)


def write_image(path, img):
    img_pil = Image.fromarray(img)
    img_pil.save(path)


def show_image(title, img):
    img_pil = Image.fromarray(img)
    img_pil.show(title)

def compute_stats(lab_img):
    l_channel = lab_img[..., 0].flatten()
    a_channel = lab_img[..., 1].flatten()
    b_channel = lab_img[..., 2].flatten()
    l_mean = np.sum(l_channel) / len(l_channel)#均值
    a_mean = np.sum(a_channel) / len(a_channel)
    b_mean = np.sum(b_channel) / len(b_channel)
    l_std = np.sqrt(np.sum((l_channel - l_mean) ** 2) / (len(l_channel) - 1))#方差
    a_std = np.sqrt(np.sum((a_channel - a_mean) ** 2) / (len(a_channel) - 1))
    b_std = np.sqrt(np.sum((b_channel - b_mean) ** 2) / (len(b_channel) - 1))
    return {
        'l': {'mean': l_mean, 'std': l_std},
        'a': {'mean': a_mean, 'std': a_std},
        'b': {'mean': b_mean, 'std': b_std}
    }


def color_transfer(source_path, target_path, output_path):
    try:
        source_img = read_image(source_path)
        target_img = read_image(target_path)
    except Exception as e:
        raise ValueError(f"图像读取失败：{e}")

    source_lab = rgb_to_lab(source_img)
    target_lab = rgb_to_lab(target_img)

    source_stats = compute_stats(source_lab)
    target_stats = compute_stats(target_lab)
    t_l = target_lab[..., 0]
    t_a = target_lab[..., 1]
    t_b = target_lab[..., 2]

    eps = 1e-6
    normalized_l = (t_l - target_stats['l']['mean']) * (source_stats['l']['std'] / (target_stats['l']['std'] + eps)) + \
                   source_stats['l']['mean']
    normalized_a = (t_a - target_stats['a']['mean']) * (source_stats['a']['std'] / (target_stats['a']['std'] + eps)) + \
                   source_stats['a']['mean']
    normalized_b = (t_b - target_stats['b']['mean']) * (source_stats['b']['std'] / (target_stats['b']['std'] + eps)) + \
                   source_stats['b']['mean']

    normalized_l = np.clip(normalized_l, 0, 100)
    normalized_a = np.clip(normalized_a, -128, 127)
    normalized_b = np.clip(normalized_b, -128, 127)
    normalized_lab = np.stack([normalized_l, normalized_a, normalized_b], axis=-1)
    result_img = lab_to_rgb(normalized_lab)
    write_image(output_path, result_img)
    print(f"颜色迁移完成，结果已保存至：{output_path}")
    show_image("Source Image", source_img)
    show_image("Target Image", target_img)
    show_image("Result Image", result_img)


if __name__ == "__main__":

    SOURCE_IMAGE_PATH = "t1.png"
    TARGET_IMAGE_PATH = "t2.png"
    OUTPUT_IMAGE_PATH = "result.jpg"

    try:
        color_transfer(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    except Exception as e:
        print(f"程序执行出错：{e}")