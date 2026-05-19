import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

EDGE_BORDER_EXCLUDE_MARGIN = 5
COLOR_DIFF_L_THRESHOLD = 20
COLOR_DIFF_AB_THRESHOLD = 20


def remove_border_pixels(mask_u8, margin=EDGE_BORDER_EXCLUDE_MARGIN):
    """
    将二值图四周 margin 像素范围内的点清零。
    """
    if mask_u8 is None:
        return mask_u8
    if margin <= 0:
        return mask_u8

    out = mask_u8.copy()
    h, w = out.shape[:2]
    m = int(max(0, margin))
    if h == 0 or w == 0:
        return out
    if m * 2 >= h or m * 2 >= w:
        out[:, :] = 0
        return out

    out[:m, :] = 0
    out[h - m:, :] = 0
    out[:, :m] = 0
    out[:, w - m:] = 0
    return out


def ensure_same_size(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2



def get_sobel_magnitude(gray):
    """
    得到连续梯度强度图，不是二值边缘图。
    用于判断弱边是否仍然存在。
    """
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)

    # 鲁棒归一化，避免极端强边把整体压低
    p99 = np.percentile(mag, 99)
    if p99 < 1e-6:
        p99 = 1.0

    mag_norm = np.clip(mag / p99 * 255.0, 0, 255).astype(np.uint8)
    return mag_norm


def recover_weak_edges(source_edge, target_edge, target_grad, source_grad,
                       search_radius=3,
                       min_grad_response=12,
                       relative_ratio=0.18):
    """
    用 source_edge 中的边作为先验，到 target_grad 中寻找弱边。

    source_edge:
        第一张图的二值边缘，比如 edge1

    target_edge:
        第二张图的二值边缘，比如 edge2

    target_grad:
        第二张图的 Sobel magnitude

    source_grad:
        第一张图的 Sobel magnitude

    返回：
        target_edge_recovered:
            补回弱边之后的 target_edge
    """

    source_bool = source_edge > 0
    target_bool = target_edge > 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * search_radius + 1, 2 * search_radius + 1)
    )

    # target_edge 附近已经有边，说明匹配到了，不需要补
    target_edge_dilated = cv2.dilate(target_edge, kernel, iterations=1) > 0
    already_matched = source_bool & target_edge_dilated

    # source 有边，但 target_edge 附近没有 Canny 边
    missing_candidate = source_bool & (~already_matched)

    # 在 target_grad 中找附近最大梯度
    target_grad_local_max = cv2.dilate(target_grad, kernel, iterations=1)

    # source 位置的梯度强度
    source_strength = source_grad.astype(np.float32)

    # 自适应弱边判断：
    # 1. target 位置附近梯度不能太低
    # 2. target 梯度至少达到 source 梯度的一定比例
    weak_edge_exists = (
        missing_candidate &
        (target_grad_local_max >= min_grad_response) &
        (target_grad_local_max.astype(np.float32) >= source_strength * relative_ratio)
    )

    recovered = target_bool.copy()
    recovered[weak_edge_exists] = True

    return (recovered.astype(np.uint8) * 255)


def tolerant_edge_diff(edge1, edge2, tolerance=2):
    """
    对二值边缘图做容忍差异比较。

    参数：
    - edge1, edge2: Canny/Sobel/TEED 得到的二值边缘图，0/255
    - tolerance: 容忍像素偏移范围，建议 1~2

    返回：
    - edge1_only_diff: edge1 中无法被 edge2 解释的边缘点
    - edge2_only_diff: edge2 中无法被 edge1 解释的边缘点
    - edge_diff: 双向差异边缘图
    """

    e1 = (edge1 > 0).astype(np.uint8)
    e2 = (edge2 > 0).astype(np.uint8)

    kernel_size = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # 把另一张图的边缘膨胀，允许 1~2 像素偏移
    e1_dilate = cv2.dilate(e1, kernel, iterations=1)
    e2_dilate = cv2.dilate(e2, kernel, iterations=1)

    # edge1 中没有被 edge2 邻域覆盖的点，才算 edge1 独有差异
    edge1_only = e1 & (~e2_dilate.astype(bool))

    # edge2 中没有被 edge1 邻域覆盖的点，才算 edge2 独有差异
    edge2_only = e2 & (~e1_dilate.astype(bool))

    edge1_only_diff = (edge1_only.astype(np.uint8)) * 255
    edge2_only_diff = (edge2_only.astype(np.uint8)) * 255
    edge_diff = cv2.bitwise_or(edge1_only_diff, edge2_only_diff)

    return edge1_only_diff, edge2_only_diff, edge_diff


def compute_color_diff_masks_directional(
    img1_bgr,
    img2_bgr,
    l_threshold=COLOR_DIFF_L_THRESHOLD,
    ab_threshold=COLOR_DIFF_AB_THRESHOLD,
):
    """
    颜色像素差异（LAB）：
    - img1_only_color：偏向 img1 的颜色变化
    - img2_only_color：偏向 img2 的颜色变化
    - color_diff：颜色变化并集

    注意：
    - 为避免遮罩/明暗变化造成整屏差异，这里仅使用 a/b 色度通道；
    - L 明度通道不参与颜色差异判定。
    """
    lab1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    d = lab1 - lab2
    ad = np.abs(d)

    color_changed = (
        (ad[:, :, 1] > int(ab_threshold)) |
        (ad[:, :, 2] > int(ab_threshold))
    )

    img1_only = color_changed & (
        (d[:, :, 1] > int(ab_threshold)) |
        (d[:, :, 2] > int(ab_threshold))
    )
    img2_only = color_changed & (
        (d[:, :, 1] < -int(ab_threshold)) |
        (d[:, :, 2] < -int(ab_threshold))
    )

    # 边界值导致的未分配点，放入 img1_only，保证并集不漏
    undecided = color_changed & (~img1_only) & (~img2_only)
    img1_only = img1_only | undecided

    img1_only_u8 = (img1_only.astype(np.uint8) * 255)
    img2_only_u8 = (img2_only.astype(np.uint8) * 255)
    color_diff_u8 = (color_changed.astype(np.uint8) * 255)
    return img1_only_u8, img2_only_u8, color_diff_u8


def percentile_stretch_to_u8(x, low_percent=1, high_percent=99):
    """
    温和拉伸到 0~255，避免极端值影响。
    """
    x = x.astype(np.float32)

    low = np.percentile(x, low_percent)
    high = np.percentile(x, high_percent)

    if high - low < 1e-6:
        return np.clip(x, 0, 255).astype(np.uint8)

    x = (x - low) / (high - low) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)

    return x


def illumination_normalize_l_channel_safe(
    img_bgr,
    blur_ksize=61,
    strength=0.4,
    use_clahe=False
):
    """
    更保守的单张图局部亮度归一化。

    重点：
    - 不强行增强阴影边界；
    - 不默认使用 CLAHE；
    - 用除法归一化削弱慢变化光照；
    - 适合后续边缘提取。

    参数：
    - blur_ksize: 光照估计尺度，建议 61~101
    - strength: 归一化强度，0~1，建议 0.4~0.7
    - use_clahe: 默认 False，避免放大阴影噪点

    返回：
    - norm_l: 归一化后的亮度图
    """

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.uint8)

    # 1. 先轻微去噪，减少阴影区域噪点被增强
    l_denoise = cv2.bilateralFilter(
        l,
        d=5,
        sigmaColor=20,
        sigmaSpace=5
    )

    l_float = l_denoise.astype(np.float32)

    # 2. 估计大尺度光照
    illumination = cv2.GaussianBlur(
        l_float,
        (blur_ksize, blur_ksize),
        0
    )

    # 3. 除法归一化，比 L - illumination 更不容易产生强 halo
    mean_illumination = np.mean(illumination)

    norm_div = l_float / (illumination + 1e-6) * mean_illumination

    # 4. 不完全使用归一化结果，和原始 L 混合，避免过度处理
    norm = (1.0 - strength) * l_float + strength * norm_div

    # 5. 温和拉伸
    norm = percentile_stretch_to_u8(
        norm,
        low_percent=1,
        high_percent=99
    )

    # 6. CLAHE 默认关闭；如果要用，也要很弱
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=1.2,
            tileGridSize=(16, 16)
        )
        norm = clahe.apply(norm)

    # 7. 最后轻微平滑，抑制噪点边缘
    norm = cv2.GaussianBlur(norm, (3, 3), 0)

    return norm

def compute_edge_maps_with_weak_recovery(img1_path, img2_path):
    """
    输入两张图路径，输出：
        img1, img2,
        edge1_recovered,
        edge2_recovered,
        img1_only_diff,
        img2_only_diff,
        edge_diff

    重点：
        不是直接比较原始 Canny edge，
        而是先用 Sobel magnitude 恢复弱边。
    """

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise ValueError(f"Cannot read image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Cannot read image: {img2_path}")

    img1, img2 = ensure_same_size(img1, img2)


    gray1 = illumination_normalize_l_channel_safe(img1)
    gray2 = illumination_normalize_l_channel_safe(img2)

    # Canny 负责强边
    edge1 = cv2.Canny(gray1, 30, 100)
    edge2 = cv2.Canny(gray2, 30, 100)

    # Sobel magnitude 负责弱边验证
    grad1 = get_sobel_magnitude(gray1)
    grad2 = get_sobel_magnitude(gray2)

    # 用 img1 的边去补 img2 漏掉的弱边
    edge2_recovered = recover_weak_edges(
        source_edge=edge1,
        target_edge=edge2,
        target_grad=grad2,
        source_grad=grad1,
        search_radius=3,
        min_grad_response=10,
        relative_ratio=0.15
    )

    # 用 img2 的边去补 img1 漏掉的弱边
    edge1_recovered = recover_weak_edges(
        source_edge=edge2,
        target_edge=edge1,
        target_grad=grad1,
        source_grad=grad2,
        search_radius=3,
        min_grad_response=10,
        relative_ratio=0.15
    )

    img1_only_diff, img2_only_diff, edge_diff = tolerant_edge_diff(
        edge1_recovered,
        edge2_recovered,
        tolerance=2
        )

    # 差异像素补充颜色差异（不仅仅是边缘）
    img1_only_color, img2_only_color, color_diff = compute_color_diff_masks_directional(
        img1,
        img2,
    )
    img1_only_diff = cv2.bitwise_or(img1_only_diff, img1_only_color)
    img2_only_diff = cv2.bitwise_or(img2_only_diff, img2_only_color)
    edge_diff = cv2.bitwise_or(edge_diff, color_diff)

    # 差异像素去边：去除图四边 margin 区域的差异点
    img1_only_diff = remove_border_pixels(img1_only_diff, EDGE_BORDER_EXCLUDE_MARGIN)
    img2_only_diff = remove_border_pixels(img2_only_diff, EDGE_BORDER_EXCLUDE_MARGIN)
    edge_diff = remove_border_pixels(edge_diff, EDGE_BORDER_EXCLUDE_MARGIN)
    return (
        img1,
        img2,
        edge1_recovered,
        edge2_recovered,
        img1_only_diff,
        img2_only_diff,
        edge_diff
    )
# ── 示例用法 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os


    IMG1_PATH = r"D:\ui2code\InteractionSequence\data\com\aestheticmedicine\screen_4.png"
    IMG2_PATH = r"D:\ui2code\InteractionSequence\data\com\aestheticmedicine\screen_25.png"

    img1, img2, edge1, edge2, img1_only_diff, img2_only_diff, edge_diff = \
        compute_edge_maps_with_weak_recovery(IMG1_PATH, IMG2_PATH)

    results = {
        "img1": img1,
        "img2": img2,
        "edge1": edge1,
        "edge2": edge2,
        "img1_only_diff": img1_only_diff,
        "img2_only_diff": img2_only_diff,
        "edge_diff": edge_diff,
    }
    panels = [
            ("img1",          img1,          "gray"),
            ("img2",          img2,          "gray"),
            ("edge1",         edge1,         "gray"),
            ("edge2",         edge2,         "gray"),
            ("img1_only_diff (红=图1独有)", img1_only_diff, "gray"),
            ("img2_only_diff (绿=图2独有)", img2_only_diff, "gray"),
            ("edge_diff",     edge_diff,     "gray"),
        ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("UI 边缘差异分析", fontsize=14)
    axes = axes.flatten()
    
    for ax, (title, arr, _) in zip(axes, panels):
        # BGR → RGB（灰度图直接显示）
        if len(arr.shape) == 3:
            display = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            ax.imshow(display)
        else:
            ax.imshow(arr, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    axes[-1].axis("off")   # 第8格留空
    plt.tight_layout()
    plt.show()
