import cv2
import numpy as np
from pathlib import Path
from collections import deque

# =========================
# 修改这里的图片路径
# =========================
IMG1_PATH = r"D:\ui2code\InteractionSequence\data\com/aestheticmedicine\screen_11.png"
IMG2_PATH = r"D:\ui2code\InteractionSequence\data\com/aestheticmedicine\screen_13.png"

SHOW_RESULT = True
DISPLAY_WINDOW_SCALE = 1.0
DIFF_DISPLAY_DILATE_ITER = 1
DIFF_CANDIDATE_THICKEN_ITER = 2


# =========================
# Closing 参数
# =========================
CLOSING_KERNEL_SIZE = 5
CLOSING_ITERATIONS = 1

# 候选边与闭合框约束
MIN_CANDIDATE_COMPONENT_LENGTH = 300
OPEN_FRAME_MIN_SIDE = 50
CLOSED_FRAME_MIN_AREA_ABS = 3000
CLOSED_FRAME_MIN_AREA_RATIO = 0.005
OUTSIDE_MAX_ALLOWED_COMPONENT_AREA = 50
HORIZONTAL_EDGE_EXCLUDE_TOP_BOTTOM_PX = 100
VERTICAL_EDGE_EXCLUDE_LEFT_RIGHT_PX = 50

#差异参数
EDGE_BORDER_EXCLUDE_MARGIN = 5
COLOR_DIFF_AB_THRESHOLD = 20


# =========================
# 基础工具
# =========================
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

def compute_edge_maps(img1_path, img2_path):
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


def illumination_normalize_l_channel(img_bgr, blur_ksize=51):
    """
    对单张图做局部光照归一化。

    作用：
    - 削弱阴影、遮罩、渐变光照；
    - 保留局部结构边缘；
    - 不强行把两张图调成相同亮度。

    返回：
    - norm_l: 归一化后的亮度图，用于边缘提取
    """

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    # 大尺度模糊，估计局部光照背景
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    illumination = cv2.GaussianBlur(
        l,
        (blur_ksize, blur_ksize),
        0
    )

    # 用 L - illumination 消除慢变化亮度
    norm = l - illumination + 128.0

    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # 轻微增强局部对比度
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    norm = clahe.apply(norm)

    return norm


def compute_edge_maps_1(img1_path, img2_path):
    """
    1. 读取图像
    2. 转灰度图并做光照归一化
    3. CLAHE 增强 + Canny
    4. 得到：
       - edge1
       - edge2
       - img1_only_diff：只属于 img1 的 edge 差异点
       - img2_only_diff：只属于 img2 的 edge 差异点
       - edge_diff：整体结构差异图
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise ValueError(f"Cannot read image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Cannot read image: {img2_path}")

    img1, img2 = ensure_same_size(img1, img2)

    
    # img1_norm, img2_norm = make_two_images_same_brightness(img1, img2)
    # gray1 = cv2.cvtColor(img1_norm, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2_norm, cv2.COLOR_BGR2GRAY)

    # gray1_raw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2_raw = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 先做光照归一化，减弱亮度不一致导致的伪差异。
    # gray1 = normalize_illumination_gray(gray1_raw)
    # gray2 = normalize_illumination_gray(gray2_raw)
   
    # edge1 = cv2.Canny(gray1, 50, 150)
    # edge2 = cv2.Canny(gray2, 50, 150)

    # edge1 = cv2.Canny(gray1, 20, 80)
    # edge2 = cv2.Canny(gray2, 20, 80)

    # gray1_blur = cv2.GaussianBlur(gray1, (3, 3), 0)
    # gray2_blur = cv2.GaussianBlur(gray2, (3, 3), 0)

    # edge1 = cv2.Canny(gray1_blur,20, 80)
    # edge2 = cv2.Canny(gray2_blur, 20, 80)

    gray1 = illumination_normalize_l_channel(img1)
    gray2 = illumination_normalize_l_channel(img2)
    gray1_blur = cv2.GaussianBlur(gray1, (3, 3), 0)
    gray2_blur = cv2.GaussianBlur(gray2, (3, 3), 0)

    edge1 = cv2.Canny(gray1_blur,20, 80)
    edge2 = cv2.Canny(gray2_blur, 20, 80)

    # clahe = cv2.createCLAHE(
    # clipLimit=2.0,
    # tileGridSize=(8, 8)
    # )

    # gray1_enhanced = clahe.apply(gray1)
    # gray2_enhanced = clahe.apply(gray2)
    # edge1 = cv2.Canny(gray1_enhanced, 20, 80)
    # edge2 = cv2.Canny(gray2_enhanced, 20, 80)

    # e1 = edge1 > 0
    # e2 = edge2 > 0


    # img1_only_diff = (e1 & (~e2)).astype(np.uint8) * 255
    # img2_only_diff = (e2 & (~e1)).astype(np.uint8) * 255
    

    # edge_diff = cv2.absdiff(edge1, edge2)

    # # 差异像素去边：去除图四边 margin 区域的差异点
    # img1_only_diff = remove_border_pixels(img1_only_diff, EDGE_BORDER_EXCLUDE_MARGIN)
    # img2_only_diff = remove_border_pixels(img2_only_diff, EDGE_BORDER_EXCLUDE_MARGIN)
    # edge_diff = remove_border_pixels(edge_diff, EDGE_BORDER_EXCLUDE_MARGIN)
    img1_only_diff, img2_only_diff, edge_diff = tolerant_edge_diff(
    edge1,
    edge2,
    tolerance=2
    )

    return img1, img2, edge1, edge2, img1_only_diff, img2_only_diff, edge_diff

def compute_raw_diff_mask(img1, img2):
    """
    raw 像素差异，仅作为观察，不参与最终判断。
    """
    diff = cv2.absdiff(img1, img2)
    raw_mask = np.any(diff != 0, axis=2).astype(np.uint8) * 255
    return raw_mask


def close_diff_map(diff_map, kernel_size=CLOSING_KERNEL_SIZE, iterations=CLOSING_ITERATIONS):
    """
    对差异图做闭运算，用于修复候选边/框的小断裂。

    注意：
    - closed_map 只用于生成候选；
    - 最终验证仍然使用原始 edge_diff。
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))

    h = cv2.morphologyEx(diff_map, cv2.MORPH_CLOSE, h_kernel, 1)
    v = cv2.morphologyEx(diff_map, cv2.MORPH_CLOSE, v_kernel, 1)

    out = cv2.bitwise_or(h, v)
    return out


def project_diff_components_to_original_components(diff_seed_map, original_edge_map):
    """
    将“差异连通区”投影到“原图连通区”上。

    规则：
    1) 先对 diff_seed_map 做连通域；
    2) 每个差异连通域在 original_edge_map 连通域标签图上找其落点；
    3) 若多个差异连通域命中同一个原图连通域，则自然并到同一候选连通域；
    4) 输出的是原图连通域组成的候选图，不再依赖差异图闭合是否完整。
    """
    diff_bool = diff_seed_map > 0
    edge_bool = original_edge_map > 0

    projected = np.zeros_like(diff_seed_map, dtype=np.uint8)
    if not np.any(diff_bool) or not np.any(edge_bool):
        return projected

    # 原图连通域标签
    edge_u8 = edge_bool.astype(np.uint8)
    edge_num_labels, edge_labels, _, _ = cv2.connectedComponentsWithStats(edge_u8, connectivity=8)
    if edge_num_labels <= 1:
        return projected

    # 差异连通域标签
    diff_u8 = diff_bool.astype(np.uint8)
    diff_num_labels, diff_labels, _, _ = cv2.connectedComponentsWithStats(diff_u8, connectivity=8)
    if diff_num_labels <= 1:
        return projected

    label_seed_masks = {}

    # 逐个差异连通域映射到原图连通域，并累计每个原图连通域上的差异种子
    for diff_label in range(1, diff_num_labels):
        diff_component = (diff_labels == diff_label)
        if not np.any(diff_component):
            continue

        overlap_edge_labels = edge_labels[diff_component]
        overlap_edge_labels = overlap_edge_labels[overlap_edge_labels > 0]
        if overlap_edge_labels.size == 0:
            continue

        unique_labels = np.unique(overlap_edge_labels)
        for lb in unique_labels:
            lb = int(lb)
            edge_component_mask = (edge_labels == lb)
            seed_on_component = diff_component & edge_component_mask
            if not np.any(seed_on_component):
                continue

            if lb not in label_seed_masks:
                label_seed_masks[lb] = seed_on_component.copy()
            else:
                label_seed_masks[lb] |= seed_on_component

    if not label_seed_masks:
        return projected

    # 对每个命中的原图连通域提取最小连接分支：
    # - 不做闭运算；
    # - 若同一区域内存在多个差异子连通区，提取能把这些子区连在一起的最小连通分支。
    for lb, seed_mask in label_seed_masks.items():
        component_mask = (edge_labels == int(lb))
        component_area = int(np.sum(component_mask))
        seed_area = int(np.sum(seed_mask))
        if component_area <= 0 or seed_area <= 0:
            continue

        min_branch = find_min_connecting_branch(component_mask, seed_mask)
        projected[min_branch] = 255

    return projected


def find_min_connecting_branch(component_mask, seed_mask):
    """
    在同一个原图连通区内提取最小连接分支：
    1) 先把 seed 拆成多个差异子连通区；
    2) 如果只有一个子区，直接返回它；
    3) 如果有多个子区，从面积最小的子区开始，迭代连接最近的未连接子区；
    4) 输出包含全部 seed 子区的最小连通分支（近似最小连接树）。
    """
    component_bool = component_mask > 0
    seed_bool = (seed_mask > 0) & component_bool
    if not np.any(component_bool) or not np.any(seed_bool):
        return np.zeros_like(component_bool, dtype=bool)

    seed_u8 = seed_bool.astype(np.uint8)
    num_labels, seed_labels, stats, _ = cv2.connectedComponentsWithStats(seed_u8, connectivity=8)
    if num_labels <= 1:
        return seed_bool

    comp_infos = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > 0:
            comp_infos.append((int(label_id), area))
    if not comp_infos:
        return seed_bool

    # 从最小差异子连通区作为起点
    comp_infos.sort(key=lambda x: x[1])
    start_label = int(comp_infos[0][0])
    tree = (seed_labels == start_label)

    connected_labels = {start_label}
    remaining_labels = {lb for lb, _ in comp_infos if lb != start_label}

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while remaining_labels:
        remaining_arr = np.array(sorted(remaining_labels), dtype=seed_labels.dtype)
        remaining_mask = np.isin(seed_labels, remaining_arr)

        found, hit_y, hit_x, hit_label, parent = bfs_find_nearest_target_on_component(
            component_bool=component_bool,
            tree_bool=tree,
            target_mask=remaining_mask,
            seed_labels=seed_labels,
            neighbors=neighbors,
        )
        if not found:
            break

        # 回溯最短路径
        cy, cx = int(hit_y), int(hit_x)
        while cy >= 0 and cx >= 0:
            tree[cy, cx] = True
            py, px = parent[cy, cx]
            if py < 0 or px < 0:
                break
            cy, cx = int(py), int(px)

        # 把命中的差异子连通区并入 tree
        tree |= (seed_labels == int(hit_label))
        connected_labels.add(int(hit_label))
        if int(hit_label) in remaining_labels:
            remaining_labels.remove(int(hit_label))

    for lb in connected_labels:
        tree |= (seed_labels == int(lb))

    return tree


def bfs_find_nearest_target_on_component(component_bool, tree_bool, target_mask, seed_labels, neighbors):
    """
    从 tree 的多源像素出发，在 component_bool 内 BFS，找最近 target。
    """
    h, w = component_bool.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    parent = np.full((h, w, 2), -1, dtype=np.int32)
    q = deque()

    ys, xs = np.where(tree_bool)
    for y, x in zip(ys, xs):
        yy = int(y)
        xx = int(x)
        visited[yy, xx] = True
        q.append((yy, xx))

    while q:
        y, x = q.popleft()
        if target_mask[y, x]:
            return True, y, x, int(seed_labels[y, x]), parent

        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx]:
                continue
            if not component_bool[ny, nx]:
                continue
            visited[ny, nx] = True
            parent[ny, nx, 0] = y
            parent[ny, nx, 1] = x
            q.append((ny, nx))

    return False, -1, -1, -1, parent


# =========================
# 几何与拓扑工具
# =========================
def contour_basic_info(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    else:
        approx = contour

    x, y, w, h = cv2.boundingRect(contour)

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "approx": approx,
        "approx_points": int(len(approx)),
        "rect": [int(x), int(y), int(w), int(h)],
    }


def component_topology_stats(component_mask):
    """
    统计连通区域的端点和分叉点。
    """
    binary = component_mask > 0
    h, w = binary.shape

    endpoints = 0
    branchpoints = 0

    ys, xs = np.where(binary)

    for y, x in zip(ys, xs):
        y1 = max(0, y - 1)
        y2 = min(h, y + 2)
        x1 = max(0, x - 1)
        x2 = min(w, x + 2)

        neighborhood = binary[y1:y2, x1:x2]
        degree = int(np.sum(neighborhood)) - 1

        if degree == 1:
            endpoints += 1
        elif degree >= 3:
            branchpoints += 1

    return {
        "endpoint_count": endpoints,
        "branchpoint_count": branchpoints,
    }


def rect_to_filled_mask(rect, shape_hw):
    h, w = shape_hw
    x, y, rw, rh = rect
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w, int(x + rw))
    y2 = min(h, int(y + rh))

    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def min_frame_area_by_image(shape_hw):
    h, w = shape_hw
    return max(
        CLOSED_FRAME_MIN_AREA_ABS,
        h * w * CLOSED_FRAME_MIN_AREA_RATIO,
    )


def region_mask_from_component_rect(rect, shape_hw, side):
    """
    根据非闭合连通区域外接框生成候选解释区域。

    side:
        below / above / left / right
    """
    h, w = shape_hw
    x, y, rw, rh = rect

    mask = np.zeros((h, w), dtype=bool)

    if side == "below":
        mask[y:h, :] = True
    elif side == "above":
        mask[0:y + rh, :] = True
    elif side == "right":
        mask[:, x:w] = True
    elif side == "left":
        mask[:, 0:x + rw] = True
    else:
        raise ValueError(f"Unknown side: {side}")

    return mask


# =========================
# 连通区域提取（候选图与原始 seed 必须有交集）
# =========================
def iter_connected_components(candidate_map, original_seed_map):
    candidate_bool = candidate_map > 0
    seed_bool = original_seed_map > 0

    if not np.any(candidate_bool):
        return

    mask_u8 = candidate_bool.astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    for label_id in range(1, num_labels):
        component_bool = labels == label_id
        seed_inside = component_bool & seed_bool

        if not np.any(seed_inside):
            continue

        component_mask = component_bool.astype(np.uint8) * 255

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=lambda c: len(c))

        yield (
            int(label_id),
            component_mask,
            contour,
            seed_inside.astype(np.uint8) * 255,
        )


# =========================
# 候选分类
# =========================
def is_closed_frame_candidate(contour, component_mask):
    """
    闭合框候选约束：
    - endpoint_count == 0
    - area >= max(3000, 图像面积 * 0.005)
    - perimeter > 0
    - approx_points >= 4
    """
    topo = component_topology_stats(component_mask)
    if topo["endpoint_count"] != 0:
        return False

    info = contour_basic_info(contour)
    if info["area"] <= 0:
        return False

    h, w = component_mask.shape[:2]
    _, _, rw, rh = info["rect"]
    rect_area = rw * rh
    if rect_area < min_frame_area_by_image((h, w)):
        return False

    if info["perimeter"] <= 0:
        return False

    if info["approx_points"] < 4:
        return False

    return True


def classify_non_closed_component(contour):
    """
    非闭合候选分类：
    - popup_edge：细长边
    - popup_open_frame：非细长开口框
    """
    info = contour_basic_info(contour)
    _, _, w, h = info["rect"]

    if w <= 0 or h <= 0:
        return "invalid"

    # 宽和高都超过阈值时，优先判为开口框
    if w > OPEN_FRAME_MIN_SIDE and h > OPEN_FRAME_MIN_SIDE:
        return "popup_open_frame"

    return "popup_edge"


def side_to_popup_type(side):
    if side == "below":
        return "top_edge_of_bottom_sheet"
    if side == "above":
        return "bottom_edge_of_top_sheet"
    if side == "right":
        return "left_edge_of_right_drawer"
    if side == "left":
        return "right_edge_of_left_drawer"
    return "unknown"




# =========================
# 候选生成
# =========================
def find_popup_candidates(candidate_map, original_seed_map, source_name):
    """
    从 closing 后的差异图中生成候选。

    - candidate_map：用于修复断裂后提候选
    - original_seed_map：用于保证候选来自原始差异像素
    """
    h, w = candidate_map.shape[:2]
    candidates = []

    for label_id, component_mask, contour, seed_mask in iter_connected_components(
        candidate_map,
        original_seed_map,
    ):
        info = contour_basic_info(contour)
        topo = component_topology_stats(component_mask)
        _, _, rw, rh = info["rect"]

        source_mask = component_mask > 0
        seed_bool = seed_mask > 0

        source_pixel_count = int(np.sum(source_mask))
        seed_pixel_count = int(np.sum(seed_bool))
        closing_added_count = source_pixel_count - seed_pixel_count

        # 候选集合统一约束：基于原始连通候选本身（source_mask）的长度阈值
        if source_pixel_count < MIN_CANDIDATE_COMPONENT_LENGTH:
            continue

        # 1) 闭合框候选（使用外接矩形作为验证区域）
        if is_closed_frame_candidate(contour, component_mask):
            region_mask = rect_to_filled_mask(info["rect"], (h, w))
            candidates.append({
                "kind": "popup_border",
                "subtype": "closed_or_rounded_rect_border",
                "source": source_name,
                "component_label": int(label_id),
                "contour": contour,
                "rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "endpoint_count": topo["endpoint_count"],
                "branchpoint_count": topo["branchpoint_count"],
                "source_mask": source_mask,
                "source_pixel_count": source_pixel_count,
                "seed_mask": seed_bool,
                "seed_pixel_count": seed_pixel_count,
                "closing_added_count": closing_added_count,
                "orientation": None,
                "side": None,
                "region_mask": region_mask,
                "region_key": (
                    "closed_border",
                    source_name,
                    int(label_id),
                    tuple(info["rect"]),
                ),
            })
            continue

        # 2) 非闭合：分为开口框 / 候选边
        non_closed_kind = classify_non_closed_component(contour)
        if non_closed_kind == "invalid":
            continue

        rect_area = rw * rh

        # 开口框与闭口框一致：用外接矩形进行面积阈值和后续验证
        if non_closed_kind == "popup_open_frame":
            if rect_area < min_frame_area_by_image((h, w)):
                continue

            region_mask = rect_to_filled_mask(info["rect"], (h, w))
            candidates.append({
                "kind": "popup_open_frame",
                "subtype": "open_frame_bbox",
                "source": source_name,
                "component_label": int(label_id),
                "contour": contour,
                "rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "endpoint_count": topo["endpoint_count"],
                "branchpoint_count": topo["branchpoint_count"],
                "source_mask": source_mask,
                "source_pixel_count": source_pixel_count,
                "seed_mask": seed_bool,
                "seed_pixel_count": seed_pixel_count,
                "closing_added_count": closing_added_count,
                "orientation": None,
                "side": None,
                "region_mask": region_mask,
                "region_key": (
                    "open_frame_bbox",
                    source_name,
                    int(label_id),
                    tuple(info["rect"]),
                ),
            })
            continue

        # 候选边约束：基于原始连通候选本身（source_mask）的长度阈值
        if non_closed_kind != "popup_edge":
            continue

        if rw >= rh:
            orientation = "horizontal"
            edge_length = rw
            min_edge_length = w * 0.5
            side_defs = ["below", "above"]
        else:
            orientation = "vertical"
            edge_length = rh
            min_edge_length = h * 0.5
            side_defs = ["right", "left"]


        # 候选边长度约束：
        # - horizontal: 边长度至少达到图像宽度的一半
        # - vertical:   边长度至少达到图像高度的一半
        if edge_length < min_edge_length:
            continue

        # 候选边位置约束：
        # - 横向边：中心点不能落在距离顶部/底部 100 像素范围内
        # - 竖向边：中心点不能落在距离左侧/右侧 50 像素范围内
        x, y, _, _ = info["rect"]
        center_x = x + rw / 2.0
        center_y = y + rh / 2.0

        if orientation == "horizontal":
            if (
                center_y < HORIZONTAL_EDGE_EXCLUDE_TOP_BOTTOM_PX
                or center_y > (h - HORIZONTAL_EDGE_EXCLUDE_TOP_BOTTOM_PX)
            ):
                continue
        else:
            if (
                center_x < VERTICAL_EDGE_EXCLUDE_LEFT_RIGHT_PX
                or center_x > (w - VERTICAL_EDGE_EXCLUDE_LEFT_RIGHT_PX)
            ):
                continue

        for side in side_defs:
            region_mask = region_mask_from_component_rect(info["rect"], (h, w), side)
            popup_type = side_to_popup_type(side)

            candidates.append({
                "kind": "popup_edge",
                "subtype": popup_type,
                "source": source_name,
                "component_label": int(label_id),
                "contour": contour,
                "rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "endpoint_count": topo["endpoint_count"],
                "branchpoint_count": topo["branchpoint_count"],
                "source_mask": source_mask,
                "source_pixel_count": source_pixel_count,
                "seed_mask": seed_bool,
                "seed_pixel_count": seed_pixel_count,
                "closing_added_count": closing_added_count,
                "orientation": orientation,
                "side": side,
                "region_mask": region_mask,
                "region_key": (
                    "popup_edge",
                    source_name,
                    int(label_id),
                    orientation,
                    side,
                    tuple(info["rect"]),
                ),
            })

    return candidates


def collect_all_connected_regions(candidate_map, original_seed_map, source_name):
    """
    收集所有提取到的连通区域（仅要求与原始 seed 有交集，不做候选筛选）。
    用于可视化：红色显示全部连通区域。
    """
    regions = []

    for label_id, component_mask, contour, seed_mask in iter_connected_components(
        candidate_map,
        original_seed_map,
    ):
        info = contour_basic_info(contour)
        source_mask = component_mask > 0
        seed_bool = seed_mask > 0

        regions.append({
            "kind": "all_connected_region",
            "source": source_name,
            "component_label": int(label_id),
            "rect": info["rect"],
            "source_mask": source_mask,
            "seed_mask": seed_bool,
            "source_pixel_count": int(np.sum(source_mask)),
            "seed_pixel_count": int(np.sum(seed_bool)),
        })

    return regions


# =========================
# 候选验证（在原始 edge_diff 上）
# =========================
def analyze_outside_discrete_noise(edge_diff, region_mask):
    """
    候选验证（在原始 edge_diff 上）：
    - 区域外连通域 area <= 100 允许；
    - 区域外连通域 area > 100 视为不允许结构。
    """
    diff_bool = edge_diff > 0
    outside_bool = diff_bool & (~region_mask)
    outside_mask = outside_bool.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(outside_mask, connectivity=8)

    small_noise_components = []
    large_connected_components = []
    all_outside_components = []

    max_area = 0
    max_label = 0
    max_rect = None

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        item = {
            "label": int(label),
            "rect": [x, y, w, h],
            "area": area,
        }
        all_outside_components.append(item)

        if area > max_area:
            max_area = int(area)
            max_label = int(label)
            max_rect = [x, y, w, h]

        if area <= OUTSIDE_MAX_ALLOWED_COMPONENT_AREA:
            small_noise_components.append(item)
        else:
            large_connected_components.append(item)

    max_component_mask = (labels == max_label) if max_label > 0 else np.zeros_like(outside_mask, dtype=bool)

    return {
        "outside_diff_count": int(np.sum(outside_bool)),
        "outside_small_noise_component_count": len(small_noise_components),
        "outside_large_connected_component_count": len(large_connected_components),
        "outside_large_connected_components": large_connected_components,
        "max_outside_component_area": int(max_area),
        "max_outside_component_rect": max_rect,
        "max_outside_component_mask": max_component_mask,
        "outside_noise_area_le_100_only": len(large_connected_components) == 0,
        # backward-compatible aliases
        "outside_discrete_point_count": len(small_noise_components),
        "outside_connected_component_count": len(large_connected_components),
        "outside_connected_components": large_connected_components,
        "only_discrete_pixels_outside": len(large_connected_components) == 0,
    }


def validate_candidate(edge_diff, candidate):
    region_mask = candidate["region_mask"]

    outside_result = analyze_outside_discrete_noise(edge_diff, region_mask)

    diff_bool = edge_diff > 0
    inside_diff_count = int(np.sum(diff_bool & region_mask))
    total_diff_count = int(np.sum(diff_bool))

    return {
        **candidate,
        "valid": outside_result["outside_noise_area_le_100_only"],
        "total_diff_count": total_diff_count,
        "inside_diff_count": inside_diff_count,
        **outside_result,
    }


def deduplicate_candidates(candidates):
    """
    同一 region_key 下保留 seed_pixel_count 最大的候选。
    """
    grouped = {}
    for c in candidates:
        key = c["region_key"]
        if key not in grouped:
            grouped[key] = c
            continue
        old = grouped[key]
        if c.get("seed_pixel_count", 0) > old.get("seed_pixel_count", 0):
            grouped[key] = c
    return list(grouped.values())


# =========================
# 主判断函数
# =========================
def judge_popup_relation(img1_path, img2_path):
    """
    流程：

    1. 分别得到 edge1、edge2；
    2. 得到 img1_only_diff、img2_only_diff、edge_diff；
    3. 将两图各自的差异连通区映射到各自原图连通区；
    4. 若不同差异连通区落在同一原图连通区，则合并为同一候选；
    5. 从映射后的原图连通区域中生成候选边/框；
    6. 候选必须和原始 seed（差异像素）有交集；
    7. 在原始整体 edge_diff 上验证候选；
    8. 候选区域外只允许离散像素点；
    9. 有效候选必须唯一。
    """
    img1, img2, edge1, edge2, img1_only_diff, img2_only_diff, edge_diff = compute_edge_maps(
        img1_path,
        img2_path,
    )

    raw_diff = compute_raw_diff_mask(img1, img2)

    # 兼容保留：用于展示/排查的差异闭合图
    img1_closed_diff = close_diff_map(img1_only_diff)
    img2_closed_diff = close_diff_map(img2_only_diff)

    # 核心变更：将差异连通区投影到各自原图连通区
    img1_projected_components = project_diff_components_to_original_components(
        img1_only_diff,
        edge1,
    )
    img2_projected_components = project_diff_components_to_original_components(
        img2_only_diff,
        edge2,
    )


    img1_all_components = collect_all_connected_regions(
        img1_projected_components,
        img1_only_diff,
        "img1_diff_projected_to_edge1",
    )
    img2_all_components = collect_all_connected_regions(
        img2_projected_components,
        img2_only_diff,
        "img2_diff_projected_to_edge2",
    )


    if not np.any(edge_diff > 0):
        return {
            "same_page": False,
            "relation": "not_popup_relation",
            "reason": "no_edge_difference",
            "popup_type": "none",
            "candidate": None,
            "valid_candidate_count": 0,
            "img1": img1,
            "img2": img2,
            "edge1": edge1,
            "edge2": edge2,
            "img1_only_diff": img1_only_diff,
            "img2_only_diff": img2_only_diff,
            "img1_closed_diff": img1_closed_diff,
            "img2_closed_diff": img2_closed_diff,
            "img1_projected_components": img1_projected_components,
            "img2_projected_components": img2_projected_components,
            "edge_diff": edge_diff,
            "raw_diff": raw_diff,
            "selected_candidate_full": None,
            "valid_candidates": [],
            "passed_candidates": [],
            "evaluated_candidates": [],
            "img1_all_components": img1_all_components,
            "img2_all_components": img2_all_components,
            "img1_candidates": [],
            "img2_candidates": [],
            "all_candidates": [],
        }

    img1_candidates = find_popup_candidates(
        img1_projected_components,
        img1_only_diff,
        "img1_diff_projected_to_edge1",
    )
    img2_candidates = find_popup_candidates(
        img2_projected_components,
        img2_only_diff,
        "img2_diff_projected_to_edge2",
    )

    all_candidates = []
    all_candidates.extend(img1_candidates)
    all_candidates.extend(img2_candidates)

    evaluated = [validate_candidate(edge_diff, c) for c in all_candidates]
    passed_candidates = [c for c in evaluated if c["valid"]]
    valid_candidates = deduplicate_candidates(passed_candidates)

    if len(valid_candidates) == 1:
        best = valid_candidates[0]
        return {
            "same_page": True,
            "relation": "popup_relation",
            "reason": "unique_popup_candidate_from_projected_original_components_and_verified_on_original_edge_diff",
            "popup_type": best["subtype"],
            "candidate": {
                "kind": best["kind"],
                "subtype": best["subtype"],
                "source": best["source"],
                "component_label": best["component_label"],
                "rect": best["rect"],
                "orientation": best.get("orientation"),
                "side": best.get("side"),
                "source_pixel_count": best["source_pixel_count"],
                "seed_pixel_count": best["seed_pixel_count"],
                "closing_added_count": best["closing_added_count"],
                "total_diff_count": best["total_diff_count"],
                "inside_diff_count": best["inside_diff_count"],
                "outside_diff_count": best["outside_diff_count"],
                "endpoint_count": best["endpoint_count"],
                "branchpoint_count": best["branchpoint_count"],
                "outside_small_noise_component_count": best["outside_small_noise_component_count"],
                "outside_large_connected_component_count": best["outside_large_connected_component_count"],
                "outside_discrete_point_count": best["outside_discrete_point_count"],
                "outside_connected_component_count": best["outside_connected_component_count"],
                "max_outside_component_area": int(best.get("max_outside_component_area", 0)),
            },
            "valid_candidate_count": len(valid_candidates),
            "img1": img1,
            "img2": img2,
            "edge1": edge1,
            "edge2": edge2,
            "img1_only_diff": img1_only_diff,
            "img2_only_diff": img2_only_diff,
            "img1_closed_diff": img1_closed_diff,
            "img2_closed_diff": img2_closed_diff,
            "img1_projected_components": img1_projected_components,
            "img2_projected_components": img2_projected_components,
            "edge_diff": edge_diff,
            "raw_diff": raw_diff,
            "selected_candidate_full": best,
            "valid_candidates": valid_candidates,
            "passed_candidates": passed_candidates,
            "evaluated_candidates": evaluated,
            "img1_all_components": img1_all_components,
            "img2_all_components": img2_all_components,
            "img1_candidates": img1_candidates,
            "img2_candidates": img2_candidates,
            "all_candidates": all_candidates,
        }

    if len(valid_candidates) == 0:
        reason = "no_popup_candidate_can_explain_original_edge_diff"
    else:
        reason = "multiple_popup_candidates_can_explain_original_edge_diff"

    return {
        "same_page": False,
        "relation": "not_popup_relation",
        "reason": reason,
        "popup_type": "none",
        "candidate": None,
        "valid_candidate_count": len(valid_candidates),
        "img1": img1,
        "img2": img2,
        "edge1": edge1,
        "edge2": edge2,
        "img1_only_diff": img1_only_diff,
        "img2_only_diff": img2_only_diff,
        "img1_closed_diff": img1_closed_diff,
        "img2_closed_diff": img2_closed_diff,
        "img1_projected_components": img1_projected_components,
        "img2_projected_components": img2_projected_components,
        "edge_diff": edge_diff,
        "raw_diff": raw_diff,
        "selected_candidate_full": None,
        "valid_candidates": valid_candidates,
        "passed_candidates": passed_candidates,
        "evaluated_candidates": evaluated,
        "img1_all_components": img1_all_components,
        "img2_all_components": img2_all_components,
        "img1_candidates": img1_candidates,
        "img2_candidates": img2_candidates,
        "all_candidates": all_candidates,
    }


# =========================
# 可视化：只显示三张图
# =========================
def draw_one_candidate(img, candidate, color=(0, 0, 255), thickness=2):
    out = img.copy()
    if candidate is None:
        return out

    source_mask = candidate.get("source_mask")
    if source_mask is None:
        return out

    mask = source_mask > 0
    if out.ndim == 2:
        out[mask] = 255
    else:
        out[mask] = color

    return out


def draw_candidates_by_kind(img, candidates, thickness=2):
    """
    候选可视化区分规则（不仅靠颜色）：
    - 边（popup_edge）：黄色 + 粗轮廓 + 中心圆点 + EDGE 标签
    - 框（popup_border）：蓝色 + 双层矩形 + BORDER 标签
    - 开口框（popup_open_frame）：紫色 + 矩形 + 对角线X + OPEN 标签
    """
    out = img.copy()
    for c in candidates:
        kind = c.get("kind", "")
        rect = c.get("rect", [0, 0, 0, 0])
        x, y, rw, rh = [int(v) for v in rect]
        x2 = x + rw
        y2 = y + rh
        source_mask = c.get("source_mask")

        if kind == "popup_edge":
            color = (0, 255, 255)  # yellow (BGR)
            label = "EDGE"
        elif kind == "popup_border":
            color = (255, 0, 0)    # blue (BGR)
            label = "BORDER"
        elif kind == "popup_open_frame":
            color = (255, 0, 255)  # purple (BGR)
            label = "OPEN"
        else:
            color = (0, 255, 255)
            label = "CAND"

        # 1) 半透明填充，让类别区域更容易一眼看到
        if source_mask is not None:
            mask = source_mask > 0
            if np.any(mask):
                alpha = 0.45
                overlay = out.copy()
                overlay[mask] = color
                out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

                # 2) 轮廓线
                mask_u8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color, max(2, thickness + 1))

        # 3) 类别专属几何标记
        if kind == "popup_border":
            cv2.rectangle(out, (x, y), (x2, y2), color, max(2, thickness + 1))
            cv2.rectangle(out, (x + 3, y + 3), (x2 - 3, y2 - 3), color, max(1, thickness))
        elif kind == "popup_open_frame":
            cv2.rectangle(out, (x, y), (x2, y2), color, max(2, thickness + 1))
            cv2.line(out, (x, y), (x2, y2), color, max(2, thickness))
            cv2.line(out, (x2, y), (x, y2), color, max(2, thickness))
        elif kind == "popup_edge":
            cx = x + max(0, rw // 2)
            cy = y + max(0, rh // 2)
            cv2.circle(out, (cx, cy), 5, color, -1)

        # 4) 标签
        text_x = max(0, x)
        text_y = max(18, y - 6)
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_non_candidate_regions(img, all_components, candidates, color=(0, 0, 255), thickness=2):
    """
    在图上标注“非候选”的连通区域（红色）。
    """
    out = img.copy()

    candidate_keys = set()
    for c in candidates:
        source = str(c.get("source", ""))
        label = c.get("component_label", -1)
        try:
            label = int(label)
        except Exception:
            label = -1
        candidate_keys.add((source, label))

    for comp in all_components:
        source = str(comp.get("source", ""))
        label = comp.get("component_label", -1)
        try:
            label = int(label)
        except Exception:
            label = -1

        if (source, label) in candidate_keys:
            continue

        source_mask = comp.get("source_mask")
        if source_mask is None:
            continue

        mask = source_mask > 0
        if not np.any(mask):
            continue

        alpha = 0.40
        overlay = out.copy()
        overlay[mask] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, max(2, thickness + 1))

    return out


def draw_emphasized_candidates_on_diff(diff_bgr, candidates):
    """
    仅用于差异图显示：候选区域红色加粗，提升肉眼可见性。
    """
    out = diff_bgr.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for c in candidates:
        source_mask = c.get("source_mask")
        if source_mask is None:
            continue

        core_u8 = (source_mask > 0).astype(np.uint8)
        if not np.any(core_u8):
            continue

        thick = cv2.dilate(core_u8, kernel, iterations=DIFF_CANDIDATE_THICKEN_ITER) > 0
        out[thick] = (0, 0, 255)

    return out


def draw_validation_region_and_max_outside_on_diff(diff_bgr, candidate):
    """
    在差异图上同时标注：
    1) 最终候选的验证区域（region_mask）
    2) 该候选验证时的最大区外连通域（max_outside_component_mask）
    """
    out = diff_bgr.copy()
    if not isinstance(candidate, dict):
        return out

    region_mask = candidate.get("region_mask")
    if region_mask is not None:
        region_bool = region_mask > 0
        if np.any(region_bool):
            overlay = out.copy()
            overlay[region_bool] = (255, 0, 0)  # blue
            out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)
            region_u8 = (region_bool.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 0, 0), 2)

    max_outside_mask = candidate.get("max_outside_component_mask")
    if max_outside_mask is not None:
        outside_bool = max_outside_mask > 0
        if np.any(outside_bool):
            overlay = out.copy()
            overlay[outside_bool] = (0, 255, 0)  # green
            out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
            outside_u8 = (outside_bool.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(outside_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (0, 255, 0), 2)

    return out


def show_resized_window(name, img, scale=DISPLAY_WINDOW_SCALE):
    h, w = img.shape[:2]
    new_w = max(200, int(w * scale))
    new_h = max(150, int(h * scale))
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, new_w, new_h)
    cv2.imshow(name, img)


def enhance_diff_for_display(edge_diff):
    """
    仅用于显示增强：让差异点在窗口中更容易看见（不参与任何判定逻辑）。
    """
    if DIFF_DISPLAY_DILATE_ITER <= 0:
        return edge_diff
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edge_diff, kernel, iterations=DIFF_DISPLAY_DILATE_ITER)


def show_three_result_images(result):
    """
    仅显示三张图：
    1) img1 候选连通区域
    2) img2 候选连通区域
    3) 差异图上的最终验证区域
    """
    img1 = result["img1"]
    img2 = result["img2"]
    edge_diff = result["edge_diff"]

    img1_candidates = result.get("img1_candidates", [])
    img2_candidates = result.get("img2_candidates", [])
    img1_all_components = result.get("img1_all_components", [])
    img2_all_components = result.get("img2_all_components", [])
    valid_candidates = result.get("valid_candidates", [])
    final_candidate = result.get("candidate")
    selected_candidate_full = result.get("selected_candidate_full")

    # 先画非候选连通区域（红色），再画候选（黄/蓝/紫）。
    img1_view = img1.copy()
    img1_view = draw_non_candidate_regions(
        img1_view,
        img1_all_components,
        img1_candidates,
        color=(0, 0, 255),
        thickness=2,
    )
    img1_view = draw_candidates_by_kind(img1_view, img1_candidates, thickness=2)

    img2_view = img2.copy()
    img2_view = draw_non_candidate_regions(
        img2_view,
        img2_all_components,
        img2_candidates,
        color=(0, 0, 255),
        thickness=2,
    )
    img2_view = draw_candidates_by_kind(img2_view, img2_candidates, thickness=2)

    edge_diff_vis = enhance_diff_for_display(edge_diff)
    global_view = np.zeros((edge_diff_vis.shape[0], edge_diff_vis.shape[1], 3), dtype=np.uint8)
    global_view[edge_diff_vis > 0] = (180, 180, 180)  # all diff in light gray
    if isinstance(selected_candidate_full, dict):
        global_view = draw_validation_region_and_max_outside_on_diff(global_view, selected_candidate_full)
    elif isinstance(final_candidate, dict):
        global_view = draw_emphasized_candidates_on_diff(global_view, [final_candidate])
    else:
        global_view = draw_emphasized_candidates_on_diff(global_view, valid_candidates)

    show_resized_window("img1_candidates_view", img1_view)
    show_resized_window("img2_candidates_view", img2_view)
    show_resized_window("global_edge_diff_result_view", global_view)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





class UIMultiModalComparator:
    def __init__(self):
        # 缓存：同一张图在同一进程里只加载一次 embedding。
        self._feature_cache = {}

    def _resolve_embedding_npz_path(self, img_path):
        img_p = Path(str(img_path)).expanduser().resolve()
        stem = img_p.stem

        # 兼容：
        # 1) app_dir/screen_x.png -> app_dir/embeddings/screen_x.npz
        # 2) app_dir/states/screen_x.png -> app_dir/embeddings/screen_x.npz
        for parent in [img_p.parent] + list(img_p.parents):
            cand = parent / "embeddings" / f"{stem}.npz"
            if cand.exists() and cand.is_file():
                return cand

        raise FileNotFoundError(
            f"Embedding npz not found for image '{img_p}'. "
            f"Expected something like '<app>/embeddings/{stem}.npz'."
        )

    @staticmethod
    def _normalize_vec(x):
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return v
        return v / n

    @staticmethod
    def _cosine(a, b):
        if a.shape != b.shape:
            raise ValueError(f"Vector shape mismatch: {a.shape} vs {b.shape}")
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an <= 1e-12 or bn <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def get_features(self, img_path):
        img_key = str(Path(str(img_path)).expanduser().resolve())
        if img_key in self._feature_cache:
            return self._feature_cache[img_key]

        emb_npz_path = self._resolve_embedding_npz_path(img_key)
        data = np.load(str(emb_npz_path), allow_pickle=False)

        if "semantic" not in data or "layout" not in data:
            raise KeyError(
                f"Embedding file missing keys in '{emb_npz_path}'. "
                "Required: 'semantic' and 'layout'."
            )

        semantic = self._normalize_vec(data["semantic"])
        layout = self._normalize_vec(data["layout"])
        self._feature_cache[img_key] = (semantic, layout)
        return semantic, layout

    def compare(
        self,
        path_a,
        path_b,
        rgb_same_threshold=0.87,
        edge_same_threshold=0.85,
        rgb_strong_same_threshold=0.92,
        edge_strong_same_threshold=0.90,
    ):
        # 直接读取 embeddings/*.npz 的 semantic/layout 向量计算余弦相似度
        sig_a, dino_a = self.get_features(path_a)
        sig_b, dino_b = self.get_features(path_b)

        rgb_sim = self._cosine(sig_a, sig_b)
        edge_sim = self._cosine(dino_a, dino_b)

        # --- 非加权判定逻辑 (双门控) ---
        # 语义门槛：确保页面主题/内容一致
        # 布局门槛：由于边缘图极其精确，通常阈值设得很高 (0.96+)
        
        is_same_page = False
        reason = "threshold_not_pass"
        if rgb_sim >= rgb_same_threshold and edge_sim >= edge_same_threshold:
            is_same_page = True
            reason = "rgb_and_edge_pass"
        elif rgb_sim >= rgb_strong_same_threshold:
            is_same_page = True
            reason = "rgb_strong_pass"
        elif edge_sim >= edge_strong_same_threshold:
            is_same_page = True
            reason = "edge_strong_pass"

        return {
            "same_page": is_same_page,
            "reason": reason,
            "scores": {
                "rgb": round(rgb_sim, 4),
                "edge": round(edge_sim, 4),
            }
        }


_default_comparator = None


def _get_default_comparator():
    global _default_comparator
    if _default_comparator is None:
        _default_comparator = UIMultiModalComparator()
    return _default_comparator


def judge_same_page(img1_path, img2_path, comparator=None):
    popup_result = judge_popup_relation(img1_path, img2_path)
    popup_bbox = None
    popup_boundary_type = "none"
    if popup_result.get("candidate"):
        popup_candidate = popup_result["candidate"]
        popup_bbox = popup_candidate.get("rect")
        if popup_candidate.get("kind") == "popup_border":
            popup_boundary_type = "boundary_box"
        elif popup_candidate.get("kind") == "popup_edge":
            popup_boundary_type = "boundary_edge"

    if popup_result["same_page"]:
        return {
            "same_page": True,
            "stage": "popup",
            "reason": popup_result["reason"],
            "popup_bbox": popup_bbox,
            "popup_boundary_type": popup_boundary_type,
            "popup_result": popup_result,
            "encoding_result": None,
        }

    if comparator is None:
        comparator = _get_default_comparator()

    encoding_result = comparator.compare(img1_path, img2_path)
    return {
        "same_page": encoding_result["same_page"],
        "stage": "encoding",
        "reason": encoding_result["reason"],
        "popup_bbox": popup_bbox,
        "popup_boundary_type": popup_boundary_type,
        "popup_result": popup_result,
        "encoding_result": encoding_result,
    }


def print_validation_max_outside_area(result):
    evaluated_candidates = result.get("evaluated_candidates", []) or []
    print("\n[VALIDATION-MAX-OUTSIDE-AREA]")
    for idx, cand in enumerate(evaluated_candidates):
        kind = str(cand.get("kind", ""))
        subtype = str(cand.get("subtype", ""))
        candidate_pixel_count = int(cand.get("source_pixel_count", 0))
        region_mask = cand.get("region_mask")
        region_pixel_count = int(np.sum(region_mask)) if region_mask is not None else 0
        max_area = int(cand.get("max_outside_component_area", 0))
        print(
            f"candidate_{idx}: kind={kind}, subtype={subtype}, "
            f"candidate_pixel_count={candidate_pixel_count}, "
            f"region_pixel_count={region_pixel_count}, "
            f"max_outside_component_area={max_area}"
        )


# =========================
# 主入口
# =========================

# if __name__ == "__main__":
#     result = judge_same_page(IMG1_PATH, IMG2_PATH)
#     print(f"same_page: {result['same_page']}")
#     print(f"stage: {result['stage']}")
#     print(f"reason: {result['reason']}")
#     print(f"popup_bbox: {result['popup_bbox']}")
#     print(f"popup_boundary_type: {result['popup_boundary_type']}")
#     if result["encoding_result"] is not None:
#         print(f"scores: {result['encoding_result']['scores']}")


if __name__ == "__main__":
    result = judge_popup_relation(IMG1_PATH, IMG2_PATH)

    print({
        "same_page": result["same_page"],
        "relation": result["relation"],
        "reason": result["reason"],
        "popup_type": result["popup_type"],
        "valid_candidate_count": result["valid_candidate_count"],
        "candidate": result["candidate"],
    })
    print_validation_max_outside_area(result)

    if SHOW_RESULT:
        show_three_result_images(result)
