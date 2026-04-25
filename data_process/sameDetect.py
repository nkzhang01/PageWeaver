import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel


# =========================================================
# 1. Define input paths and parameters here
# =========================================================



device = "cuda" if torch.cuda.is_available() else "cpu"

# SigLIP 2 NaFlex
model_name = "google/siglip2-so400m-patch16-naflex"

# popup detection parameters
popup_target_size = 1000
popup_detect_density = 5
popup_shreshood = 0.5
popup_patch_size = 25
popup_min_area = 50


# whether to show visualization windows
show_windows = True
_default_embedder = None


# =========================================================
# 2. Basic utilities
# =========================================================
def read_image_bgr(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image does not exist: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def resize_for_screen(img, max_w=1200, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_pil_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def make_edge_pil(img_bgr, low=80, high=160):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edge_rgb)


def cosine_similarity_np(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def apply_mask_on_white_background(img, mask):
    """
    Keep mask region, fill the rest with white.
    """
    if mask is None:
        return np.full_like(img, 255)
    out = np.full_like(img, 255)
    out[mask > 0] = img[mask > 0]
    return out


def overlay_mask_on_image(img, mask, color=(0, 0, 255)):
    """
    Highlight mask region on original image.
    """
    out = img.copy()
    if mask is not None:
        out[mask > 0] = color
    return out


# =========================================================
# 3. Popup relation detection: merged single-file version
# =========================================================
def load_popup_image(image_path):
    if image_path is None:
        return None
    arr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if arr is not None:
        return arr
    if not os.path.exists(str(image_path)):
        return None
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _ensure_bgr_uint8(img):
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resolve_popup_input(x):
    if x is None:
        return None
    if isinstance(x, (str, os.PathLike)):
        return load_popup_image(x)
    if isinstance(x, np.ndarray):
        return _ensure_bgr_uint8(x)
    return None


def _safe_corr(a, b):
    a = np.asarray(a).astype(np.float32).reshape(-1)
    b = np.asarray(b).astype(np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 1.0 if np.mean(np.abs(a - b)) < 1e-6 else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _hue_circular_diff(h1, h2):
    diff = np.abs(h1 - h2)
    return np.minimum(diff, 180.0 - diff)


def get_adaptive_kernel_size(diff_points, min_kernel=3, max_kernel=50):
    if len(diff_points) < 3:
        return min_kernel
    points = np.array(diff_points)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    nearest_dist = np.median(dists[:, 1])
    avg_dist = np.mean(nearest_dist)

    kernel_size = int(avg_dist / 2) + 2
    kernel_size = max(min_kernel, min(kernel_size, max_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size



def get_diff_regions(diff_points, sampling_boundary,min_area=50, original_shape=None, padding=10):
       
    h, w = original_shape[:2]

    # 1. 创建原始差异点掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y) in diff_points:
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = 255


    # 2. 获取自适应核大小并执行膨胀（用于连通邻近点）
    kernel_size = get_adaptive_kernel_size(diff_points)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=2)

    # 3. 提取连通域轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    original_regions = []
    padding = kernel_size

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # 提取该区域内的原始差异点
        roi_mask = mask[y:y + bh, x:x + bw]
        ys, xs = np.where(roi_mask > 0)

        if len(xs) < 5:  # 过滤噪点
            continue

        # 计算差异点的实际物理边界
        px_min, px_max = xs.min(), xs.max()
        py_min, py_max = ys.min(), ys.max()

        # 核心逻辑：判断这些点是否包含图像边缘点
        # 全局坐标
        abs_x1, abs_x2 = x + px_min, x + px_max
        abs_y1, abs_y2 = y + py_min, y + py_max

        # 设定一个微小的容差误差 (epsilon)，防止浮点数计算偏差
        eps = 1.0


        # 判断是否触碰了采样网格的边界
        is_at_grid_left = abs_x1 <= (sampling_boundary['left'] + eps)
        is_at_grid_top = abs_y1 <= (sampling_boundary['top'] + eps)
        is_at_grid_right = abs_x2 >= (sampling_boundary['right'] - eps)
        is_at_grid_bottom = abs_y2 >= (sampling_boundary['bottom'] - eps)

        # ── 边界处理策略 ──
        # 如果是内部点：x1 = abs_x1 (不向外扩)
        # 如果是图像边缘点：x1 = abs_x1 - padding (允许向外扩)

        final_x1 = max(0, abs_x1 - (padding if is_at_grid_left else 0))
        final_y1 = max(0, abs_y1 - (padding if is_at_grid_top else 0))
        final_x2 = min(w - 1, abs_x2 + (padding if is_at_grid_right else 0))
        final_y2 = min(h - 1, abs_y2 + (padding if is_at_grid_bottom else 0))

        # 面积过滤
        if (final_x2 - final_x1) * (final_y2 - final_y1) < min_area:
            continue

        original_regions.append((final_x1, final_y1, final_x2, final_y2))

    return original_regions


def get_diff_regions_1(diff_points, min_area=50, original_shape=None):
    if original_shape is None:
        return []

    mask = np.zeros(original_shape[:2], dtype=np.uint8)
    for (x, y) in diff_points:
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            mask[y, x] = 255

    kernel_size = get_adaptive_kernel_size(diff_points)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    original_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue

        roi_original = mask[y:y + h, x:x + w]
        original_point_count = cv2.countNonZero(roi_original)

        if original_point_count < 3:
            continue

        original_regions.append((x, y, x + w, y + h))

    return original_regions


def build_detection_region_mask_from_points(shape, demo_points, cover_radius):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(max(1, cover_radius))

    for i in range(demo_points.shape[0]):
        x = int(round(demo_points[i][0][0]))
        y = int(round(demo_points[i][0][1]))

        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w - 1, x + r)
        y2 = min(h - 1, y + r)

        mask[y1:y2 + 1, x1:x2 + 1] = 255

    return mask


def build_changed_unchanged_masks(
    img_resized_bgr,
    img_original_bgr,
    regions_orig,
    valid_detection_mask_resized
):
    img_resized_bgr = _ensure_bgr_uint8(img_resized_bgr)
    img_original_bgr = _ensure_bgr_uint8(img_original_bgr)

    if img_resized_bgr is None or img_original_bgr is None:
        return None

    h_r, w_r = img_resized_bgr.shape[:2]

    changed_mask_resized = np.zeros((h_r, w_r), dtype=np.uint8)
    for (x1, y1, x2, y2) in regions_orig:
        x1 = max(0, min(int(x1), w_r - 1))
        x2 = max(0, min(int(x2), w_r - 1))
        y1 = max(0, min(int(y1), h_r - 1))
        y2 = max(0, min(int(y2), h_r - 1))
        if x2 > x1 and y2 > y1:
            changed_mask_resized[y1:y2, x1:x2] = 255

    valid_detection_mask_resized = (valid_detection_mask_resized > 0).astype(np.uint8) * 255

    unchanged_mask_resized = np.zeros_like(changed_mask_resized)
    unchanged_mask_resized[
        (valid_detection_mask_resized == 255) & (changed_mask_resized == 0)
    ] = 255

    changed_mask_original = cv2.resize(
        changed_mask_resized,
        (img_original_bgr.shape[1], img_original_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    unchanged_mask_original = cv2.resize(
        unchanged_mask_resized,
        (img_original_bgr.shape[1], img_original_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return {
        "changed_mask_original": changed_mask_original,
        "unchanged_mask_original": unchanged_mask_original,
    }


def popup_detection_from_masks(base_img_bgr, current_img_bgr, changed_mask, unchanged_mask):
    metrics = {
        "valid": False,
        "is_popup": False,
        "change_ratio": 0.0,
        "structure_similarity": 0.0,
        "uniform_darkening_score": 0.0,
        "brightness_dominance_score": 0.0,
        "uniform_brightening_score": 0.0,
        "brightness_increase_dominance_score": 0.0,
        "structure_preserved": False,
        "background_uniform_darkening": False,
        "background_uniform_brightening": False,
        "brightness_dominant_not_recolor": False,
        "brightness_increase_dominant_not_recolor": False,
        "popup_by_darkening": False,
        "popup_by_brightening": False,
    }

    base = _ensure_bgr_uint8(base_img_bgr)
    current = _ensure_bgr_uint8(current_img_bgr)
    if base is None or current is None:
        return metrics

    if base.shape[:2] != current.shape[:2]:
        current = cv2.resize(current, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)

    if changed_mask.shape[:2] != base.shape[:2]:
        changed_mask = cv2.resize(changed_mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    if unchanged_mask.shape[:2] != base.shape[:2]:
        unchanged_mask = cv2.resize(unchanged_mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    changed_mask = (changed_mask > 0).astype(np.uint8) * 255
    unchanged_mask = (unchanged_mask > 0).astype(np.uint8) * 255

    changed_bool = changed_mask > 0
    unchanged_bool = unchanged_mask > 0

    if changed_bool.sum() < 20 or unchanged_bool.sum() < 20:
        return metrics

    valid_bool = (changed_mask > 0) | (unchanged_mask > 0)
    metrics["change_ratio"] = float(
        np.count_nonzero(changed_mask > 0) / (np.count_nonzero(valid_bool) + 1e-8)
    )

    # metrics["change_ratio"] = float(np.mean(changed_bool))

    hsv_base = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_curr = cv2.cvtColor(current, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_base = hsv_base[:, :, 2]
    v_curr = hsv_curr[:, :, 2]
    # 1) background structure preserved
    gx1 = cv2.Sobel(v_base, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(v_base, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(v_curr, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(v_curr, cv2.CV_32F, 0, 1, ksize=3)

    grad_base = cv2.magnitude(gx1, gy1)
    grad_curr = cv2.magnitude(gx2, gy2)
    grad_mix = np.maximum(grad_base, grad_curr)

    texture_vals = grad_mix[unchanged_bool]
    if texture_vals.size == 0:
        return metrics

    grad_thresh = float(np.percentile(texture_vals, 60))
    texture_mask = (grad_mix >= grad_thresh) & unchanged_bool
    if float(np.mean(texture_mask)) < 0.01:
        texture_mask = unchanged_bool

    structure_similarity = _safe_corr(grad_base[texture_mask], grad_curr[texture_mask])
    metrics["structure_similarity"] = structure_similarity
    metrics["structure_preserved"] = structure_similarity >= 0.75

    # 2) background trend on unchanged region: uniform darkening OR uniform brightening
    drop_map = v_base - v_curr
    rise_map = v_curr - v_base
    bg_drop = drop_map[unchanged_bool]
    bg_rise = rise_map[unchanged_bool]

    def _uniform_shift_metrics(shift_values):
        median_shift = float(np.median(shift_values))
        mad = float(np.median(np.abs(shift_values - median_shift)))

        if mad < 1e-6:
            inlier_mask = np.abs(shift_values - median_shift) <= 3.0
        else:
            inlier_mask = np.abs(shift_values - median_shift) <= 3.5 * mad

        if int(np.count_nonzero(inlier_mask)) < max(50, int(shift_values.size * 0.25)):
            inlier_mask = np.ones_like(shift_values, dtype=bool)

        inlier_values = shift_values[inlier_mask]
        mean_shift = float(np.mean(inlier_values))#平均变暗/变亮幅度
        std_shift = float(np.std(inlier_values))#变化是否一致
        positive_ratio = float(np.mean(inlier_values > 4.0))#有多少背景像素确实在按这个方向变化
        #连续分数
        uniform_score = np.clip(
            (mean_shift / 25.0) * positive_ratio * (1.0 / (1.0 + std_shift / 20.0)),
            0.0,
            1.0,
        )
        uniform_bool = (
            mean_shift > 7.0 and
            positive_ratio > 0.72 and
            std_shift < max(10.0, mean_shift * 0.9)
        )
        return inlier_values, uniform_score, uniform_bool

    inlier_drop, uniform_darkening_score, background_uniform_darkening = _uniform_shift_metrics(bg_drop)
    inlier_rise, uniform_brightening_score, background_uniform_brightening = _uniform_shift_metrics(bg_rise)

    metrics["uniform_darkening_score"] = float(uniform_darkening_score)
    metrics["uniform_brightening_score"] = float(uniform_brightening_score)
    metrics["background_uniform_darkening"] = bool(background_uniform_darkening)
    metrics["background_uniform_brightening"] = bool(background_uniform_brightening)

    # 3) brightness-dominant change, not recoloring
    h_diff = _hue_circular_diff(hsv_base[:, :, 0], hsv_curr[:, :, 0])[unchanged_bool]#色相
    s_diff = np.abs(hsv_base[:, :, 1] - hsv_curr[:, :, 1])[unchanged_bool]#饱和度
    mean_h = float(np.mean(h_diff))
    mean_s = float(np.mean(s_diff))
    mean_v_drop = float(np.mean(np.clip(inlier_drop, 0.0, None)))
    mean_v_rise = float(np.mean(np.clip(inlier_rise, 0.0, None)))

    color_shift = 0.55 * (mean_h / 90.0) + 0.45 * (mean_s / 128.0)
    brightness_drop_strength = mean_v_drop / 40.0
    brightness_rise_strength = mean_v_rise / 40.0

    brightness_dominance_score = np.clip(brightness_drop_strength - color_shift + 0.4, 0.0, 1.0)#连续变暗分数
    brightness_increase_dominance_score = np.clip(brightness_rise_strength - color_shift + 0.4, 0.0, 1.0)#变亮连续分数

    metrics["brightness_dominance_score"] = float(brightness_dominance_score)
    metrics["brightness_increase_dominance_score"] = float(brightness_increase_dominance_score)
    metrics["brightness_dominant_not_recolor"] = (
        mean_v_drop > 7.0 and
        mean_h < 10.0 and
        mean_s < 22.0 and
        brightness_drop_strength > (color_shift + 0.15)
    )
    metrics["brightness_increase_dominant_not_recolor"] = (
        mean_v_rise > 7.0 and
        mean_h < 10.0 and
        mean_s < 22.0 and
        brightness_rise_strength > (color_shift + 0.15)
    )

    metrics["valid"] = True
    metrics["popup_by_darkening"] = bool(
        metrics["structure_preserved"] and
        metrics["background_uniform_darkening"] and
        metrics["brightness_dominant_not_recolor"]
    )
    metrics["popup_by_brightening"] = bool(
        metrics["structure_preserved"] and
        metrics["background_uniform_brightening"] and
        metrics["brightness_increase_dominant_not_recolor"]
    )
    metrics["is_popup"] = bool(
        (metrics["popup_by_darkening"] or metrics["popup_by_brightening"]) and
        0.10 <= metrics["change_ratio"] <= 0.90
    )

    return metrics


def is_popup_relation(
    before_input,
    after_input,
    target_size=1000,
    detect_density=5,
    shreshood=0.5,
    patch_size=25,
    min_area=50
):
    img1_original = _resolve_popup_input(before_input)
    img2_original = _resolve_popup_input(after_input)

    if img1_original is None or img2_original is None:
        return {
            "is_popup": False,
            "metrics": {"valid": False},
            "changed_mask_original": None,
            "unchanged_mask_original": None,
        }

    img1_original = _ensure_bgr_uint8(img1_original)
    img2_original = _ensure_bgr_uint8(img2_original)

    img1 = img1_original.copy()
    img2 = img2_original.copy()

    if img1.shape[0] < img1.shape[1]:
        new_h = int(target_size * img1.shape[0] / img1.shape[1])
        img1 = cv2.resize(img1, (target_size, new_h))
        img2 = cv2.resize(img2, (target_size, new_h))
    else:
        new_w = int(target_size * img1.shape[1] / img1.shape[0])
        img1 = cv2.resize(img1, (new_w, target_size))
        img2 = cv2.resize(img2, (new_w, target_size))

    height, width, _ = img1.shape

    if not hasattr(cv2, "SIFT_create"):
        return {
            "is_popup": False,
            "metrics": {"valid": False},
            "changed_mask_original": None,
            "unchanged_mask_original": None,
        }
    sift = cv2.SIFT_create()

    size = max(4, int(width * 0.01))
    interval = max(2, detect_density * size)

    x_min = size + interval
    y_min = size + interval
    x_max = width
    y_max = height

    search_width = max(1, (x_max - x_min) // interval)
    search_height = max(1, (y_max - y_min) // interval)
    search_num = search_width * search_height

    demo_src = np.float32([[0] * 2] * search_num).reshape(-1, 1, 2)
    for i in range(search_width):
        for j in range(search_height):
            demo_src[i + j * search_width][0][0] = x_min + i * interval
            demo_src[i + j * search_width][0][1] = y_min + j * interval

    demo_dst = demo_src.copy()

    # 计算采样点网格的四个边界值
    # 注意：这些是采样点能触达的最外层坐标
    grid_x_min = x_min
    grid_y_min = y_min
    grid_x_max = x_min + (search_width - 1) * interval
    grid_y_max = y_min + (search_height - 1) * interval

    # 封装成一个字典或对象方便后续函数调用
    sampling_boundary = {
        'left': grid_x_min,
        'top': grid_y_min,
        'right': grid_x_max,
        'bottom': grid_y_max
    }

    valid_detection_mask_resized = build_detection_region_mask_from_points(
        shape=img2.shape,
        demo_points=demo_dst,
        cover_radius=max(size, interval // 2)
    )

    kp_src = [
        cv2.KeyPoint(float(demo_src[i][0][0]), float(demo_src[i][0][1]), size)
        for i in range(demo_src.shape[0])
    ]
    kp_dst = [
        cv2.KeyPoint(float(demo_dst[i][0][0]), float(demo_dst[i][0][1]), size)
        for i in range(demo_dst.shape[0])
    ]

    _, descriptors_image1 = sift.compute(img1, kp_src)
    _, descriptors_image2 = sift.compute(img2, kp_dst)

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    diffRight = []
    h2, w2 = img2.shape[:2]

    for i in range(search_num):
        x = int(demo_dst[i][0][0])
        y = int(demo_dst[i][0][1])
        x_src = int(demo_src[i][0][0])
        y_src = int(demo_src[i][0][1])

        if x < 0 or x >= w2 or y < 0 or y >= h2:
            diffRight.append([x, y])
            continue

        if descriptors_image1 is None or descriptors_image2 is None or \
           descriptors_image1[i] is None or descriptors_image2[i] is None:
            diffRight.append([x, y])
            continue

        d1 = descriptors_image1[i].astype(np.float32)
        d2 = descriptors_image2[i].astype(np.float32)
        d1 /= (np.linalg.norm(d1) + 1e-7)
        d2 /= (np.linalg.norm(d2) + 1e-7)
        diff = np.linalg.norm(d1 - d2)

        is_different = False

        if diff > shreshood:
            is_different = True
        else:
            y1a = max(0, y_src - patch_size)
            y2a = min(h2, y_src + patch_size)
            x1a = max(0, x_src - patch_size)
            x2a = min(w2, x_src + patch_size)

            y1b = max(0, y - patch_size)
            y2b = min(h2, y + patch_size)
            x1b = max(0, x - patch_size)
            x2b = min(w2, x + patch_size)

            patch1 = hsv1[y1a:y2a, x1a:x2a]
            patch2 = hsv2[y1b:y2b, x1b:x2b]

            min_h = min(patch1.shape[0], patch2.shape[0])
            min_w = min(patch1.shape[1], patch2.shape[1])

            if min_h > 8 and min_w > 8:
                patch1 = cv2.resize(patch1, (min_w, min_h))
                patch2 = cv2.resize(patch2, (min_w, min_h))

                mean1 = np.mean(patch1, axis=(0, 1))
                mean2 = np.mean(patch2, axis=(0, 1))

                h_mean_diff = abs(mean1[0] - mean2[0])
                h_mean_diff = min(h_mean_diff, 180 - h_mean_diff)
                s_mean_diff = abs(mean1[1] - mean2[1])
                v_mean_diff = abs(mean1[2] - mean2[2])

                if h_mean_diff > 15 or s_mean_diff > 40 or v_mean_diff > 180:
                    is_different = True

        if is_different:
            diffRight.append([x, y])

    # right_regions_orig = get_diff_regions(
    #     diffRight,
    #     min_area=min_area,
    #     original_shape=img2.shape
    # )

    right_regions_orig = get_diff_regions(
        diffRight,
        sampling_boundary,
        min_area=min_area,
        original_shape=img2.shape
    )

    masks = build_changed_unchanged_masks(
        img_resized_bgr=img2,
        img_original_bgr=img2_original,
        regions_orig=right_regions_orig,
        valid_detection_mask_resized=valid_detection_mask_resized
    )

    if masks is None:
        return {
            "is_popup": False,
            "metrics": {"valid": False},
            "changed_mask_original": None,
            "unchanged_mask_original": None,
        }

    metrics = popup_detection_from_masks(
        base_img_bgr=img1_original,
        current_img_bgr=img2_original,
        changed_mask=masks["changed_mask_original"],
        unchanged_mask=masks["unchanged_mask_original"]
    )

    return {
        "is_popup": bool(metrics.get("is_popup", False)),
        "metrics": metrics,
        "changed_mask_original": masks["changed_mask_original"],
        "unchanged_mask_original": masks["unchanged_mask_original"],
    }

# =========================================================
# 4. SigLIP 2 NaFlex embedding
# =========================================================
class SigLIP2Embedder:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode_pil(self, img_pil):
        inputs = self.processor(images=img_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if hasattr(self.model, "get_image_features"):
            feat = self.model.get_image_features(**inputs)
        else:
            outputs = self.model(**inputs)
            if hasattr(outputs, "image_embeds"):
                feat = outputs.image_embeds
            else:
                feat = outputs[0]

        feat = F.normalize(feat, p=2, dim=-1)
        return feat[0].detach().cpu().numpy()


def confirm_same_page_with_embedding(
    img1_path,
    img2_path,
    embedder,
    rgb_same_threshold=0.87,
    edge_same_threshold=0.85,
    rgb_strong_same_threshold=0.92,
    edge_strong_same_threshold=0.90,
    canny_low=80,
    canny_high=160
):
    img1_bgr = read_image_bgr(img1_path)
    img2_bgr = read_image_bgr(img2_path)

    img1_rgb_pil = bgr_to_pil_rgb(img1_bgr)
    img2_rgb_pil = bgr_to_pil_rgb(img2_bgr)

    emb1_rgb = embedder.encode_pil(img1_rgb_pil)
    emb2_rgb = embedder.encode_pil(img2_rgb_pil)
    rgb_sim = cosine_similarity_np(emb1_rgb, emb2_rgb)

    img1_edge_pil = make_edge_pil(img1_bgr, low=canny_low, high=canny_high)
    img2_edge_pil = make_edge_pil(img2_bgr, low=canny_low, high=canny_high)

    emb1_edge = embedder.encode_pil(img1_edge_pil)
    emb2_edge = embedder.encode_pil(img2_edge_pil)
    edge_sim = cosine_similarity_np(emb1_edge, emb2_edge)

    is_same_page = False
    reason = "different_page"

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
        "is_same_page": bool(is_same_page),
        "rgb_similarity": float(rgb_sim),
        "edge_similarity": float(edge_sim),
        "reason": reason,
        "thresholds": {
            "rgb_same_threshold": rgb_same_threshold,
            "edge_same_threshold": edge_same_threshold,
            "rgb_strong_same_threshold": rgb_strong_same_threshold,
            "edge_strong_same_threshold": edge_strong_same_threshold,
        }
    }


# =========================================================
# 5. Main pipeline: popup first, then embedding
# =========================================================
def judge_screen_relation(
    img1_path,
    img2_path,
    embedder=None
):
    if embedder is None:
        global _default_embedder
        if _default_embedder is None:
            _default_embedder = SigLIP2Embedder(model_name=model_name, device=device)
        embedder = _default_embedder

    popup_res = is_popup_relation(
        img1_path,
        img2_path,
        target_size=popup_target_size,
        detect_density=popup_detect_density,
        shreshood=popup_shreshood,
        patch_size=popup_patch_size,
        min_area=popup_min_area
    )

    if popup_res.get("is_popup", False):
        return {
            "relation": "same_page_popup",
            "same_page": True,
            "popup_result": popup_res,
            "embedding_result": None
        }

    embedding_res = confirm_same_page_with_embedding(
        img1_path=img1_path,
        img2_path=img2_path,
        embedder=embedder,
    )

    relation = "same_page_non_popup" if embedding_res["is_same_page"] else "different_page"

    return {
        "relation": relation,
        "same_page": bool(embedding_res["is_same_page"]),
        "popup_result": popup_res,
        "embedding_result": embedding_res
    }


# =========================================================
# 6. Visualization
# =========================================================
def show_compare_windows(img1_path, img2_path, popup_result):
    img1 = read_image_bgr(img1_path)
    img2 = read_image_bgr(img2_path)

    changed_mask = popup_result.get("changed_mask_original", None)
    unchanged_mask = popup_result.get("unchanged_mask_original", None)

    # Highlight changed area on original images
    img1_diff_overlay = overlay_mask_on_image(img1, changed_mask, color=(0, 0, 255))
    img2_diff_overlay = overlay_mask_on_image(img2, changed_mask, color=(0, 0, 255))

    # Extracted changed area
    img1_diff_part = apply_mask_on_white_background(img1, changed_mask)
    img2_diff_part = apply_mask_on_white_background(img2, changed_mask)

    # Extracted unchanged area
    img1_non_diff_part = apply_mask_on_white_background(img1, unchanged_mask)
    img2_non_diff_part = apply_mask_on_white_background(img2, unchanged_mask)

    show_list = [
        ("img1_diff_overlay", img1_diff_overlay),
        ("img2_diff_overlay", img2_diff_overlay),
        ("img1_diff_part", img1_diff_part),
        ("img2_diff_part", img2_diff_part),
        ("img1_non_diff_part", img1_non_diff_part),
        ("img2_non_diff_part", img2_non_diff_part),
    ]

    for win_name, img in show_list:
        cv2.imshow(win_name, resize_for_screen(img))

    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================================================
# 7. Main entry
# =========================================================
if __name__ == "__main__":
    # jiffy,musicplayer,valleythreesixfive
    img1_path = r"D:\ui2code\InteractionSequence\data_process\output\com\musicplayer\screen_125.png"
    img2_path = r"D:\ui2code\InteractionSequence\data_process\output\com\musicplayer\screen_44.png"
    result = judge_screen_relation(
        img1_path=img1_path,
        img2_path=img2_path
    )

    print("Final relation:", result["relation"])
    print("Same page:", result["same_page"])
    print()

    print("popup_result:")
    popup_result_to_print = {
        "is_popup": result["popup_result"]["is_popup"],
        "metrics": result["popup_result"]["metrics"],
    }
    print(popup_result_to_print)
    print()

    print("embedding_result:")
    print(result["embedding_result"])

    if show_windows:
        show_compare_windows(img1_path, img2_path, result["popup_result"])

