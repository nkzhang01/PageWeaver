import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoImageProcessor
import torch.nn.functional as F

# =========================
# 直接修改这里的路径
# =========================
IMG1_PATH = r"D:\ui2code\InteractionSequence\data\com\jiffy\screen_25.png"
IMG2_PATH = r"D:\ui2code\InteractionSequence\data\com\jiffy\screen_16.png"


SHOW_WINDOWS =  True

MIN_POPUP_BORDER_CONTINUOUS_PIXELS = 10
MIN_POPUP_BORDER_AREA = 200



# =========================
# 基础工具
# =========================

def load_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def ensure_same_size(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2


# =========================
# 轮廓工具
# =========================

def contour_basic_info(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    else:
        approx = contour

    x, y, w, h = cv2.boundingRect(contour)

    is_convex = False
    if len(approx) >= 4:
        is_convex = bool(cv2.isContourConvex(approx))

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "approx": approx,
        "approx_points": int(len(approx)),
        "is_convex": is_convex,
        "rect": [int(x), int(y), int(w), int(h)],
    }


def contour_to_filled_mask(contour, shape_hw):
    """
    将闭合差异轮廓填充成候选区域。
    注意：这个区域来自差异轮廓本身，不是人工构造的外接矩形。
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(
        mask,
        [contour],
        contourIdx=-1,
        color=255,
        thickness=-1
    )

    return mask > 0


def half_plane_mask_from_edge_rect(rect, shape_hw, side):
    """
    对 bottom sheet / drawer 的单边界生成候选区域。

    side:
        below: 边以下区域
        above: 边以上区域
        left : 边左侧区域
        right: 边右侧区域
    """
    h, w = shape_hw
    x, y, rw, rh = rect

    mask = np.zeros((h, w), dtype=bool)

    if side == "below":
        mask[y:h, :] = True

    elif side == "above":
        mask[0:y + rh, :] = True

    elif side == "left":
        mask[:, 0:x + rw] = True

    elif side == "right":
        mask[:, x:w] = True

    else:
        raise ValueError(f"unknown side: {side}")

    return mask


def iter_connected_edge_components(edge_diff):
    mask = (edge_diff > 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            yield label, component_mask, contour


# =========================
# 候选 1：闭合弹窗框
# =========================

def is_closed_popup_border(contour, component_mask):
    """
    判断 edge_diff 中的某个轮廓是否像：
    - 矩形弹窗框
    - 圆角矩形弹窗框
    - 近似矩形弹窗框

    注意：
    圆角矩形不要求 approx_points == 4。
    """
    info = contour_basic_info(contour)
    x, y, w, h = info["rect"]

    # Contours are already connected components. Require a continuous visible span.
    if max(w, h) < MIN_POPUP_BORDER_CONTINUOUS_PIXELS:
        return False

    if info["area"] < MIN_POPUP_BORDER_AREA:
        return False

    if info["approx_points"] < 4:
        return False

    if not info["is_convex"]:
        return False

    return True


def find_popup_border_candidates(edge_diff):
    """
    从 edge_diff 差异像素中找闭合弹窗框候选。
    """
    h, w = edge_diff.shape[:2]

    candidates = []

    for idx, component_mask, contour in iter_connected_edge_components(edge_diff):
        if not is_closed_popup_border(contour, component_mask):
            continue

        info = contour_basic_info(contour)
        region_mask = contour_to_filled_mask(contour, (h, w))
        source_mask = component_mask > 0

        candidates.append({
            "kind": "popup_border",
            "subtype": "closed_or_rounded_rect_border",
            "contour_index": idx,
            "contour": contour,
            "rect": info["rect"],
            "area": info["area"],
            "perimeter": info["perimeter"],
            "approx_points": info["approx_points"],
            "connected_component_mask": source_mask,
            "source_mask": source_mask,
            "source_pixel_count": int(np.sum(source_mask)),
            "region_mask": region_mask,
        })

    return candidates


# =========================
# 候选 2：bottom sheet / drawer 单边
# =========================

def is_separator_edge(contour, component_mask):
    """
    判断 edge_diff 中的某个轮廓是否可能是弹窗边：
    - 水平边：bottom sheet / top sheet
    - 垂直边：drawer
    """
    info = contour_basic_info(contour)
    x, y, w, h = info["rect"]

    if w <= 0 or h <= 0:
        return False

    return True


def find_popup_edge_candidates(edge_diff, excluded_component_ids=None):
    """
    从 edge_diff 差异像素中寻找单条弹窗边候选。

    水平边：
        below -> bottom sheet
        above -> top sheet

    垂直边：
        right -> right drawer
        left  -> left drawer
    """
    h_img, w_img = edge_diff.shape[:2]
    excluded_component_ids = set(excluded_component_ids or [])

    candidates = []

    for idx, component_mask, contour in iter_connected_edge_components(edge_diff):
        if idx in excluded_component_ids:
            continue

        if not is_separator_edge(contour, component_mask):
            continue

        info = contour_basic_info(contour)
        x, y, w, h = info["rect"]
        connected_component_mask = component_mask > 0
        source_mask = connected_component_mask
        edge_rect = info["rect"]

        if w >= h:
            candidates.append({
                "kind": "popup_edge",
                "subtype": "top_edge_of_bottom_sheet",
                "contour_index": idx,
                "contour": contour,
                "rect": edge_rect,
                "component_rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "side": "below",
                "connected_component_mask": connected_component_mask,
                "source_mask": source_mask,
                "source_pixel_count": int(np.sum(source_mask)),
                "region_mask": half_plane_mask_from_edge_rect(
                    edge_rect,
                    (h_img, w_img),
                    "below"
                ),
            })

            candidates.append({
                "kind": "popup_edge",
                "subtype": "bottom_edge_of_top_sheet",
                "contour_index": idx,
                "contour": contour,
                "rect": edge_rect,
                "component_rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "side": "above",
                "connected_component_mask": connected_component_mask,
                "source_mask": source_mask,
                "source_pixel_count": int(np.sum(source_mask)),
                "region_mask": half_plane_mask_from_edge_rect(
                    edge_rect,
                    (h_img, w_img),
                    "above"
                ),
            })

        if h >= w:
            candidates.append({
                "kind": "popup_edge",
                "subtype": "left_edge_of_right_drawer",
                "contour_index": idx,
                "contour": contour,
                "rect": edge_rect,
                "component_rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "side": "right",
                "connected_component_mask": connected_component_mask,
                "source_mask": source_mask,
                "source_pixel_count": int(np.sum(source_mask)),
                "region_mask": half_plane_mask_from_edge_rect(
                    edge_rect,
                    (h_img, w_img),
                    "right"
                ),
            })

            candidates.append({
                "kind": "popup_edge",
                "subtype": "right_edge_of_left_drawer",
                "contour_index": idx,
                "contour": contour,
                "rect": edge_rect,
                "component_rect": info["rect"],
                "area": info["area"],
                "perimeter": info["perimeter"],
                "approx_points": info["approx_points"],
                "side": "left",
                "connected_component_mask": connected_component_mask,
                "source_mask": source_mask,
                "source_pixel_count": int(np.sum(source_mask)),
                "region_mask": half_plane_mask_from_edge_rect(
                    edge_rect,
                    (h_img, w_img),
                    "left"
                ),
            })

    return candidates


# =========================
# 外部噪声分析
# =========================

def analyze_outside_noise(edge_diff, region_mask):
    """
    检查候选区域外的差异像素。

    要求：
    - 区域外允许孤立噪声点；
    - 区域外不允许形成连通结构。

    当前严格定义：
    - 连通域 area == 1：孤立噪声；
    - 连通域 area > 1 ：连通结构。
    """
    diff_bool = edge_diff > 0
    outside_bool = diff_bool & (~region_mask)

    outside_mask = outside_bool.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        outside_mask,
        connectivity=8
    )

    noise_points = []
    connected_regions = []

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

        if area == 1:
            noise_points.append(item)
        else:
            connected_regions.append(item)

    return {
        "outside_diff_count": int(np.sum(outside_bool)),
        "noise_point_count": len(noise_points),
        "connected_region_count": len(connected_regions),
        "connected_regions": connected_regions,
        "only_discrete_noise_outside": len(connected_regions) == 0,
    }


# =========================
# 验证候选
# =========================

def validate_candidate(edge_diff, candidate):
    """
    验证候选弹窗框 / 弹窗边是否能解释 edge_diff。

    popup_border：
        差异点应该在框内；
        框外只能有离散噪声。

    popup_edge：
        差异点应该在边的一侧；
        另一侧只能有离散噪声。
    """
    region_mask = candidate["region_mask"]

    outside_result = analyze_outside_noise(edge_diff, region_mask)

    diff_bool = edge_diff > 0

    inside_diff_count = int(np.sum(diff_bool & region_mask))
    total_diff_count = int(np.sum(diff_bool))

    return {
        **candidate,
        "valid": outside_result["only_discrete_noise_outside"],
        "total_diff_count": total_diff_count,
        "inside_diff_count": inside_diff_count,
        **outside_result,
    }


# =========================
# 主判断：要求唯一
# =========================

def judge_popup_relation(img1_path, img2_path):
    """
    主函数。

    same_page=True 当且仅当：

    1. edge_diff 中存在弹窗框或弹窗边；
    2. 该弹窗框或边来自 edge_diff 差异像素本身；
    3. 候选区域外只允许离散噪声，不允许连通结构；
    4. 满足条件的候选数量必须唯一。
    """
    # 1. 读取并转为灰度图（用于结构分析）
    img_a = cv2.imread(img1_path)
    img_b = cv2.imread(img2_path)
    
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # 2. 提取边缘结构 (Canny) 
    # 这一步是为了消除“背景压暗/遮罩”的影响，只保留 UI 的骨架
    edge1 = cv2.Canny(gray_a, 50, 150)
    edge2 = cv2.Canny(gray_b, 50, 150)

    # 3. 计算“结构差异图”
    # 如果背景没动，那么这个 diff 图里只有弹窗的轮廓
    edge_diff = cv2.absdiff(edge1, edge2)


    if not np.any(edge_diff > 0):
        return {
            "same_page": False,
            "relation": "not_popup_relation",
            "reason": "no_edge_difference",
            "popup_type": "none",
            "candidate": None,
            "valid_candidate_count": 0,
            "edge1": edge1,
            "edge2": edge2,
            "edge_diff": edge_diff,
            "candidate_regions": [],
            "selected_candidate_full": None,
        }

    border_candidates = find_popup_border_candidates(edge_diff)
    border_component_ids = {
        c["contour_index"]
        for c in border_candidates
    }
    edge_candidates = find_popup_edge_candidates(
        edge_diff,
        excluded_component_ids=border_component_ids
    )

    all_candidates = border_candidates + edge_candidates

    evaluated = [
        validate_candidate(edge_diff, c)
        for c in all_candidates
    ]

    valid_candidates = [
        c for c in evaluated
        if c["valid"]
    ]

    if len(valid_candidates) == 1:
        best = valid_candidates[0]

        return {
            "same_page": True,
            "relation": "popup_relation",
            "reason": "unique_popup_border_or_edge_from_edge_difference",
            "popup_type": best["subtype"],
            "candidate": {
                "kind": best["kind"],
                "subtype": best["subtype"],
                "rect": best["rect"],
                "component_rect": best.get("component_rect"),
                "area": best["area"],
                "perimeter": best["perimeter"],
                "approx_points": best["approx_points"],
                "total_diff_count": best["total_diff_count"],
                "inside_diff_count": best["inside_diff_count"],
                "outside_diff_count": best["outside_diff_count"],
                "source_pixel_count": best.get("source_pixel_count", 0),
                "noise_point_count": best["noise_point_count"],
                "connected_region_count": best["connected_region_count"],
            },
            "valid_candidate_count": len(valid_candidates),
            "edge1": edge1,
            "edge2": edge2,
            "edge_diff": edge_diff,
            "candidate_regions": evaluated,
            "selected_candidate_full": best,
        }

    if len(valid_candidates) == 0:
        reason = "no_popup_border_or_edge_can_explain_edge_difference"
    else:
        reason = "multiple_popup_borders_or_edges_can_explain_edge_difference"

    return {
        "same_page": False,
        "relation": "not_popup_relation",
        "reason": reason,
        "popup_type": "none",
        "candidate": None,
        "valid_candidate_count": len(valid_candidates),
        "edge1": edge1,
        "edge2": edge2,
        "edge_diff": edge_diff,
        "candidate_regions": evaluated,
        "selected_candidate_full": None,
    }


# =========================
# 可视化
# =========================

def overlay_edge_diff(img, edge_diff, color=(0, 255, 0)):
    """
    将 edge_diff 差异点叠加到原图上。
    """
    out = img.copy()
    out[edge_diff > 0] = color
    return out


def draw_candidate_on_image(img, candidate_full, color=(0, 0, 255), thickness=3):
    """
    把选中的弹窗框或弹窗边画到图上。
    红色表示最终唯一候选。
    """
    out = img.copy()

    if candidate_full is None:
        return out

    source_mask = candidate_full.get("source_mask")

    if source_mask is not None:
        out[source_mask] = color

    return out


def draw_candidate_regions_on_image(img, candidates):
    """
    Show candidate connected components:
    - popup_edge: red
    - popup_border: yellow
    """
    out = img.copy()

    for candidate in candidates:
        mask = candidate.get("connected_component_mask")
        if mask is None:
            mask = candidate.get("source_mask")
        if mask is None:
            continue

        if candidate.get("kind") == "popup_border":
            color = (0, 255, 255)  # yellow in BGR
        elif candidate.get("kind") == "popup_edge":
            color = (0, 0, 255)  # red in BGR
        else:
            continue

        out[mask] = color

    return out


def show_result(img1_path, img2_path, result):
    img1 = load_bgr(img1_path)
    img2 = load_bgr(img2_path)
    img1, img2 = ensure_same_size(img1, img2)

    edge1 = result["edge1"]
    edge2 = result["edge2"]
    edge_diff = result["edge_diff"]
    candidates = result.get("candidate_regions", [])
    selected = result.get("selected_candidate_full")

    img1_overlay = overlay_edge_diff(img1, edge_diff, color=(0, 255, 0))
    img2_overlay = overlay_edge_diff(img2, edge_diff, color=(0, 255, 0))

    img1_overlay = draw_candidate_regions_on_image(img1_overlay, candidates)
    img2_overlay = draw_candidate_regions_on_image(img2_overlay, candidates)

    if selected is not None:
        selected_color = (0, 255, 255)
        if selected.get("kind") == "popup_edge":
            selected_color = (0, 0, 255)
        img1_overlay = draw_candidate_on_image(img1_overlay, selected, color=selected_color)
        img2_overlay = draw_candidate_on_image(img2_overlay, selected, color=selected_color)

    cv2.imshow("edge1", edge1)
    cv2.imshow("edge2", edge2)
    cv2.imshow("edge_diff", edge_diff)

    cv2.imshow("img1_candidate_regions_edge_red_box_yellow", img1_overlay)
    cv2.imshow("img2_candidate_regions_edge_red_box_yellow", img2_overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



class UIMultiModalComparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. SigLIP2: 语义分支 (google/siglip2-so400m-patch16-naflex)
        sig_id = "google/siglip2-so400m-patch16-naflex"
        self.sig_model = AutoModel.from_pretrained(sig_id).to(self.device).eval()
        self.sig_proc = AutoProcessor.from_pretrained(sig_id)

        # 2. DINOv2: 布局分支 (facebook/dinov2-base)
        dino_id = "facebook/dinov2-base"
        self.dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_id)

    def _get_edge_image(self, img_path):
        """生成 3 通道的 Canny 边缘图"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 提取边缘
        edges = cv2.Canny(gray, 50, 150)
        # 线条稍加粗，增强 DINOv2 的特征捕获能力
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        # 转回 RGB 格式供模型使用
        edge_rgb = cv2.merge([edges, edges, edges])
        return Image.fromarray(edge_rgb)

    @torch.no_grad()
    def get_features(self, img_path):
        # --- 语义特征 (使用原图) ---
        raw_img = Image.open(img_path).convert("RGB")
        sig_inputs = self.sig_proc(images=raw_img, return_tensors="pt").to(self.device)
        sig_feat = self.sig_model.get_image_features(**sig_inputs)
        sig_feat = F.normalize(sig_feat, p=2, dim=-1)

        # --- 布局特征 (使用边缘图) ---
        edge_img = self._get_edge_image(img_path)
        dino_inputs = self.dino_proc(images=edge_img, return_tensors="pt").to(self.device)
        dino_out = self.dino_model(**dino_inputs)
        dino_feat = dino_out.last_hidden_state[:, 0, :] # CLS Token
        dino_feat = F.normalize(dino_feat, p=2, dim=-1)

        return sig_feat, dino_feat

    def compare(
        self,
        path_a,
        path_b,
        rgb_same_threshold=0.87,
        edge_same_threshold=0.85,
        rgb_strong_same_threshold=0.92,
        edge_strong_same_threshold=0.90,
    ):
        # 提取特征
        sig_a, dino_a = self.get_features(path_a)
        sig_b, dino_b = self.get_features(path_b)

        # 计算相似度
        rgb_sim = torch.mm(sig_a, sig_b.t()).item()
        edge_sim = torch.mm(dino_a, dino_b.t()).item()

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
    
# =========================
# 主入口
# =========================

if __name__ == "__main__":
    result = judge_same_page(IMG1_PATH, IMG2_PATH)
    print(f"same_page: {result['same_page']}")
    print(f"stage: {result['stage']}")
    print(f"reason: {result['reason']}")
    print(f"popup_bbox: {result['popup_bbox']}")
    print(f"popup_boundary_type: {result['popup_boundary_type']}")
    if result["encoding_result"] is not None:
        print(f"scores: {result['encoding_result']['scores']}")
    if SHOW_WINDOWS:
        show_result(IMG1_PATH, IMG2_PATH, result["popup_result"])
