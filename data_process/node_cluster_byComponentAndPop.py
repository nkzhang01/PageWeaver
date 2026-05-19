import os
import re
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict, deque, Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from testPOP_2 import judge_popup_relation



# ============================================================
# 1. 固定输入路径
# ============================================================

DATA_ROOT = Path(r"D:\ui2code\InteractionSequence\data")
COMPONENT_RESULTS_ROOT = Path(r"D:\ui2code\InteractionSequence\results\extract_results")
OUTPUT_CLUSTER_FILENAME = "page_componentAndPOP_clusters_by_utg.json"
OUTPUT_UTG_COMPONENT_FILENAME = "utg_componentAndPOP.json"


# ============================================================
# 2. 参数设置
# ============================================================

# 大部分英文 + 少量其他语言，推荐 bge-m3
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# BBOX 已经归一化，直接使用 IoU
IOU_THRESHOLD = 0.80

# embedding 语义相似度阈值
SEMANTIC_THRESHOLD = 0.90

# 至少有几个相同组件，就把两个页面连接聚类
MIN_SHARED_COMPONENTS = 1

# True：比较交互图中所有可达节点
# False：只比较直接边 from -> to
USE_TRANSITIVE_REACHABILITY = True

# 顶部区域阈值：组件中心点 cy < 0.15 认为在顶部区域
TOP_REGION_THRESHOLD = 0.15


# ============================================================
# 3. Component Type 映射到 9 个大类
# ============================================================

TYPE_GROUPS = {
    "button": [
        "Button",
        "Pressable",
        "TouchableHighlight",
        "TouchableOpacity Function Component",
        "TouchableWithoutFeedback",
    ],
    "switch": [
        "Switch",
    ],
    "list": [
        "SectionList",
        "flatlist-simple",
        "flatlist-selectable",
        "VirtualizedList",
    ],
    "view": [
        "View Function Component",
    ],
    "text": [
        "Text Functional Component",
    ],
    "textinput": [
        "TextInput",
    ],
    "image": [
        "Function Component",
        "ImageBackground",
    ],
    "scrollview": [
        "ScrollView",
    ],
    "other": [
        "Modal",
        "ActivityIndicator",
        "KeyboardAvoidingView",
        "RefreshControl",
        "StatusBar Android and iOS Component",
    ],
}


def normalize_type(component_type: str) -> str:
    component_type = str(component_type).strip()

    for group_name, type_list in TYPE_GROUPS.items():
        if component_type in type_list:
            return group_name

    return "unknown"


# ============================================================
# 4. 文本归一化与通用组件过滤
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    return text


# Function Description 中包含这些词就过滤
# 注意：不包含 routing / route
GENERIC_DESCRIPTION_WORDS = [
    "navigation",
    "navigating",
    "navigate",
    "navigates",
    "navigated",
    "navigator",
]


# Component Keyword 中包含这些词就过滤
GENERIC_KEYWORD_WORDS = [
    "ellipsis",
    "more",
    "back arrow",
]


def is_top_area_component(comp: dict) -> bool:
    """
    判断组件是否位于顶部区域。
    BBOX 已经是归一化坐标。

    顶部区域定义：
    组件中心点 cy < 0.15
    """

    x1, y1, x2, y2 = comp["bbox"]

    cy = (y1 + y2) / 2
    h = y2 - y1

    # 中心点位于顶部 15% 内；
    # 同时加高度限制，避免误删顶部大 banner / 大图片 / 大容器。
    return cy < TOP_REGION_THRESHOLD and h < 0.18


def is_top_icon_component(comp: dict) -> bool:
    """
    顶部区域内的图标组件过滤。

    判断依据：
    1. 位于顶部区域；
    2. keyword/function/type 中包含 icon / 图标；
    3. 或者顶部小型 button/image，通常也是图标按钮。
    """

    if not is_top_area_component(comp):
        return False

    keyword = normalize_text(comp.get("keyword", ""))
    function = normalize_text(comp.get("function", ""))
    ctype = normalize_text(comp.get("type", ""))
    type_group = comp.get("type_group", "")

    text = keyword + " " + function + " " + ctype

    x1, y1, x2, y2 = comp["bbox"]
    w = x2 - x1
    h = y2 - y1
    area = max(w, 0.0) * max(h, 0.0)

    # 明确包含 icon / 图标
    if "icon" in text or "图标" in text:
        return True

    # 顶部区域内的小型 button/image，通常是返回、搜索、更多、设置等图标按钮
    if type_group in {"button", "image"} and area < 0.035 and w < 0.20 and h < 0.20:
        return True

    return False


def is_generic_component(comp: dict) -> bool:
    """
    过滤通用组件，不按底部/顶部导航区域做复杂判断。

    过滤规则：
    1. Component Function Description 中包含：
       navigation / navigating / navigate / navigates / navigated / navigator

    2. Component Keyword 中包含：
       ellipsis / more / back arrow

    3. 顶部区域 cy < 0.15 内的 icon / 图标组件过滤
    """

    keyword = normalize_text(comp.get("keyword", ""))
    function = normalize_text(comp.get("function", ""))

    # 规则 1：description 中包含导航相关词
    for word in GENERIC_DESCRIPTION_WORDS:
        if word in function:
            return True

    # 规则 2：keyword 中包含通用组件词
    for word in GENERIC_KEYWORD_WORDS:
        if word in keyword:
            return True

    # 规则 3：顶部区域内 icon 都过滤
    if is_top_icon_component(comp):
        return True

    return False


# ============================================================
# 5. 页面名处理
# ============================================================

def normalize_page_name_from_node_id(node_id):
    """
    screen_25.png -> screen_25

    对应组件分析文件：
    screen_25_type.txt
    """

    if node_id is None:
        return None

    node_id = str(node_id).strip()

    name = os.path.basename(node_id)
    name = name.split("?")[0].split("#")[0]

    lower = name.lower()

    if lower.endswith(".png"):
        name = name[:-4]
    elif lower.endswith(".jpg"):
        name = name[:-4]
    elif lower.endswith(".jpeg"):
        name = name[:-5]

    return name


# ============================================================
# 6. 读取 utg.json
# ============================================================

def load_utg_graph(utg_path: Path):
    """
    读取 utg.json 格式：

    {
      "N": [
        {
          "id": "screen_25.png",
          "name": "DashboardActivity",
          "type": "screen",
          "props": {...}
        }
      ],
      "E": [
        {
          "id": "screen_1.png-->screen_33.png",
          "from": "screen_1.png",
          "to": "screen_33.png",
          "conditions": [...],
          "actions": [...]
        }
      ]
    }

    返回：
    adjacency = {
        "screen_1": {"screen_33"},
        "screen_25": set(),
        ...
    }
    """

    if not utg_path.exists():
        raise FileNotFoundError(f"utg.json 不存在: {utg_path}")

    data = json.loads(
        utg_path.read_text(encoding="utf-8", errors="ignore")
    )

    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        nodes = data.get("N", [])
        edges = data.get("E", [])
    elif isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
    else:
        raise ValueError(f"Unsupported utg schema: {utg_path}")

    adjacency = defaultdict(set)

    # 读取节点
    for node in nodes:
        node_id = node.get("id")
        page_name = normalize_page_name_from_node_id(node_id)

        if page_name is None:
            continue

        adjacency[page_name]

    # 读取有向边
    for edge in edges:
        src_id = edge.get("from")
        dst_id = edge.get("to")

        src_page = normalize_page_name_from_node_id(src_id)
        dst_page = normalize_page_name_from_node_id(dst_id)

        if src_page is None or dst_page is None:
            continue

        adjacency[src_page]
        adjacency[dst_page]

        if src_page != dst_page:
            adjacency[src_page].add(dst_page)

    adjacency = {k: set(v) for k, v in adjacency.items()}

    print(f"[UTG] 节点数 = {len(adjacency)}")
    print(f"[UTG] 边数 = {sum(len(v) for v in adjacency.values())}")

    return adjacency


def compute_reachable_pages(adjacency):
    """
    计算每个页面在有向交互图中可达的页面集合。

    USE_TRANSITIVE_REACHABILITY = True:
        screen_1 -> screen_2 -> screen_3
        screen_1 可达 screen_2 和 screen_3

    USE_TRANSITIVE_REACHABILITY = False:
        只保留直接边。
    """

    reachable = {}

    for src in adjacency:
        if not USE_TRANSITIVE_REACHABILITY:
            reachable[src] = set(adjacency.get(src, set()))
            continue

        visited = set()
        queue = deque(adjacency.get(src, set()))

        while queue:
            cur = queue.popleft()

            if cur in visited:
                continue

            visited.add(cur)

            for nxt in adjacency.get(cur, set()):
                if nxt not in visited:
                    queue.append(nxt)

        visited.discard(src)
        reachable[src] = visited

    mode = "transitive" if USE_TRANSITIVE_REACHABILITY else "direct"
    total_pairs = sum(len(v) for v in reachable.values())

    print(f"[UTG] 可达模式 = {mode}")
    print(f"[UTG] 可达页面对数 = {total_pairs}")

    return reachable


def build_all_page_pairs(available_pages):
    """
    构造所有无向页面对（a < b）。
    """
    pages = sorted(available_pages)
    return [(a, b) for a, b in combinations(pages, 2)]


def build_direct_neighbor_pairs(adjacency, available_pages):
    """
    构造“直接相邻页面对”：
    只要存在 src -> dst 的直接边，就认为 (src, dst) 为相邻页面对（无向去重）。
    """
    pairs = set()
    for src, dst_set in adjacency.items():
        if src not in available_pages:
            continue
        for dst in dst_set:
            if dst not in available_pages or src == dst:
                continue
            a, b = sorted([src, dst])
            pairs.add((a, b))
    return pairs


def build_page_image_path_map(app_dir: Path, page_names):
    """
    为 page_name（如 screen_25）解析对应截图路径。
    优先匹配 app_dir 下的同名 .png/.jpg/.jpeg 文件。
    """
    suffixes = [".png", ".jpg", ".jpeg"]
    suffix_set = set(suffixes)
    page_to_image_path = {}

    stem_to_paths = defaultdict(list)
    for p in app_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffix_set:
            continue
        stem_to_paths[p.stem].append(p)

    for page_name in page_names:
        selected = None

        for suffix in suffixes:
            candidate = app_dir / f"{page_name}{suffix}"
            if candidate.exists():
                selected = candidate
                break

        if selected is None and page_name in stem_to_paths:
            selected = sorted(stem_to_paths[page_name], key=lambda x: str(x))[0]

        if selected is not None:
            page_to_image_path[page_name] = selected

    return page_to_image_path


def filter_page_dict_by_valid_pages(page_dict, valid_pages):
    valid_pages = set(valid_pages)
    return {
        page_name: value
        for page_name, value in page_dict.items()
        if page_name in valid_pages
    }


def judge_popup_relation_between_pages(page_a, page_b, page_to_image_path, popup_cache):
    """
    使用 data_process/testPOP.py 的 judge_popup_relation 判断两个页面是否为弹窗关系。
    带缓存，避免重复计算。
    """
    cache_key = tuple(sorted((page_a, page_b)))
    if cache_key in popup_cache:
        return popup_cache[cache_key]

    path_a = page_to_image_path.get(page_a)
    path_b = page_to_image_path.get(page_b)

    if path_a is None or path_b is None:
        out = {
            "checked": False,
            "is_popup_relation": False,
            "reason": "missing_image_path",
            "popup_type": "none",
            "valid_candidate_count": 0,
        }
        popup_cache[cache_key] = out
        return out

    try:
        popup_result = judge_popup_relation(str(path_a), str(path_b))
        out = {
            "checked": True,
            "is_popup_relation": bool(popup_result.get("same_page", False)),
            "reason": str(popup_result.get("reason", "")),
            "popup_type": str(popup_result.get("popup_type", "none")),
            "valid_candidate_count": int(popup_result.get("valid_candidate_count", 0)),
        }
    except Exception as e:
        out = {
            "checked": False,
            "is_popup_relation": False,
            "reason": f"popup_judge_error: {e}",
            "popup_type": "none",
            "valid_candidate_count": 0,
        }

    popup_cache[cache_key] = out
    return out


# ============================================================
# 7. 解析组件分析文件
# ============================================================

COMPONENT_PATTERN = re.compile(
        r"(?P<keyword>.*?)\n"  # 标题
        r"(?P<function>.*?)\n"  # 描述
        r"@@@(?P<ctype>[^@]+)@@@\n"  # 类型行
        r"\((?P<bbox>[\d\.,\s-]+)\)",# 坐标行，可选结尾换行符
        re.DOTALL
    )


def parse_bbox(bbox_str: str):
    parts = [p.strip() for p in bbox_str.split(",")]

    if len(parts) != 4:
        raise ValueError(f"Invalid bbox: {bbox_str}")

    bbox = tuple(float(p) for p in parts)

    return bbox


def is_normalized_bbox(bbox):
    """
    可选检查：确认 bbox 是否是 0~1 之间的归一化坐标。
    """
    return all(0.0 <= v <= 1.0 for v in bbox)


def parse_component_file(file_path: Path):
    """
    解析一个 screen_xxx_type.txt 文件。
    """

    text = file_path.read_text(encoding="utf-8", errors="ignore")

    page_name = file_path.name

    if page_name.endswith("_type.txt"):
        page_name = page_name[: -len("_type.txt")]

    components = []

    for idx, match in enumerate(COMPONENT_PATTERN.finditer(text)):
        keyword = match.group("keyword").strip()
        function = match.group("function").strip()
        ctype = match.group("ctype").strip()
        bbox_str = match.group("bbox").strip()

        try:
            bbox = parse_bbox(bbox_str)
        except Exception as e:
            print(
                f"[WARN] bbox 解析失败，跳过组件: "
                f"{file_path.name}, idx={idx}, bbox={bbox_str}, error={e}"
            )
            continue

        if not is_normalized_bbox(bbox):
            print(
                f"[WARN] bbox 不是归一化坐标，仍继续使用: "
                f"{file_path.name}, idx={idx}, bbox={bbox}"
            )

        comp = {
            "page_name": page_name,
            "file_path": str(file_path),
            "local_index": idx,
            "keyword": keyword,
            "function": function,
            "type": ctype,
            "type_group": normalize_type(ctype),
            "bbox": bbox,
        }

        components.append(comp)

    return components


def load_all_pages(component_dir: Path):
    """
    读取目录下所有 *_type.txt 文件。
    """

    page_to_components = {}

    files = sorted(component_dir.glob("*_type.txt"))

    print(f"[INFO] 找到 {len(files)} 个 *_type.txt 文件")

    for file_path in files:
        page_name = file_path.name[: -len("_type.txt")]
        components = parse_component_file(file_path)

        page_to_components[page_name] = components

        print(f"[INFO] {page_name}: 原始组件数 = {len(components)}")

    return page_to_components


# ============================================================
# 8. 过滤通用组件
# ============================================================

def filter_components(components):
    kept = []
    removed = []

    for comp in components:
        if is_generic_component(comp):
            removed.append(comp)
        else:
            kept.append(comp)

    return kept, removed


def filter_all_pages(page_to_components):
    page_to_core_components = {}
    page_to_removed_components = {}

    for page_name, components in page_to_components.items():
        core, removed = filter_components(components)

        page_to_core_components[page_name] = core
        page_to_removed_components[page_name] = removed

        print(
            f"[FILTER] {page_name}: "
            f"保留核心组件 = {len(core)}, "
            f"过滤通用组件 = {len(removed)}"
        )

    return page_to_core_components, page_to_removed_components


# ============================================================
# 9. BBOX IoU
# ============================================================

def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def bbox_iou(box1, box2):
    """
    BBOX 已经是归一化坐标，直接计算 IoU。
    """

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)
    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def bbox_overlap(comp1, comp2):
    """
    使用统一 IoU 阈值：
    IoU >= 0.80
    """

    iou = bbox_iou(comp1["bbox"], comp2["bbox"])

    return iou >= IOU_THRESHOLD, iou, IOU_THRESHOLD


# ============================================================
# 10. Embedding 语义相似度
# ============================================================

def build_component_text(comp):
    """
    构造用于 embedding 的组件语义文本。
    """

    return (
        "Component keyword: " + comp.get("keyword", "") + ". "
        "Function description: " + comp.get("function", "") + ". "
    )


def attach_embeddings(page_to_core_components, model):
    """
    对所有核心组件统一计算 embedding。
    """

    all_components = []
    all_texts = []

    for page_name, components in page_to_core_components.items():
        for comp in components:
            all_components.append(comp)
            all_texts.append(build_component_text(comp))

    print(f"[EMBED] 需要计算 embedding 的核心组件数 = {len(all_components)}")

    if not all_components:
        return

    embeddings = model.encode(
        all_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    for comp, emb in zip(all_components, embeddings):
        comp["embedding"] = emb.astype(np.float32)


def embedding_similarity(comp1, comp2):
    """
    embedding 已经 normalize，所以点积就是 cosine similarity。
    """

    emb1 = comp1.get("embedding", None)
    emb2 = comp2.get("embedding", None)

    if emb1 is None or emb2 is None:
        return 0.0

    return float(np.dot(emb1, emb2))


# ============================================================
# 11. 判断两个组件是否相同
# ============================================================

def same_component(comp1, comp2):
    """
    SameComponent(c1, c2)
    =
    SameType(c1, c2)
    AND
    IoU(BBox1, BBox2) >= 0.80
    AND
    EmbeddingSimilarity(Text1, Text2) >= 0.90
    """

    # 规则 1：组件大类相同
    if comp1["type_group"] != comp2["type_group"]:
        return False, {
            "reason": "different_type",
            "type_group_1": comp1["type_group"],
            "type_group_2": comp2["type_group"],
        }

    # 规则 2：BBOX IoU
    is_overlap, iou, iou_threshold = bbox_overlap(comp1, comp2)

    if not is_overlap:
        return False, {
            "reason": "bbox_not_overlap",
            "iou": iou,
            "iou_threshold": iou_threshold,
            "type_group": comp1["type_group"],
        }

    # 规则 3：Embedding 语义相似
    sim = embedding_similarity(comp1, comp2)

    if sim < SEMANTIC_THRESHOLD:
        return False, {
            "reason": "semantic_not_similar",
            "embedding_similarity": sim,
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "iou": iou,
            "iou_threshold": iou_threshold,
            "type_group": comp1["type_group"],
        }

    return True, {
        "reason": "same_component",
        "embedding_similarity": sim,
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "iou": iou,
        "iou_threshold": iou_threshold,
        "type_group": comp1["type_group"],
    }


# ============================================================
# 12. 页面之间匹配组件
# ============================================================

def match_two_pages(page_a, comps_a, page_b, comps_b):
    """
    匹配两个页面之间的相同组件。

    greedy best match：
    对 page_a 的每个组件，在 page_b 中找还没使用过的、
    满足 same_component 且 embedding 相似度最高的组件。
    """

    matched_pairs = []
    used_b = set()

    for i, comp_a in enumerate(comps_a):
        best_j = None
        best_info = None
        best_score = -1.0

        for j, comp_b in enumerate(comps_b):
            if j in used_b:
                continue

            same, info = same_component(comp_a, comp_b)

            if not same:
                continue

            score = info.get("embedding_similarity", 0.0)

            if score > best_score:
                best_score = score
                best_j = j
                best_info = info

        if best_j is not None:
            used_b.add(best_j)
            comp_b = comps_b[best_j]

            matched_pairs.append({
                "page_a": page_a,
                "page_b": page_b,
                "a_index": comp_a["local_index"],
                "b_index": comp_b["local_index"],
                "a_keyword": comp_a["keyword"],
                "b_keyword": comp_b["keyword"],
                "a_function": comp_a["function"],
                "b_function": comp_b["function"],
                "a_type": comp_a["type"],
                "b_type": comp_b["type"],
                "type_group": comp_a["type_group"],
                "a_bbox": comp_a["bbox"],
                "b_bbox": comp_b["bbox"],
                "iou": best_info["iou"],
                "embedding_similarity": best_info["embedding_similarity"],
            })

    return matched_pairs


# ============================================================
# 13. Union-Find 聚类
# ============================================================

class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def groups(self):
        result = defaultdict(list)

        for item in self.parent:
            root = self.find(item)
            result[root].append(item)

        return list(result.values())


# ============================================================
# 14. 基于 UTG 可达关系构造比较页面对
# ============================================================

def build_reachable_page_pairs(reachable, available_pages):
    """
    只保留：
    1. src 有组件分析文件；
    2. dst 有组件分析文件；
    3. src != dst。

    有向图可达关系有方向，
    但聚类是无向合并，所以比较 pair 会转换成排序后的无向对。
    """

    pairs = set()
    missing_pages = set()

    for src, dst_set in reachable.items():
        if src not in available_pages:
            missing_pages.add(src)
            continue

        for dst in dst_set:
            if dst not in available_pages:
                missing_pages.add(dst)
                continue

            if src == dst:
                continue

            a, b = sorted([src, dst])
            pairs.add((a, b))

    if missing_pages:
        print(f"[WARN] UTG 中有 {len(missing_pages)} 个页面没有对应的 *_type.txt 文件")
        for name in sorted(missing_pages)[:20]:
            print(f"       missing: {name}_type.txt")
        if len(missing_pages) > 20:
            print("       ...")

    return sorted(pairs)


def cluster_pages_by_utg_reachable_components(
    page_to_core_components,
    reachable,
    adjacency,
    app_dir: Path,
    page_to_image_path=None,
):
    page_names = sorted(page_to_core_components.keys())
    available_pages = set(page_names)

    compare_pairs = build_all_page_pairs(available_pages)
    direct_neighbor_pairs = build_direct_neighbor_pairs(adjacency, available_pages)
    if page_to_image_path is None:
        page_to_image_path = build_page_image_path_map(app_dir, page_names)
    popup_cache = {}

    print(f"[COMPARE] 页面两两比较对数（全量） = {len(compare_pairs)}")
    print(f"[COMPARE] 直接相邻页面对数（可做弹窗判断） = {len(direct_neighbor_pairs)}")
    print(f"[POPUP] 可用于弹窗判断的页面截图数 = {len(page_to_image_path)}")

    uf = UnionFind(page_names)
    pair_matches = {}
    pair_connection_reason = {}
    connected_pairs = 0

    for page_a, page_b in compare_pairs:
        comps_a = page_to_core_components.get(page_a, [])
        comps_b = page_to_core_components.get(page_b, [])

        matched_pairs = match_two_pages(page_a, comps_a, page_b, comps_b)
        same_component_relation = len(matched_pairs) >= MIN_SHARED_COMPONENTS

        popup_info = {
            "checked": False,
            "is_popup_relation": False,
            "reason": "not_direct_neighbor",
            "popup_type": "none",
            "valid_candidate_count": 0,
        }

        if same_component_relation:
            popup_info["reason"] = "skip_popup_due_to_component_match"
        elif (page_a, page_b) in direct_neighbor_pairs:
            popup_info = judge_popup_relation_between_pages(
                page_a,
                page_b,
                page_to_image_path,
                popup_cache,
            )

        popup_relation = bool(popup_info.get("is_popup_relation", False))

        if same_component_relation or popup_relation:
            uf.union(page_a, page_b)
            connected_pairs += 1

            pair_key = f"{page_a} <-> {page_b}"
            pair_matches[pair_key] = matched_pairs
            pair_connection_reason[pair_key] = {
                "same_component_relation": same_component_relation,
                "shared_component_count": len(matched_pairs),
                "popup_relation": popup_relation,
                "popup_checked": bool(popup_info.get("checked", False)),
                "popup_reason": popup_info.get("reason", ""),
                "popup_type": popup_info.get("popup_type", "none"),
            }

            print(
                f"[MATCH] {page_a} <-> {page_b}: "
                f"相同组件数 = {len(matched_pairs)}, "
                f"popup_relation = {popup_relation}"
            )

    clusters = uf.groups()

    clusters = [sorted(cluster) for cluster in clusters]
    clusters.sort(key=lambda x: (len(x), x[0]), reverse=True)

    print(f"[CLUSTER] 满足“组件相同或弹窗关系”的页面对数 = {connected_pairs}")
    print(f"[CLUSTER] 聚类数量 = {len(clusters)}")

    return clusters, pair_matches, compare_pairs, pair_connection_reason


# ============================================================
# 15. JSON 保存工具
# ============================================================

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, set):
        return sorted(list(obj))

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)

    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)

    return obj


def save_results(
    component_dir,
    utg_json_path,
    output_path,
    clusters,
    pair_matches,
    compare_pairs,
    pair_connection_reason,
    reachable,
    page_to_core_components,
    page_to_removed_components,
):
    result = {
        "component_dir": str(component_dir),
        "utg_json_path": str(utg_json_path),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "iou_threshold": IOU_THRESHOLD,
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "min_shared_components": MIN_SHARED_COMPONENTS,
        "use_transitive_reachability": USE_TRANSITIVE_REACHABILITY,
        "cluster_union_rule": "same_component_relation OR popup_relation_on_direct_neighbor",
        "top_region_threshold": TOP_REGION_THRESHOLD,
        "filter_rule": {
            "description_contains": GENERIC_DESCRIPTION_WORDS,
            "keyword_contains": GENERIC_KEYWORD_WORDS,
            "top_icon_filter": True,
            "top_region": f"component center y < {TOP_REGION_THRESHOLD}",
        },
        "compare_pairs": [
            {
                "page_a": a,
                "page_b": b,
            }
            for a, b in compare_pairs
        ],
        "clusters": [
            {
                "cluster_id": idx,
                "size": len(cluster),
                "pages": cluster,
            }
            for idx, cluster in enumerate(clusters)
        ],
        "pair_matches": pair_matches,
        "pair_connection_reason": pair_connection_reason,
        "reachable": {
            page: sorted(list(dst_set))
            for page, dst_set in reachable.items()
        },
        "page_component_stats": {
            page_name: {
                "core_component_count": len(page_to_core_components.get(page_name, [])),
                "filtered_generic_component_count": len(page_to_removed_components.get(page_name, [])),
                "filtered_generic_components": [
                    {
                        "index": comp["local_index"],
                        "keyword": comp["keyword"],
                        "function": comp["function"],
                        "type": comp["type"],
                        "type_group": comp["type_group"],
                        "bbox": comp["bbox"],
                    }
                    for comp in page_to_removed_components.get(page_name, [])
                ],
            }
            for page_name in sorted(page_to_core_components.keys())
        },
    }

    result = make_json_safe(result)

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[SAVE] 聚类结果已保存到: {output_path}")


# ============================================================
# 16. 主函数
# ============================================================

def _detect_utg_schema(data):
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    raise ValueError("Unsupported UTG schema: expected N/E or nodes/edges")


def save_utg_component_graph(utg_json_path: Path, clusters, output_path: Path):
    """
    Save utg_component graph:
    - Keep original UTG structure and edges
    - Add node 'page' based on component clusters
    - Do NOT modify original utg.json
    """
    data = json.loads(utg_json_path.read_text(encoding="utf-8", errors="ignore"))
    n_key, _ = _detect_utg_schema(data)

    page_to_node_name = {}
    for node in data.get(n_key, []):
        node_id = node.get("id", "")
        page_name = normalize_page_name_from_node_id(node_id)
        if page_name is None:
            continue
        node_name = str(node.get("name", "")).strip()
        if node_name:
            page_to_node_name[page_name] = node_name

    page_to_cluster = {}
    page_name_used_count = defaultdict(int)

    for cluster_idx, cluster_pages in enumerate(clusters):
        candidate_names = [
            page_to_node_name[p]
            for p in cluster_pages
            if p in page_to_node_name
        ]

        if candidate_names:
            name_counter = Counter(candidate_names)
            # 先按出现次数降序，再按名字字典序，保证结果稳定
            base_page_name = sorted(
                name_counter.items(),
                key=lambda x: (-x[1], x[0])
            )[0][0]
        else:
            base_page_name = f"page_{cluster_idx + 1}"

        page_name_used_count[base_page_name] += 1
        page_value = f"{base_page_name}{page_name_used_count[base_page_name]}"

        for page_name in cluster_pages:
            page_to_cluster[page_name] = page_value

    for node in data.get(n_key, []):
        node_id = node.get("id", "")
        page_name = normalize_page_name_from_node_id(node_id)
        if page_name in page_to_cluster:
            node["page"] = page_to_cluster[page_name]
        else:
            node["page"] = "unclustered"

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[SAVE] utg_component saved to: {output_path}")

def iter_apps(data_root: Path):
    for category_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        category = category_dir.name
        for app_dir in sorted([p for p in category_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
            app = app_dir.name
            yield category, app, app_dir


def process_one_app(category, app, app_dir: Path, model):
    component_dir = COMPONENT_RESULTS_ROOT / category / app
    utg_json_path = app_dir / "utg.json"

    if not component_dir.exists():
        print(f"[SKIP] missing component dir: {component_dir}")
        return False
    if not utg_json_path.exists():
        print(f"[SKIP] missing utg.json: {utg_json_path}")
        return False

    print(f"\n[APP] {category}/{app}")

    adjacency = load_utg_graph(utg_json_path)
    reachable = compute_reachable_pages(adjacency)
    page_to_components = load_all_pages(component_dir)

    if not page_to_components:
        print("[SKIP] no *_type.txt files")
        return False

    # 仅处理当前 app 下仍有截图文件的页面，过滤掉已删除图片留下的历史组件分析文件。
    page_to_image_path = build_page_image_path_map(app_dir, page_to_components.keys())
    valid_pages = set(page_to_image_path.keys())
    stale_pages = sorted(set(page_to_components.keys()) - valid_pages)
    if stale_pages:
        print(f"[FILTER-IMG] 过滤无截图页面数 = {len(stale_pages)}")
        for name in stale_pages[:20]:
            print(f"       stale_component_file: {name}_type.txt")
        if len(stale_pages) > 20:
            print("       ...")

    page_to_components = filter_page_dict_by_valid_pages(page_to_components, valid_pages)
    if not page_to_components:
        print("[SKIP] no valid pages with image files after filtering")
        return False

    print(f"[FILTER-IMG] 保留有截图页面数 = {len(page_to_components)}")

    page_to_core_components, page_to_removed_components = filter_all_pages(page_to_components)
    attach_embeddings(page_to_core_components, model)

    clusters, pair_matches, compare_pairs, pair_connection_reason = cluster_pages_by_utg_reachable_components(
        page_to_core_components,
        reachable,
        adjacency,
        app_dir,
        page_to_image_path=page_to_image_path,
    )

    output_json = component_dir / OUTPUT_CLUSTER_FILENAME
    save_results(
        component_dir,
        utg_json_path,
        output_json,
        clusters,
        pair_matches,
        compare_pairs,
        pair_connection_reason,
        reachable,
        page_to_core_components,
        page_to_removed_components,
    )

    utg_component_output = app_dir / OUTPUT_UTG_COMPONENT_FILENAME
    save_utg_component_graph(
        utg_json_path,
        clusters,
        utg_component_output,
    )
    return True


def main_batch():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"data root not found: {DATA_ROOT}")
    if not COMPONENT_RESULTS_ROOT.exists():
        raise FileNotFoundError(f"component results root not found: {COMPONENT_RESULTS_ROOT}")

    print(f"[MODEL] load embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total = 0
    ok = 0
    fail = 0

    for category, app, app_dir in iter_apps(DATA_ROOT):
        total += 1
        try:
            done = process_one_app(category, app, app_dir, model)
            if done:
                ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {category}/{app}: {e}")

    print(f"\n[DONE] total_apps={total}, success={ok}, fail={fail}")


if __name__ == "__main__":
    main_batch()
    # model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # process_one_app("com","aestheticmedicine", DATA_ROOT,model)
