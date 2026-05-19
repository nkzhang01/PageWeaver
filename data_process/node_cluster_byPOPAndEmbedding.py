import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from testPOP_2 import judge_popup_relation


# ============================================================
# 1. Paths and parameters
# ============================================================

DATA_ROOT = Path(r"D:\ui2code\InteractionSequence\data")
OUTPUT_CLUSTER_FILENAME = "page_POPAndEmbedding_clusters_by_utg.json"
OUTPUT_UTG_COMPONENT_FILENAME = "utg_POPAndEmbedding.json"

PAGE_SEMANTIC_SIMILARITY_THRESHOLD = 0.85
USE_TRANSITIVE_REACHABILITY = True


# ============================================================
# 2. Common helpers
# ============================================================

def normalize_page_name_from_node_id(node_id):
    if node_id is None:
        return None

    name = os.path.basename(str(node_id).strip())
    name = name.split("?")[0].split("#")[0]
    lower = name.lower()

    if lower.endswith(".png"):
        name = name[:-4]
    elif lower.endswith(".jpg"):
        name = name[:-4]
    elif lower.endswith(".jpeg"):
        name = name[:-5]

    return name


def _detect_utg_schema(data):
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    raise ValueError("Unsupported UTG schema: expected N/E or nodes/edges")


def load_utg_data(utg_path: Path):
    text = utg_path.read_text(encoding="utf-8", errors="ignore")
    marker = "var utg ="
    pos = text.find(marker)
    if pos >= 0:
        text = text[pos + len(marker) :].strip()
    return json.loads(text)


def load_utg_graph(utg_path: Path):
    data = load_utg_data(utg_path)
    n_key, e_key = _detect_utg_schema(data)
    nodes = data.get(n_key, [])
    edges = data.get(e_key, [])

    adjacency = defaultdict(set)
    for node in nodes:
        page_name = normalize_page_name_from_node_id(node.get("id"))
        if page_name is None:
            continue
        adjacency[page_name]

    for edge in edges:
        src = normalize_page_name_from_node_id(edge.get("from"))
        dst = normalize_page_name_from_node_id(edge.get("to"))
        if src is None or dst is None:
            continue
        adjacency[src]
        adjacency[dst]
        if src != dst:
            adjacency[src].add(dst)

    adjacency = {k: set(v) for k, v in adjacency.items()}
    print(f"[UTG] 节点数 = {len(adjacency)}")
    print(f"[UTG] 边数 = {sum(len(v) for v in adjacency.values())}")
    return adjacency


def compute_reachable_pages(adjacency):
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


def build_reachable_page_pairs(reachable, available_pages):
    pairs = set()
    available_pages = set(available_pages)

    for src, dst_set in reachable.items():
        if src not in available_pages:
            continue
        for dst in dst_set:
            if dst not in available_pages or src == dst:
                continue
            a, b = sorted([src, dst])
            pairs.add((a, b))
    return sorted(pairs)


def build_direct_page_pair_set(adjacency, available_pages):
    pairs = set()
    available_pages = set(available_pages)

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

    states_dir = app_dir / "states"
    if states_dir.exists() and states_dir.is_dir():
        for p in states_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in suffix_set:
                continue
            stem_to_paths[p.stem].append(p)

    for page_name in page_names:
        selected = None
        for suffix in suffixes:
            c1 = app_dir / f"{page_name}{suffix}"
            c2 = app_dir / "states" / f"{page_name}{suffix}"
            if c1.exists():
                selected = c1
                break
            if c2.exists():
                selected = c2
                break

        if selected is None and page_name in stem_to_paths:
            selected = sorted(stem_to_paths[page_name], key=lambda x: str(x))[0]

        if selected is not None:
            page_to_image_path[page_name] = selected

    return page_to_image_path


def _normalize_vec(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return None
    return v / n


def load_page_semantic_embeddings(app_dir: Path, page_names):
    emb_dir = app_dir / "embeddings"
    out = {}
    if not emb_dir.exists() or not emb_dir.is_dir():
        print(f"[WARN] embedding dir not found: {emb_dir}")
        return out

    missing = 0
    bad = 0
    for page in sorted(set(page_names)):
        npz_path = emb_dir / f"{page}.npz"
        if not npz_path.exists():
            missing += 1
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
            if "semantic" not in data:
                bad += 1
                continue
            vec = _normalize_vec(data["semantic"])
            if vec is None:
                bad += 1
                continue
            out[page] = vec
        except Exception:
            bad += 1

    print(f"[PAGE-SEM] loaded={len(out)}, missing={missing}, bad={bad}, dir={emb_dir}")
    return out


def page_semantic_similarity(page_a, page_b, page_semantic_map):
    va = page_semantic_map.get(page_a)
    vb = page_semantic_map.get(page_b)
    if va is None or vb is None:
        return None
    return float(np.dot(va, vb))


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
            result[self.find(item)].append(item)
        return list(result.values())


# ============================================================
# 3. Clustering
# ============================================================

def cluster_pages_by_utg_reachable_embedding_and_popup(
    page_names,
    reachable,
    page_semantic_map,
    page_to_image_path,
    direct_adjacent_pairs,
):
    page_names = sorted(set(page_names))
    available_pages = set(page_names)
    compare_pairs = build_reachable_page_pairs(reachable, available_pages)

    print(f"[COMPARE] 受 UTG 可达关系约束后的页面比较对数 = {len(compare_pairs)}")
    print(f"[COMPARE] 直接相邻页面对数（可做弹窗判断） = {len(direct_adjacent_pairs)}")

    uf = UnionFind(page_names)
    pair_matches = {}
    connected_pairs = 0

    for page_a, page_b in compare_pairs:
        page_sem_sim = page_semantic_similarity(page_a, page_b, page_semantic_map)
        sem_pass = (
            (page_sem_sim is not None)
            and (page_sem_sim >= PAGE_SEMANTIC_SIMILARITY_THRESHOLD)
        )

        popup_relation = False
        popup_reason = ""
        popup_type = "none"
        popup_checked = False
        is_adjacent_pair = (page_a, page_b) in direct_adjacent_pairs

        # 先做页面语义相似度，语义不通过时仅对相邻页面对做弹窗判断
        if (not sem_pass) and is_adjacent_pair:
            img_a = page_to_image_path.get(page_a)
            img_b = page_to_image_path.get(page_b)
            if img_a is not None and img_b is not None:
                popup_checked = True
                try:
                    popup_result = judge_popup_relation(str(img_a), str(img_b))
                    popup_relation = bool(popup_result.get("same_page", False))
                    popup_reason = str(popup_result.get("reason", ""))
                    popup_type = str(popup_result.get("popup_type", "none"))
                except Exception as e:
                    popup_reason = f"popup_judge_error: {e}"
                    print(f"[WARN] popup judge failed for {page_a} <-> {page_b}: {e}")

        merge_pass = sem_pass or popup_relation
        if merge_pass:
            uf.union(page_a, page_b)
            connected_pairs += 1

        pair_key = f"{page_a} <-> {page_b}"
        pair_matches[pair_key] = {
            "page_sem_sim": page_sem_sim,
            "page_sem_pass": sem_pass,
            "popup_checked": popup_checked,
            "popup_relation": popup_relation,
            "popup_type": popup_type,
            "popup_reason": popup_reason,
            "merge_by": (
                "embedding_similarity"
                if sem_pass
                else ("popup_relation" if popup_relation else "none")
            ),
            "same_page": merge_pass,
        }

        if merge_pass:
            print(
                f"[MATCH] {page_a} <-> {page_b}: "
                f"page_sem_sim = {page_sem_sim if page_sem_sim is not None else 'None'}, "
                f"popup_relation = {popup_relation}, "
                f"merge_by = {pair_matches[pair_key]['merge_by']}"
            )

    clusters = uf.groups()
    clusters = [sorted(c) for c in clusters]
    clusters.sort(key=lambda x: (len(x), x[0]), reverse=True)

    print(f"[CLUSTER] 有效同页关系页面对数 = {connected_pairs}")
    print(f"[CLUSTER] 聚类数量 = {len(clusters)}")
    return clusters, pair_matches, compare_pairs


# ============================================================
# 4. Save outputs
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
    app_dir,
    utg_json_path,
    output_path,
    clusters,
    pair_matches,
    compare_pairs,
    reachable,
):
    result = {
        "app_dir": str(app_dir),
        "utg_json_path": str(utg_json_path),
        "page_semantic_similarity_threshold": PAGE_SEMANTIC_SIMILARITY_THRESHOLD,
        "use_transitive_reachability": USE_TRANSITIVE_REACHABILITY,
        "compare_pairs": [{"page_a": a, "page_b": b} for a, b in compare_pairs],
        "clusters": [
            {
                "cluster_id": idx,
                "size": len(cluster),
                "pages": cluster,
            }
            for idx, cluster in enumerate(clusters)
        ],
        "pair_matches": pair_matches,
        "reachable": {page: sorted(list(dst_set)) for page, dst_set in reachable.items()},
    }

    output_path.write_text(
        json.dumps(make_json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[SAVE] 聚类结果已保存到: {output_path}")


def save_utg_component_graph(utg_json_path: Path, clusters, output_path: Path):
    data = load_utg_data(utg_json_path)
    n_key, _ = _detect_utg_schema(data)

    page_to_node_name = {}
    for node in data.get(n_key, []):
        page_name = normalize_page_name_from_node_id(node.get("id", ""))
        if page_name is None:
            continue
        node_name = str(node.get("name", "")).strip()
        if node_name:
            page_to_node_name[page_name] = node_name

    page_to_cluster = {}
    page_name_used_count = defaultdict(int)

    for cluster_idx, cluster_pages in enumerate(clusters):
        candidate_names = [page_to_node_name[p] for p in cluster_pages if p in page_to_node_name]
        if candidate_names:
            name_counter = Counter(candidate_names)
            base_page_name = sorted(name_counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            base_page_name = f"page_{cluster_idx + 1}"

        page_name_used_count[base_page_name] += 1
        page_value = f"{base_page_name}{page_name_used_count[base_page_name]}"
        for page_name in cluster_pages:
            page_to_cluster[page_name] = page_value

    for node in data.get(n_key, []):
        page_name = normalize_page_name_from_node_id(node.get("id", ""))
        if page_name in page_to_cluster:
            node["page"] = page_to_cluster[page_name]
        else:
            node["page"] = "unclustered"

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[SAVE] utg_component saved to: {output_path}")


# ============================================================
# 5. Batch process
# ============================================================

def detect_utg_path(app_dir: Path):
    for name in ("utg.json", "utg.js"):
        p = app_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def iter_apps(data_root: Path):
    # Mode A: single app directory
    root_utg = detect_utg_path(data_root)
    if root_utg is not None:
        category = data_root.parent.name if data_root.parent else "single_app"
        yield category, data_root.name, data_root, root_utg
        return

    # Mode B: data_root/category/app
    for category_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        category = category_dir.name
        found_direct_app = False

        # data_root maybe already one category directory (children are apps)
        for app_dir in sorted([p for p in category_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
            utg_path = detect_utg_path(app_dir)
            if utg_path is None:
                continue
            found_direct_app = True
            yield category, app_dir.name, app_dir, utg_path

        # Mode C fallback for data_root/app
        if not found_direct_app:
            utg_path = detect_utg_path(category_dir)
            if utg_path is not None:
                yield data_root.name, category_dir.name, category_dir, utg_path


def process_one_app(category, app, app_dir: Path, utg_path: Path):
    print(f"\n[APP] {category}/{app}")

    adjacency = load_utg_graph(utg_path)
    reachable = compute_reachable_pages(adjacency)

    page_names = sorted(adjacency.keys())
    if not page_names:
        print("[SKIP] no page nodes in UTG")
        return False

    page_to_image_path = build_page_image_path_map(app_dir, page_names)
    page_semantic_map = load_page_semantic_embeddings(app_dir, page_names)
    direct_adjacent_pairs = build_direct_page_pair_set(adjacency, page_names)

    clusters, pair_matches, compare_pairs = cluster_pages_by_utg_reachable_embedding_and_popup(
        page_names=page_names,
        reachable=reachable,
        page_semantic_map=page_semantic_map,
        page_to_image_path=page_to_image_path,
        direct_adjacent_pairs=direct_adjacent_pairs,
    )

    # 按要求：输出 page_POPAndEmbedding_clusters_by_utg.json 到每个 APP 目录
    output_json = app_dir / OUTPUT_CLUSTER_FILENAME
    save_results(
        app_dir=app_dir,
        utg_json_path=utg_path,
        output_path=output_json,
        clusters=clusters,
        pair_matches=pair_matches,
        compare_pairs=compare_pairs,
        reachable=reachable,
    )

    utg_component_output = app_dir / OUTPUT_UTG_COMPONENT_FILENAME
    save_utg_component_graph(utg_path, clusters, utg_component_output)
    return True


def main_batch():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"data root not found: {DATA_ROOT}")

    total = 0
    ok = 0
    fail = 0

    for category, app, app_dir, utg_path in iter_apps(DATA_ROOT):
        total += 1
        try:
            done = process_one_app(category, app, app_dir, utg_path)
            if done:
                ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {category}/{app}: {e}")

    print(f"\n[DONE] total_apps={total}, success={ok}, fail={fail}")


if __name__ == "__main__":
    main_batch()
