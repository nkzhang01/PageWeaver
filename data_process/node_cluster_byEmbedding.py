import copy
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


class UIClusterEngine:
    def __init__(self, k=5, vector_key="combined"):
        self.k = k
        self.vector_key = vector_key  # "semantic" / "layout" / "combined"

    def _load_embeddings(self, app_dir: Path):
        emb_dir = app_dir / "embeddings"
        if not emb_dir.exists():
            raise FileNotFoundError(f"Embedding directory not found: {emb_dir}")

        npz_files = sorted([p for p in emb_dir.glob("*.npz") if p.name != "index.npz"])
        if not npz_files:
            raise FileNotFoundError(f"No per-image npz files found in: {emb_dir}")

        image_names = []
        features = []

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                if self.vector_key not in data:
                    raise KeyError(f"{npz_path} has no key '{self.vector_key}'")

                vec = data[self.vector_key].astype(np.float32).reshape(-1)
                if vec.size == 0:
                    continue

                if "image_name" in data:
                    image_name = str(data["image_name"].item()).strip()
                elif "image_path" in data:
                    image_name = Path(str(data["image_path"].item())).name
                else:
                    image_name = f"{npz_path.stem}.png"

                if not image_name:
                    continue

                image_names.append(image_name)
                features.append(vec)

        if not features:
            raise ValueError("No valid embeddings loaded.")

        return image_names, np.stack(features, axis=0)

    def cluster_image_names(self, app_dir: Path):
        image_names, features = self._load_embeddings(app_dir)
        print(
            f"Loaded {len(image_names)} embeddings from {app_dir / 'embeddings'} "
            f"(vector_key='{self.vector_key}', k={self.k})."
        )

        sim_matrix = np.dot(features, features.T)
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, : self.k + 1]

        num_imgs = features.shape[0]
        adj = np.zeros((num_imgs, num_imgs), dtype=np.uint8)

        for i in range(num_imgs):
            for j in top_k_indices[i]:
                if i in top_k_indices[j]:
                    adj[i, j] = 1
                    adj[j, i] = 1

        visited = [False] * num_imgs
        clusters = []
        for i in range(num_imgs):
            if visited[i]:
                continue
            q = deque([i])
            visited[i] = True
            cluster = []
            while q:
                cur = q.popleft()
                cluster.append(image_names[cur])
                neighbors = np.where(adj[cur] == 1)[0]
                for n in neighbors:
                    if not visited[n]:
                        visited[n] = True
                        q.append(n)
            clusters.append(cluster)

        return clusters


def load_utg(utg_path: Path):
    text = utg_path.read_text(encoding="utf-8", errors="ignore")
    marker = "var utg ="
    pos = text.find(marker)
    if pos >= 0:
        text = text[pos + len(marker):].strip()
    return json.loads(text)


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_schema(data):
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    raise ValueError("Unsupported schema: expected N/E or nodes/edges")


def detect_utg_path(app_dir: Path):
    for name in ("utg.json", "utg.js"):
        p = app_dir / name
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f"utg.json/utg.js not found in {app_dir}")


def get_node_id(node):
    return str(node.get("id", "")).strip()


def choose_cluster_base_name(cluster_node_ids, node_map):
    names = []
    for nid in cluster_node_ids:
        node = node_map.get(nid, {})
        n = str(node.get("name", "")).strip()
        if n:
            names.append(n)
    if not names:
        return "unknown_page"
    counts = Counter(names)
    max_cnt = max(counts.values())
    # stable tie-break: first appearance in cluster order
    for n in names:
        if counts[n] == max_cnt:
            return n
    return "unknown_page"


def build_page_names_for_clusters(clusters_node_ids, node_map):
    base_names = [choose_cluster_base_name(c, node_map) for c in clusters_node_ids]
    total_by_base = Counter(base_names)
    running = defaultdict(int)

    cluster_page_names = []
    for base in base_names:
        if total_by_base[base] == 1:
            cluster_page_names.append(base)
            continue
        running[base] += 1
        cluster_page_names.append(f"{base}{running[base]}")
    return cluster_page_names


def annotate_utg_with_cluster_pages(utg_data, image_clusters):
    n_key, _ = detect_schema(utg_data)
    nodes = utg_data.get(n_key, [])
    node_map = {get_node_id(n): n for n in nodes if get_node_id(n)}

    # map cluster image name -> node id (same as UTG id in current dataset)
    clusters_node_ids = []
    for cluster in image_clusters:
        node_ids = []
        seen = set()
        for img_name in cluster:
            nid = str(img_name).strip()
            if nid in node_map and nid not in seen:
                node_ids.append(nid)
                seen.add(nid)
        if node_ids:
            clusters_node_ids.append(node_ids)

    cluster_page_names = build_page_names_for_clusters(clusters_node_ids, node_map)
    node_to_page = {}
    for node_ids, page_name in zip(clusters_node_ids, cluster_page_names):
        for nid in node_ids:
            node_to_page[nid] = page_name

    out = copy.deepcopy(utg_data)
    out_nodes = []
    for n in out.get(n_key, []):
        nid = get_node_id(n)
        nn = copy.deepcopy(n)
        if nid in node_to_page:
            nn["page"] = node_to_page[nid]
        out_nodes.append(nn)
    out[n_key] = out_nodes

    return out, node_to_page


def run_for_app(app_dir: Path, k=5, vector_key="combined"):
    app_dir = app_dir.resolve()
    utg_path = detect_utg_path(app_dir)
    utg_data = load_utg(utg_path)

    engine = UIClusterEngine(k=k, vector_key=vector_key)
    image_clusters = engine.cluster_image_names(app_dir)

    utg_embedding, node_to_page = annotate_utg_with_cluster_pages(utg_data, image_clusters)
    out_path = app_dir / "utg_embedding.json"
    save_json(out_path, utg_embedding)

    print(f"Saved: {out_path}")
    print(f"Clusters: {len(image_clusters)}, nodes_with_page: {len(node_to_page)}")


def collect_app_dirs(data_root: Path):
    app_dirs = []
    data_root = data_root.resolve()
    if not data_root.exists():
        return app_dirs

    # data/<category>/<app>
    for category_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        for app_dir in sorted([p for p in category_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
            if (app_dir / "embeddings").exists():
                app_dirs.append(app_dir)
    return app_dirs


def run_batch_for_data_root(data_root: Path, k=5, vector_key="combined"):
    app_dirs = collect_app_dirs(data_root)
    if not app_dirs:
        print(f"No app dirs found under: {data_root}")
        return

    print(f"Found {len(app_dirs)} apps under: {data_root}")
    ok = 0
    failed = 0

    for idx, app_dir in enumerate(app_dirs, start=1):
        print(f"\n[{idx}/{len(app_dirs)}] Processing: {app_dir}")
        try:
            run_for_app(app_dir, k=k, vector_key=vector_key)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {app_dir}: {e}")

    print(f"\nBatch done. success={ok}, failed={failed}, total={len(app_dirs)}")


if __name__ == "__main__":
    # Single app mode (kept)
    # app_dir = Path(r"D:\ui2code\InteractionSequence\data\com\jiffy")
    # run_for_app(app_dir, k=3)

    # Batch mode: data/<category>/<app>
    data_root = Path(r"D:\ui2code\InteractionSequence\data")
    run_batch_for_data_root(data_root, k=3)
