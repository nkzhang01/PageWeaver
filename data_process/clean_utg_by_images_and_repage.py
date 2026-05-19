import json
import re
from pathlib import Path


DATA_ROOT = Path(r"D:\ui2code\InteractionSequence\data")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_schema(data):
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    raise ValueError("Unsupported schema: expected N/E or nodes/edges")


def get_node_id(node):
    return str(node.get("id", "")).strip()


def get_edge_from_to(edge):
    return str(edge.get("from", "")).strip(), str(edge.get("to", "")).strip()


def _image_candidates(node_id: str):
    p = Path(node_id)
    if p.suffix:
        stem = p.stem
        return [p.name, f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.webp"]
    return [f"{node_id}.png", f"{node_id}.jpg", f"{node_id}.jpeg", f"{node_id}.webp"]


def has_image(app_dir: Path, node_id: str):
    if not node_id:
        return False
    for name in _image_candidates(node_id):
        p1 = app_dir / name
        if p1.exists() and p1.is_file():
            return True
    return False


def natural_key_node_id(node_id: str):
    m = re.search(r"(\d+)", node_id)
    if m:
        return int(m.group(1)), node_id
    return 10**9, node_id


def choose_base_name(component, node_map):
    counts = {}
    first_pos = {}
    for i, nid in enumerate(component):
        node = node_map.get(nid, {})
        base = str(node.get("name", "")).strip() or "page"
        counts[base] = counts.get(base, 0) + 1
        if base not in first_pos:
            first_pos[base] = i
    best = sorted(counts.items(), key=lambda kv: (-kv[1], first_pos[kv[0]], kv[0]))[0][0]
    cleaned = re.sub(r"[^\w]+", "", best) or "page"
    return cleaned


def build_page_map(valid_node_ids, label_nodes, label_edges):
    node_set = set(valid_node_ids)
    adj = {nid: set() for nid in valid_node_ids}

    for e in label_edges:
        a, b = get_edge_from_to(e)
        if a in node_set and b in node_set and bool(e.get("same_page", False)):
            adj[a].add(b)
            adj[b].add(a)

    visited = set()
    components = []
    ordered_nodes = sorted(valid_node_ids, key=natural_key_node_id)
    for nid in ordered_nodes:
        if nid in visited:
            continue
        stack = [nid]
        visited.add(nid)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(adj.get(cur, []), key=natural_key_node_id):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        components.append(sorted(comp, key=natural_key_node_id))

    components.sort(key=lambda c: natural_key_node_id(c[0]) if c else (10**9, ""))

    node_map = {get_node_id(n): n for n in label_nodes}
    base_counters = {}
    page_map = {}
    for comp in components:
        base = choose_base_name(comp, node_map)
        idx = base_counters.get(base, 0) + 1
        base_counters[base] = idx
        page_name = f"{base}{idx}"
        for nid in comp:
            page_map[nid] = page_name
    return page_map


def filter_graph(data, n_key, e_key, keep_nodes):
    keep = set(keep_nodes)
    nodes = [n for n in data.get(n_key, []) if get_node_id(n) in keep]
    edges = []
    for e in data.get(e_key, []):
        a, b = get_edge_from_to(e)
        if a in keep and b in keep:
            edges.append(e)
    data[n_key] = nodes
    data[e_key] = edges
    return data


def remove_page_field(data, n_key):
    for n in data.get(n_key, []):
        if "page" in n:
            n.pop("page", None)
    return data


def remove_same_page_field(data, e_key):
    for e in data.get(e_key, []):
        if "same_page" in e:
            e.pop("same_page", None)
    return data


def apply_page_map(data, n_key, page_map):
    for n in data.get(n_key, []):
        nid = get_node_id(n)
        if nid in page_map:
            n["page"] = page_map[nid]
    return data


def process_app(app_dir: Path):
    label_path = app_dir / "utg_label.json"
    label = load_json(label_path)

    label_n_key, label_e_key = detect_schema(label)

    label_nodes = label.get(label_n_key, [])
    label_ids = {get_node_id(n) for n in label_nodes if get_node_id(n)}
    image_ok_ids = {nid for nid in label_ids if has_image(app_dir, nid)}

    keep_ids = sorted((label_ids & image_ok_ids), key=natural_key_node_id)

    old_label_nodes = len(label_nodes)
    old_label_edges = len(label.get(label_e_key, []))

    # Only operate on utg_label: node/edge deletions by image existence.
    label = filter_graph(label, label_n_key, label_e_key, keep_ids)

    # Persist updated utg_label first.
    save_json(label_path, label)

    # Generate utg.json from utg_label content:
    # remove node "page" and edge "same_page" only.
    utg_out = json.loads(json.dumps(label))
    utg_out = remove_page_field(utg_out, label_n_key)
    utg_out = remove_same_page_field(utg_out, label_e_key)
    utg_path = app_dir / "utg.json"
    save_json(utg_path, utg_out)

    return {
        "app": str(app_dir),
        "label_nodes": (old_label_nodes, len(label.get(label_n_key, []))),
        "label_edges": (old_label_edges, len(label.get(label_e_key, []))),
        "utg_nodes": len(utg_out.get(label_n_key, [])),
        "utg_edges": len(utg_out.get(label_e_key, [])),
    }


def find_apps(data_root: Path):
    apps = []
    for p in sorted(data_root.rglob("*")):
        if not p.is_dir():
            continue
        if (p / "utg_label.json").exists():
            apps.append(p)
    return apps


def main():
    apps = find_apps(DATA_ROOT)
    if not apps:
        print(f"No app found under: {DATA_ROOT}")
        return

    ok = 0
    fail = 0
    for app in apps:
        try:
            r = process_app(app)
            ok += 1
            print(
                f"[OK] {r['app']} | "
                f"label N {r['label_nodes'][0]}->{r['label_nodes'][1]}, "
                f"label E {r['label_edges'][0]}->{r['label_edges'][1]}, "
                f"utg N={r['utg_nodes']}, utg E={r['utg_edges']}"
            )
        except Exception as e:
            fail += 1
            print(f"[FAIL] {app}: {e}")

    print(f"Done. total={len(apps)}, success={ok}, fail={fail}")


if __name__ == "__main__":
    main()
