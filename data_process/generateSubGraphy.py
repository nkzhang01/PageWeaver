import copy
import json
import re
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def detect_schema(data):
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    raise ValueError("Unsupported UTG schema: expected N/E or nodes/edges")


def get_node_id(node):
    return str(node.get("id", "")).strip()


def get_edge_endpoints(edge):
    return str(edge.get("from", "")).strip(), str(edge.get("to", "")).strip()


def get_node_page(node):
    page = str(node.get("page", node.get("page_type", ""))).strip()
    return page if page else "UNKNOWN_PAGE"


def marker_base_from_node(node):
    props = node.get("props") if isinstance(node, dict) else {}
    if not isinstance(props, dict):
        props = {}
    content = str(props.get("content", "")).strip()
    head = content.split("\n", 1)[0].strip() if content else ""
    return head or "mode"


def sanitize_filename(name):
    s = str(name or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unknown_page"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def reachable_candidate_count(src, candidate_set, adj):
    visited = set([src])
    stack = [src]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):
            if nxt in visited:
                continue
            visited.add(nxt)
            stack.append(nxt)
    visited.discard(src)
    return len([nid for nid in visited if nid in candidate_set])


def pick_home_node_for_page(page_node_ids, edges, node_to_page, page, node_order_idx):
    page_set = set(page_node_ids)

    # 候选：有其他页面节点可以跳到它（跨页面入边）
    candidates = []
    candidate_set = set()
    for e in edges:
        a, b = get_edge_endpoints(e)
        if b not in page_set or not a:
            continue
        src_page = node_to_page.get(a, "UNKNOWN_PAGE")
        if src_page != page:
            if b not in candidate_set:
                candidate_set.add(b)
                candidates.append(b)

    # 如果没有候选，退化为 page 内第一个节点
    if not candidates:
        return page_node_ids[0], []

    if len(candidates) == 1:
        return candidates[0], candidates

    # 候选多个：在 page 子图里选“可达更多其他候选节点”的作为首页
    adj = {nid: set() for nid in page_node_ids}
    for e in edges:
        a, b = get_edge_endpoints(e)
        if a in page_set and b in page_set and a != b:
            adj[a].add(b)

    scored = []
    cset = set(candidates)
    for nid in candidates:
        cnt = reachable_candidate_count(nid, cset, adj)
        # tie-break: 在原节点顺序更靠前
        scored.append((cnt, -node_order_idx.get(nid, 10**9), nid))
    scored.sort(reverse=True)
    return scored[0][2], candidates


def build_node_with_mode(node_obj, mode_value):
    n = copy.deepcopy(node_obj)
    n["mode"] = str(mode_value or "")
    return n


def build_outputs_for_one_app(app_dir: Path, utg_component_path: Path):
    data = load_json(utg_component_path)
    n_key, e_key = detect_schema(data)
    nodes = data.get(n_key, [])
    edges = data.get(e_key, [])

    node_by_id = {}
    node_order = []
    node_to_page = {}
    page_to_node_ids = {}
    node_order_idx = {}

    for idx, node in enumerate(nodes):
        nid = get_node_id(node)
        if not nid:
            continue
        page = get_node_page(node)
        node_by_id[nid] = node
        node_to_page[nid] = page
        node_order.append(nid)
        node_order_idx[nid] = idx
        page_to_node_ids.setdefault(page, []).append(nid)

    # 先确定每个 page 的首页 + mode，保证后续所有图一致
    node_mode_map = {nid: "" for nid in node_by_id.keys()}
    page_home_map = {}
    page_candidates_map = {}
    page_ordered_node_ids = {}

    for page, page_nodes in page_to_node_ids.items():
        if not page_nodes:
            continue
        home_node, candidates = pick_home_node_for_page(
            page_node_ids=page_nodes,
            edges=edges,
            node_to_page=node_to_page,
            page=page,
            node_order_idx=node_order_idx,
        )
        page_home_map[page] = home_node
        page_candidates_map[page] = candidates

        ordered = [home_node] + [nid for nid in page_nodes if nid != home_node]
        page_ordered_node_ids[page] = ordered

        # 首页
        node_mode_map[home_node] = "BASE"

        # 其他首页候选：mode = props.content 第一行 + 顺序数字
        seq = 1
        for nid in candidates:
            if nid == home_node:
                continue
            base = marker_base_from_node(node_by_id.get(nid, {}))
            node_mode_map[nid] = f"{base}{seq}"
            seq += 1

    out_root = app_dir / "page_subgraphs"
    state_dir = out_root / "state"
    transition_dir = out_root / "transition"
    state_dir.mkdir(parents=True, exist_ok=True)
    transition_dir.mkdir(parents=True, exist_ok=True)

    state_items = []
    transition_items = []

    # 1) 先生成每个 page 的页面状态子图
    for page, ordered_node_ids in page_ordered_node_ids.items():
        page_set = set(ordered_node_ids)
        sub_nodes = [
            build_node_with_mode(node_by_id[nid], node_mode_map.get(nid, ""))
            for nid in ordered_node_ids
            if nid in node_by_id
        ]

        sub_edges = []
        for e in edges:
            a, b = get_edge_endpoints(e)
            if a in page_set and b in page_set:
                sub_edges.append(copy.deepcopy(e))

        graph = {n_key: sub_nodes, e_key: sub_edges}
        page_file_id = sanitize_filename(page)
        graph_path = state_dir / f"{page_file_id}.json"
        write_json(graph_path, graph)

        state_items.append(
            {
                "page": page,
                "home_node_id": page_home_map.get(page, ""),
                "home_candidates": page_candidates_map.get(page, []),
                "node_count": len(sub_nodes),
                "edge_count": len(sub_edges),
                "graph_path": str(graph_path),
            }
        )

    # 2) 再生成每个 page 的页面跳转图（从该 page 到其他 page 的跨页面边）
    for page, ordered_node_ids in page_ordered_node_ids.items():
        page_set = set(ordered_node_ids)

        trans_edges = []
        for e in edges:
            a, b = get_edge_endpoints(e)
            if a in page_set and b in node_to_page and node_to_page.get(b) != page:
                trans_edges.append(copy.deepcopy(e))

        # 节点：当前 page 中参与跳转的源节点 + 被跳转到的其他 page 节点
        src_ids = unique_keep_order([get_edge_endpoints(e)[0] for e in trans_edges])
        dst_ids = unique_keep_order([get_edge_endpoints(e)[1] for e in trans_edges])
        node_ids = unique_keep_order(src_ids + dst_ids)

        trans_nodes = [
            build_node_with_mode(node_by_id[nid], node_mode_map.get(nid, ""))
            for nid in node_ids
            if nid in node_by_id
        ]

        trans_graph = {n_key: trans_nodes, e_key: trans_edges}
        page_file_id = sanitize_filename(page)
        graph_path = transition_dir / f"{page_file_id}_transition.json"
        write_json(graph_path, trans_graph)

        transition_items.append(
            {
                "page": page,
                "home_node_id": page_home_map.get(page, ""),
                "node_count": len(trans_nodes),
                "edge_count": len(trans_edges),
                "graph_path": str(graph_path),
            }
        )

    write_json(
        out_root / "page_state_subgraphs.json",
        {
            "app_dir": str(app_dir),
            "source_utg_component": str(utg_component_path),
            "schema": {"node_key": n_key, "edge_key": e_key},
            "page_count": len(page_ordered_node_ids),
            "node_mode_map": node_mode_map,
            "items": state_items,
        },
    )

    write_json(
        out_root / "page_transition_graphs.json",
        {
            "app_dir": str(app_dir),
            "source_utg_component": str(utg_component_path),
            "schema": {"node_key": n_key, "edge_key": e_key},
            "page_count": len(page_ordered_node_ids),
            "node_mode_map": node_mode_map,
            "items": transition_items,
        },
    )

    return {
        "page_count": len(page_ordered_node_ids),
        "state_graph_count": len(state_items),
        "transition_graph_count": len(transition_items),
        "out_root": str(out_root),
    }


def find_app_entries(data_root: Path):
    entries = []
    seen = set()
    for pattern in ("utg_component.json", "utg_ component.json"):
        for p in sorted(data_root.rglob(pattern)):
            if not p.is_file():
                continue
            app_dir = p.parent
            key = str(app_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            entries.append((app_dir, p))
    return sorted(entries, key=lambda x: str(x[0]))


def main():
    data_root = Path(__file__).resolve().parents[1] / "data"
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")

    app_entries = find_app_entries(data_root)
    if not app_entries:
        print(f"No utg_component.json found under: {data_root}")
        return

    ok_count = 0
    fail_count = 0

    for app_dir, utg_component_path in app_entries:
        rel = str(app_dir.relative_to(data_root))
        try:
            result = build_outputs_for_one_app(app_dir, utg_component_path)
            ok_count += 1
            print(
                f"[OK] {rel} | page_count={result['page_count']} | "
                f"state_graph_count={result['state_graph_count']} | "
                f"transition_graph_count={result['transition_graph_count']}"
            )
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] {rel}: {e}")

    print(
        f"Done. total={len(app_entries)}, success={ok_count}, fail={fail_count}, data_root={data_root}"
    )


if __name__ == "__main__":
    main()
