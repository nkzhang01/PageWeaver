import copy
import json
import re
from pathlib import Path


CLUSTER_RELATIVE_PATH = Path("Cluster") / "page_cluster.json"
UTG_PROCESS_FILENAME = "UTG.json"
UTG_LABEL_FILENAME = "UTG_Label.json"
DEFAULT_OUTPUT_SUBGRAPH_DIR_NAME = "page_subgraphs"
DEFAULT_OUTPUT_PTG_FILENAME = "PTG.json"
GT_OUTPUT_SUBGRAPH_DIR_NAME = "page_subgraphs_GT"
GT_OUTPUT_PTG_FILENAME = "PTG_GT.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def normalize_mockup_id(value):
    text = str(value or "").strip()
    if text.lower().endswith(".png"):
        text = text[:-4]
    return text


def sanitize_filename(name):
    text = str(name or "").strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "unknown_page"


def unique_keep_order(items):
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def build_node(node_obj, page_name=None):
    node = copy.deepcopy(node_obj)
    node.pop("mode", None)
    if page_name is not None:
        node["page"] = page_name
    return node


def load_page_clusters(cluster_path: Path):
    cluster_data = load_json(cluster_path)
    pages = cluster_data.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError(f"Invalid cluster file, pages must be a list: {cluster_path}")

    group_to_mockups = {}
    mockup_to_group = {}

    for index, page in enumerate(pages, start=1):
        states = page.get("states", [])
        if not isinstance(states, list):
            continue

        mockups = []
        for state in states:
            mockup = normalize_mockup_id(state.get("mockup", ""))
            if mockup:
                mockups.append(mockup)

        mockups = unique_keep_order(mockups)
        if not mockups:
            continue

        group_id = str(page.get("page_id", "")).strip() or f"P{index}"
        group_to_mockups[group_id] = mockups
        for mockup in mockups:
            mockup_to_group[mockup] = group_id

    return group_to_mockups, mockup_to_group


def build_node_lookup(nodes):
    node_by_id = {}
    normalized_to_id = {}

    for node in nodes:
        node_id = get_node_id(node)
        if not node_id:
            continue
        node_by_id[node_id] = node
        normalized_to_id[normalize_mockup_id(node_id)] = node_id

    return node_by_id, normalized_to_id


def resolve_group_node_ids(group_to_mockups, normalized_to_id):
    resolved = {}
    for group_id, mockups in group_to_mockups.items():
        node_ids = []
        for mockup in mockups:
            node_id = normalized_to_id.get(normalize_mockup_id(mockup))
            if node_id:
                node_ids.append(node_id)
        resolved[group_id] = unique_keep_order(node_ids)
    return resolved


def resolve_node_to_group(mockup_to_group, normalized_to_id):
    node_to_group = {}
    for mockup, group_id in mockup_to_group.items():
        node_id = normalized_to_id.get(normalize_mockup_id(mockup))
        if node_id:
            node_to_group[node_id] = group_id
    return node_to_group


def reachable_candidate_count(source, candidate_set, adjacency):
    visited = {source}
    stack = [source]
    while stack:
        current = stack.pop()
        for target in adjacency.get(current, ()):
            if target in visited:
                continue
            visited.add(target)
            stack.append(target)
    visited.discard(source)
    return len([node_id for node_id in visited if node_id in candidate_set])


def pick_home_node_for_group(group_node_ids, edges, node_to_group, group_id):
    group_set = set(group_node_ids)
    candidates = []
    seen_candidates = set()

    for edge in edges:
        source, target = get_edge_endpoints(edge)
        if target not in group_set or not source:
            continue
        source_group = node_to_group.get(source)
        if source_group and source_group != group_id and target not in seen_candidates:
            candidates.append(target)
            seen_candidates.add(target)

    if not candidates:
        return group_node_ids[0], []

    if len(candidates) == 1:
        return candidates[0], candidates

    adjacency = {node_id: set() for node_id in group_node_ids}
    for edge in edges:
        source, target = get_edge_endpoints(edge)
        if source in group_set and target in group_set and source != target:
            adjacency[source].add(target)

    order_index = {node_id: index for index, node_id in enumerate(group_node_ids)}
    candidate_set = set(candidates)
    scored = []
    for node_id in candidates:
        reachable_count = reachable_candidate_count(node_id, candidate_set, adjacency)
        scored.append((reachable_count, -order_index.get(node_id, 10**9), node_id))
    scored.sort(reverse=True)
    return scored[0][2], candidates


def build_named_clusters(group_to_node_ids, edges, node_to_group, entry_node_id=None):
    cluster_to_node_ids = {}
    node_to_cluster = {}
    cluster_home_map = {}
    cluster_candidates_map = {}
    entry_node_id = str(entry_node_id or "").strip()

    for group_id, group_node_ids in group_to_node_ids.items():
        if not group_node_ids:
            continue

        home_node, candidates = pick_home_node_for_group(group_node_ids, edges, node_to_group, group_id)
        if entry_node_id and entry_node_id in group_node_ids:
            cluster_name = normalize_mockup_id(entry_node_id)
        else:
            cluster_name = normalize_mockup_id(home_node)
        ordered_node_ids = [home_node] + [node_id for node_id in group_node_ids if node_id != home_node]

        cluster_to_node_ids[cluster_name] = ordered_node_ids
        cluster_home_map[cluster_name] = home_node
        cluster_candidates_map[cluster_name] = candidates
        for node_id in ordered_node_ids:
            node_to_cluster[node_id] = cluster_name

    return cluster_to_node_ids, node_to_cluster, cluster_home_map, cluster_candidates_map


def build_psg(cluster_name, cluster_node_ids, node_by_id, edges, n_key, e_key):
    cluster_set = set(cluster_node_ids)
    sub_nodes = [
        build_node(node_by_id[node_id])
        for node_id in cluster_node_ids
        if node_id in node_by_id
    ]

    sub_edges = []
    for edge in edges:
        source, target = get_edge_endpoints(edge)
        if source in cluster_set and target in cluster_set:
            sub_edges.append(copy.deepcopy(edge))

    return {n_key: sub_nodes, e_key: sub_edges}


def build_ptg(cluster_name, cluster_node_ids, node_by_id, node_to_cluster, edges, n_key, e_key):
    cluster_set = set(cluster_node_ids)
    trans_edges = []
    trans_node_ids = []

    for edge in edges:
        source, target = get_edge_endpoints(edge)
        if not source or not target:
            continue

        source_page = node_to_cluster.get(source)
        target_page = node_to_cluster.get(target)
        if source in cluster_set and target_page and target_page != cluster_name:
            trans_edges.append(copy.deepcopy(edge))
            trans_node_ids.extend([source, target])

    trans_node_ids = unique_keep_order(trans_node_ids)
    trans_nodes = []
    for node_id in trans_node_ids:
        if node_id not in node_by_id:
            continue
        trans_nodes.append(build_node(node_by_id[node_id], node_to_cluster.get(node_id, "UNKNOWN_PAGE")))

    return {n_key: trans_nodes, e_key: trans_edges}


def build_app_ptg(node_by_id, node_to_cluster, edges, n_key, e_key):
    trans_edges = []
    trans_node_ids = []
    seen_edge_keys = set()

    for edge in edges:
        source, target = get_edge_endpoints(edge)
        if not source or not target:
            continue

        source_page = node_to_cluster.get(source)
        target_page = node_to_cluster.get(target)
        if not source_page or not target_page or source_page == target_page:
            continue

        edge_id = str(edge.get("id", "")).strip()
        edge_key = edge_id or json.dumps(edge, ensure_ascii=False, sort_keys=True)
        if edge_key in seen_edge_keys:
            continue
        seen_edge_keys.add(edge_key)

        trans_edges.append(copy.deepcopy(edge))
        trans_node_ids.extend([source, target])

    trans_node_ids = unique_keep_order(trans_node_ids)
    trans_nodes = []
    for node_id in trans_node_ids:
        if node_id not in node_by_id:
            continue
        trans_nodes.append(build_node(node_by_id[node_id], node_to_cluster.get(node_id, "UNKNOWN_PAGE")))

    return {n_key: trans_nodes, e_key: trans_edges}


def order_page_pairs_dfs(page_pairs, page_order=None, entry_page=None):
    """Traverse undirected page pairs, keeping every pair once, including cycles.

    Start at the entry page, then cover disconnected components in page order.
    Visit neighbors in the order their pairs first appeared in the UTG edges.
    """
    page_order = page_order or {}
    adjacency = {}
    for page_pair in page_pairs:
        page_a, page_b = page_pair
        adjacency.setdefault(page_a, []).append((page_b, page_pair))
        adjacency.setdefault(page_b, []).append((page_a, page_pair))

    roots = sorted(adjacency, key=lambda page: (page_order.get(page, 10**9), page))
    if entry_page in adjacency:
        roots.remove(entry_page)
        roots.insert(0, entry_page)

    visited_pages = set()
    visited_pairs = set()
    ordered_pairs = []
    for root in roots:
        if root in visited_pages:
            continue
        visited_pages.add(root)
        stack = [iter(adjacency[root])]
        while stack:
            item = next(stack[-1], None)
            if item is None:
                stack.pop()
                continue
            target, page_pair = item
            if page_pair in visited_pairs:
                continue
            visited_pairs.add(page_pair)
            ordered_pairs.append(page_pair)
            if target not in visited_pages:
                visited_pages.add(target)
                stack.append(iter(adjacency[target]))

    return ordered_pairs


def build_pair_navigation_graphs(
    node_by_id, node_to_cluster, edges, n_key, e_key, page_order=None, entry_page=None
):
    page_order = page_order or {}
    pair_edges = {}
    pair_node_ids = {}

    for edge in edges:
        source, target = get_edge_endpoints(edge)
        if not source or not target:
            continue

        source_page = node_to_cluster.get(source)
        target_page = node_to_cluster.get(target)
        if not source_page or not target_page or source_page == target_page:
            continue

        ordered_pair = tuple(
            sorted(
                (source_page, target_page),
                key=lambda page: (page_order.get(page, 10**9), page),
            )
        )
        pair_edges.setdefault(ordered_pair, []).append(copy.deepcopy(edge))
        pair_node_ids.setdefault(ordered_pair, []).extend([source, target])

    pair_graphs = {}
    for page_pair in order_page_pairs_dfs(pair_edges, page_order, entry_page):
        graph_edges = pair_edges[page_pair]
        graph_node_ids = unique_keep_order(pair_node_ids.get(page_pair, []))
        graph_nodes = []
        for node_id in graph_node_ids:
            if node_id not in node_by_id:
                continue
            graph_nodes.append(build_node(node_by_id[node_id], node_to_cluster.get(node_id, "UNKNOWN_PAGE")))
        pair_graphs[page_pair] = {n_key: graph_nodes, e_key: graph_edges}

    return pair_graphs


def pair_navigation_filename(page_a, page_b):
    return f"{sanitize_filename(page_a)}_{sanitize_filename(page_b)}.json"


def clear_json_files(directory: Path):
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        if path.is_file():
            path.unlink()


def build_outputs_from_cluster_maps(
    app_dir: Path,
    group_to_mockups,
    mockup_to_group,
    source_cluster_path: Path,
    utg_process_path: Path,
    output_subgraph_dir_name: str = DEFAULT_OUTPUT_SUBGRAPH_DIR_NAME,
    output_ptg_filename: str = DEFAULT_OUTPUT_PTG_FILENAME,
):
    if not utg_process_path.exists():
        raise FileNotFoundError(f"UTG_process.json not found: {utg_process_path}")

    utg = load_json(utg_process_path)
    n_key, e_key = detect_schema(utg)
    nodes = utg.get(n_key, [])
    edges = utg.get(e_key, [])
    entry_node_id = get_node_id(nodes[0]) if nodes else ""

    node_by_id, normalized_to_id = build_node_lookup(nodes)
    group_to_node_ids = resolve_group_node_ids(group_to_mockups, normalized_to_id)
    node_to_group = resolve_node_to_group(mockup_to_group, normalized_to_id)
    cluster_to_node_ids, node_to_cluster, cluster_home_map, cluster_candidates_map = build_named_clusters(
        group_to_node_ids,
        edges,
        node_to_group,
        entry_node_id=entry_node_id,
    )

    out_root = app_dir / output_subgraph_dir_name
    state_dir = out_root / "state"
    transition_dir = out_root / "transition"
    pair_navigation_dir = out_root / "pair_navigation"
    state_items = []
    transition_items = []
    pair_navigation_items = []

    clear_json_files(state_dir)
    clear_json_files(transition_dir)

    for cluster_name, cluster_node_ids in cluster_to_node_ids.items():
        if not cluster_node_ids:
            continue

        psg = build_psg(cluster_name, cluster_node_ids, node_by_id, edges, n_key, e_key)
        page_file_id = sanitize_filename(cluster_name)
        psg_path = state_dir / f"{page_file_id}.json"
        write_json(psg_path, psg)

        state_items.append(
            {
                "page": cluster_name,
                "home_node_id": cluster_home_map.get(cluster_name, ""),
                "home_candidates": cluster_candidates_map.get(cluster_name, []),
                "node_count": len(psg[n_key]),
                "edge_count": len(psg[e_key]),
                "graph_path": str(psg_path),
            }
        )

        ptg = build_ptg(cluster_name, cluster_node_ids, node_by_id, node_to_cluster, edges, n_key, e_key)
        ptg_path = transition_dir / f"{page_file_id}_transition.json"
        write_json(ptg_path, ptg)

        transition_items.append(
            {
                "page": cluster_name,
                "home_node_id": cluster_home_map.get(cluster_name, ""),
                "node_count": len(ptg[n_key]),
                "edge_count": len(ptg[e_key]),
                "graph_path": str(ptg_path),
            }
        )

    app_ptg = build_app_ptg(node_by_id, node_to_cluster, edges, n_key, e_key)
    app_ptg_path = out_root / "app_transition_graph.json"
    root_ptg_path = app_dir / output_ptg_filename
    write_json(app_ptg_path, app_ptg)
    write_json(root_ptg_path, app_ptg)

    clear_json_files(pair_navigation_dir)
    page_order = {page_name: index for index, page_name in enumerate(cluster_to_node_ids)}
    pair_graphs = build_pair_navigation_graphs(
        node_by_id,
        node_to_cluster,
        edges,
        n_key,
        e_key,
        page_order=page_order,
        entry_page=node_to_cluster.get(entry_node_id),
    )
    used_pair_filenames = set()
    for (page_a, page_b), ppng in pair_graphs.items():
        pair_file_name = pair_navigation_filename(page_a, page_b)
        if pair_file_name in used_pair_filenames:
            base = pair_file_name[:-5]
            suffix = 2
            while f"{base}_{suffix}.json" in used_pair_filenames:
                suffix += 1
            pair_file_name = f"{base}_{suffix}.json"
        used_pair_filenames.add(pair_file_name)

        ppng_path = pair_navigation_dir / pair_file_name
        write_json(ppng_path, ppng)
        pair_navigation_items.append(
            {
                "page_a": page_a,
                "page_b": page_b,
                "node_count": len(ppng[n_key]),
                "edge_count": len(ppng[e_key]),
                "graph_path": str(ppng_path),
            }
        )

    write_json(
        out_root / "page_state_subgraphs.json",
        {
            "app_dir": str(app_dir),
            "source_cluster": str(source_cluster_path),
            "source_utg_process": str(utg_process_path),
            "schema": {"node_key": n_key, "edge_key": e_key},
            "page_count": len(state_items),
            "items": state_items,
        },
    )

    write_json(
        out_root / "page_transition_graphs.json",
        {
            "app_dir": str(app_dir),
            "source_cluster": str(source_cluster_path),
            "source_utg_process": str(utg_process_path),
            "schema": {"node_key": n_key, "edge_key": e_key},
            "page_count": len(transition_items),
            "app_ptg": {
                "node_count": len(app_ptg[n_key]),
                "edge_count": len(app_ptg[e_key]),
                "graph_path": str(app_ptg_path),
                "root_graph_path": str(root_ptg_path),
            },
            "items": transition_items,
        },
    )

    write_json(
        out_root / "page_pair_navigation_graphs.json",
        {
            "app_dir": str(app_dir),
            "source_cluster": str(source_cluster_path),
            "source_utg_process": str(utg_process_path),
            "schema": {"node_key": n_key, "edge_key": e_key},
            "pair_count": len(pair_navigation_items),
            "items": pair_navigation_items,
        },
    )

    return {
        "page_count": len(state_items),
        "state_graph_count": len(state_items),
        "transition_graph_count": len(transition_items),
        "pair_navigation_graph_count": len(pair_navigation_items),
        "app_ptg_node_count": len(app_ptg[n_key]),
        "app_ptg_edge_count": len(app_ptg[e_key]),
        "out_root": str(out_root),
    }


def build_outputs_for_one_app(app_dir: Path):
    cluster_path = app_dir / CLUSTER_RELATIVE_PATH
    utg_process_path = app_dir / UTG_PROCESS_FILENAME
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

    group_to_mockups, mockup_to_group = load_page_clusters(cluster_path)
    return build_outputs_from_cluster_maps(
        app_dir,
        group_to_mockups,
        mockup_to_group,
        cluster_path,
        utg_process_path,
        DEFAULT_OUTPUT_SUBGRAPH_DIR_NAME,
        DEFAULT_OUTPUT_PTG_FILENAME,
    )



def find_app_entries(data_root: Path):
    entries = []
    for cluster_path in sorted(data_root.rglob(str(CLUSTER_RELATIVE_PATH))):
        if not cluster_path.is_file():
            continue
        app_dir = cluster_path.parent.parent
        if (app_dir / UTG_PROCESS_FILENAME).exists():
            entries.append(app_dir)
    return entries


def main():
    data_root = Path(__file__).resolve().parents[1] / "dataset"
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")

    app_entries = find_app_entries(data_root)
    if not app_entries:
        print(f"No {CLUSTER_RELATIVE_PATH} found under: {data_root}")
        return

    ok_count = 0
    fail_count = 0

    for app_dir in app_entries:
        rel = str(app_dir.relative_to(data_root))
        try:
            result = build_outputs_for_one_app(app_dir)
            ok_count += 1
            print(
                f"[OK] {rel} | page_count={result['page_count']} | "
                f"state_graph_count={result['state_graph_count']} | "
                f"transition_graph_count={result['transition_graph_count']} | "
                f"pair_navigation_graph_count={result['pair_navigation_graph_count']} | "
                f"app_ptg_nodes={result['app_ptg_node_count']} | "
                f"app_ptg_edges={result['app_ptg_edge_count']}"
            )
        except Exception as exc:
            fail_count += 1
            print(f"[FAIL] {rel}: {exc}")

    print(
        f"Done. total={len(app_entries)}, success={ok_count}, fail={fail_count}, data_root={data_root}"
    )


if __name__ == "__main__":
    main()
