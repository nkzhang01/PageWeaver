import copy
import json
import re
from pathlib import Path

from sameDetect import (
    judge_screen_relation as same_detect_judge_screen_relation,
)

_same_detect_pair_cache = {}


def _same_detect_same_page_result(path_a, path_b):
    if same_detect_judge_screen_relation is None:
        raise RuntimeError("sameDetect is unavailable. Please install required dependencies (e.g. transformers, torch).")

    p1 = str(path_a or "").strip()
    p2 = str(path_b or "").strip()
    if not p1 or not p2:
        raise ValueError("sameDetect requires non-empty image paths.")
    if not Path(p1).exists() or not Path(p2).exists():
        raise FileNotFoundError(f"sameDetect image path not found: '{p1}' or '{p2}'")

    cache_key = tuple(sorted((p1, p2)))
    if cache_key in _same_detect_pair_cache:
        return _same_detect_pair_cache[cache_key]

    result = same_detect_judge_screen_relation(
        img1_path=p1,
        img2_path=p2,
    )
    embedding_result = result.get("embedding_result") or {}
    popup_result = result.get("popup_result") or {}
    popup_metrics = popup_result.get("metrics") or {}
    out = {
        "is_same_page": bool(result.get("same_page", False)),
        "relation": str(result.get("relation", "")),
        "reason": str(embedding_result.get("reason", "")),
        "rgb_similarity": float(embedding_result.get("rgb_similarity", 0.0)),
        "edge_similarity": float(embedding_result.get("edge_similarity", 0.0)),
        "change_ratio": float(popup_metrics.get("change_ratio", 0.0)),
        "structure_similarity": float(popup_metrics.get("structure_similarity", 0.0)),
        "popup_by_darkening": bool(popup_metrics.get("popup_by_darkening", False)),
        "popup_by_brightening": bool(popup_metrics.get("popup_by_brightening", False)),
    }

    _same_detect_pair_cache[cache_key] = out
    return out


def load_utg(utg_path):
    text = utg_path.read_text(encoding="utf-8")
    marker = "var utg ="
    pos = text.find(marker)
    if pos >= 0:
        text = text[pos + len(marker) :].strip()
    return json.loads(text)


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


def unique_keep_order(items, limit=None):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def resolve_image_path(app_dir, node):
    node_id = get_node_id(node)
    candidates = []

    if node_id:
        candidates.append(node_id)

    # Compatibility fallback: only for locating screenshot path.
    image_field = node.get("image")
    if isinstance(image_field, str) and image_field.strip():
        candidates.append(Path(image_field).name)

    expanded = []
    for c in candidates:
        p = Path(c)
        if p.suffix:
            expanded.append(p.name)
            expanded.append(p.stem + ".png")
            expanded.append(p.stem + ".jpg")
            expanded.append(p.stem + ".jpeg")
        else:
            expanded.append(p.name + ".png")
            expanded.append(p.name + ".jpg")
            expanded.append(p.name + ".jpeg")

    expanded = unique_keep_order(expanded)

    search_paths = []
    for name in expanded:
        search_paths.append(app_dir / name)
        search_paths.append(app_dir / "states" / name)

    for p in search_paths:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    return None

def build_edge_relation_judgements(edges, node_features):
    judgements = []

    for edge in edges:
        edge_id = edge.get("id", "")
        a, b = get_edge_endpoints(edge)
        item = {
            "edge_id": edge_id if isinstance(edge_id, str) else str(edge_id),
            "from": a,
            "to": b,
            "relation": "",
            "same_page": False,
            "is_popup_relation": False,
            "reason": "",
            "rgb_similarity": 0.0,
            "edge_similarity": 0.0,
            "change_ratio": 0.0,
            "structure_similarity": 0.0,
            "popup_by_darkening": False,
            "popup_by_brightening": False,
            "judged": False,
        }

        if not a or not b:
            item["reason"] = "missing_edge_endpoint"
            judgements.append(item)
            continue

        fa = node_features.get(a)
        fb = node_features.get(b)
        if fa is None or fb is None:
            item["reason"] = "missing_node_image"
            judgements.append(item)
            continue

        rel = _same_detect_same_page_result(
            fa.get("image_path", ""),
            fb.get("image_path", ""),
        )
        relation = str(rel.get("relation", ""))
        same_page = bool(rel.get("is_same_page", False))
        is_popup = relation == "same_page_popup"
        item.update(
            {
                "relation": relation,
                "same_page": same_page,
                "is_popup_relation": is_popup,
                "reason": str(rel.get("reason", "")),
                "rgb_similarity": float(rel.get("rgb_similarity", 0.0)),
                "edge_similarity": float(rel.get("edge_similarity", 0.0)),
                "change_ratio": float(rel.get("change_ratio", 0.0)),
                "structure_similarity": float(rel.get("structure_similarity", 0.0)),
                "popup_by_darkening": bool(rel.get("popup_by_darkening", False)),
                "popup_by_brightening": bool(rel.get("popup_by_brightening", False)),
                "judged": True,
            }
        )
        judgements.append(item)

    return judgements


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_filename(name):
    s = str(name or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "page_type"


def build_cluster_name(node_ids, node_map):
    counts = {}
    first_pos = {}
    for idx, nid in enumerate(node_ids):
        node = node_map.get(nid) or {}
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
        if name not in first_pos:
            first_pos[name] = idx

    if not counts:
        return "unknown_page"

    best_name = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], first_pos.get(kv[0], 10**9), kv[0]),
    )[0][0]
    return best_name


def make_unique_page_type_id(base_name, name_counters):
    base_id = sanitize_filename(base_name)
    if not base_id:
        base_id = "page_type"
    idx = int(name_counters.get(base_id, 0)) + 1
    name_counters[base_id] = idx
    return f"{base_id}{idx}"


def pick_home_node(node_ids, edges, utg_first_node_id="", current_page_type_id="", node_to_type=None):
    if not node_ids:
        return ""

    node_set = set(node_ids)
    node_to_type = node_to_type or {}

    # Rule 1: if UTG first node is in this cluster, use it directly.
    if utg_first_node_id and utg_first_node_id in node_set:
        return utg_first_node_id

    first_pos = {nid: idx for idx, nid in enumerate(node_ids)}
    indegree_from_other_type = {nid: 0 for nid in node_ids}

    for e in edges:
        a, b = get_edge_endpoints(e)
        if b not in node_set or a == b:
            continue
        src_type = node_to_type.get(a, "")
        dst_type = node_to_type.get(b, "")
        if not src_type or not dst_type:
            continue
        if current_page_type_id and dst_type != current_page_type_id:
            continue
        if src_type != dst_type:
            indegree_from_other_type[b] += 1

    candidates = [nid for nid in node_ids if indegree_from_other_type[nid] > 0]
    if not candidates:
        return node_ids[0]
    if len(candidates) == 1:
        return candidates[0]

    cand_set = set(candidates)
    class_adj = {nid: set() for nid in node_ids}
    for e in edges:
        a, b = get_edge_endpoints(e)
        if a in node_set and b in node_set and a != b:
            class_adj[a].add(b)

    def reachable_candidate_count(src):
        visited = set()
        stack = [src]
        while stack:
            cur = stack.pop()
            for nxt in class_adj.get(cur, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        visited.discard(src)
        return len([nid for nid in visited if nid in cand_set])

    scored = []
    for nid in candidates:
        reach_cnt = reachable_candidate_count(nid)
        scored.append((reach_cnt, -first_pos.get(nid, 10**9), nid))
    scored.sort(reverse=True)
    return scored[0][2]


def order_cluster_node_ids(node_ids, edges, utg_first_node_id="", current_page_type_id="", node_to_type=None):
    if not node_ids:
        return []
    home_node = pick_home_node(
        node_ids=node_ids,
        edges=edges,
        utg_first_node_id=utg_first_node_id,
        current_page_type_id=current_page_type_id,
        node_to_type=node_to_type,
    )
    if not home_node:
        return list(node_ids)
    return [home_node] + [nid for nid in node_ids if nid != home_node]


def make_subgraph(nodes, edges, keep_node_ids, n_key, e_key):
    keep_list = unique_keep_order(keep_node_ids)
    keep = set(keep_list)
    node_by_id = {get_node_id(n): n for n in nodes if get_node_id(n) in keep}
    sub_nodes = [copy.deepcopy(node_by_id[nid]) for nid in keep_list if nid in node_by_id]
    sub_edges = []
    for e in edges:
        a, b = get_edge_endpoints(e)
        if a in keep and b in keep:
            sub_edges.append(copy.deepcopy(e))
    return {n_key: sub_nodes, e_key: sub_edges}


def marker_base_from_node(node):
    props = node.get("props") if isinstance(node, dict) else {}
    if not isinstance(props, dict):
        props = {}
    content = str(props.get("content", "")).strip()
    head = content.split("\n", 1)[0].strip() if content else ""
    return head or "marker"


def build_cluster_marker_map(node_ids, edges, page_type_id, node_to_type, node_map):
    marker_map = {nid: "" for nid in node_ids}
    if not node_ids:
        return marker_map

    node_set = set(node_ids)
    home_node = node_ids[0]
    marked_nodes = {home_node}

    for e in edges:
        a, b = get_edge_endpoints(e)
        if b not in node_set or a in node_set or a == b:
            continue
        src_type = node_to_type.get(a, "")
        dst_type = node_to_type.get(b, "")
        if not src_type or not dst_type:
            continue
        if dst_type == page_type_id and src_type != page_type_id:
            marked_nodes.add(b)

    marker_map[home_node] = "BASE"
    seq = 1
    for nid in node_ids:
        if nid == home_node:
            continue
        if nid not in marked_nodes:
            continue
        base = marker_base_from_node(node_map.get(nid, {}))
        marker_map[nid] = f"{base}{seq}"
        seq += 1

    return marker_map


def collect_intra_edges_with_relations(node_ids, edge_judgements):
    node_set = set(node_ids)
    out = []
    for item in edge_judgements:
        a = str(item.get("from", "")).strip()
        b = str(item.get("to", "")).strip()
        if not a or not b:
            continue
        if a in node_set and b in node_set:
            out.append(
                {
                    "edge_id": item.get("edge_id", ""),
                    "from": a,
                    "to": b,
                    "relation": str(item.get("relation", "")),
                    "same_page": bool(item.get("same_page", False)),
                    "is_popup_relation": bool(item.get("is_popup_relation", False)),
                    "judged": bool(item.get("judged", False)),
                }
            )
    return out


def build_same_page_components(node_ids, edge_judgements):
    node_ids = sorted(set(node_ids))
    node_set = set(node_ids)
    adj = {nid: set() for nid in node_ids}

    for item in edge_judgements:
        a = str(item.get("from", "")).strip()
        b = str(item.get("to", "")).strip()
        relation = str(item.get("relation", ""))
        if a not in node_set or b not in node_set:
            continue
        if relation not in {"same_page_popup", "same_page_non_popup"}:
            continue
        adj[a].add(b)
        adj[b].add(a)

    components = []
    visited = set()
    for nid in node_ids:
        if nid in visited:
            continue
        stack = [nid]
        visited.add(nid)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj.get(cur, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        components.append(sorted(comp))

    return sorted(components, key=lambda x: (-len(x), x[0] if x else ""))


def build_outputs(app_dir, utg_data, n_key, e_key, out_dir):
    nodes = utg_data.get(n_key, [])
    edges = utg_data.get(e_key, [])
    node_map = {get_node_id(n): n for n in nodes if get_node_id(n)}
    utg_first_node_id = ""
    for n in nodes:
        nid = get_node_id(n)
        if nid:
            utg_first_node_id = nid
            break

    node_features = {}
    for node in nodes:
        nid = get_node_id(node)
        if not nid:
            continue
        img_path = resolve_image_path(app_dir, node)
        if img_path is None:
            continue
        feats = {"image_path": str(img_path)}
        node_features[nid] = feats

    edge_judgements = build_edge_relation_judgements(edges, node_features)
    valid_node_ids = [get_node_id(n) for n in nodes if get_node_id(n) in node_features]
    clusters = build_same_page_components(valid_node_ids, edge_judgements)

    node_to_temp_type = {}
    for i, node_ids in enumerate(clusters):
        pt = f"_tmp_page_type_{i:03d}"
        for nid in node_ids:
            node_to_temp_type[nid] = pt

    page_types = []
    page_type_name_counters = {}
    for i, node_ids in enumerate(clusters):
        tmp_pt = f"_tmp_page_type_{i:03d}"
        ordered_node_ids = order_cluster_node_ids(
            node_ids=node_ids,
            edges=edges,
            utg_first_node_id=utg_first_node_id,
            current_page_type_id=tmp_pt,
            node_to_type=node_to_temp_type,
        )
        intra_edges = collect_intra_edges_with_relations(ordered_node_ids, edge_judgements)
        raw_name = build_cluster_name(ordered_node_ids, node_map)
        page_type_name = raw_name
        page_type_id = make_unique_page_type_id(page_type_name, page_type_name_counters)
        page_types.append(
            {
                "page_type_id": page_type_id,
                "page_type_name": page_type_name,
                "node_count": len(ordered_node_ids),
                "node_ids": ordered_node_ids,
                "intra_edges": intra_edges,
            }
        )

    node_to_type = {}
    for p in page_types:
        pt = p["page_type_id"]
        for nid in p["node_ids"]:
            node_to_type[nid] = pt

    cluster_marker_maps = {}
    node_marker_map = {}
    for p in page_types:
        pt = p["page_type_id"]
        marker_map = build_cluster_marker_map(
            node_ids=p["node_ids"],
            edges=edges,
            page_type_id=pt,
            node_to_type=node_to_type,
            node_map=node_map,
        )
        cluster_marker_maps[pt] = marker_map
        node_marker_map.update(marker_map)

    # Intra-page-type subgraphs (same UTG structure as input).
    intra_items = []
    for p in page_types:
        pt = p["page_type_id"]
        sub = make_subgraph(nodes, edges, p["node_ids"], n_key, e_key)
        marker_map = cluster_marker_maps.get(pt, {})
        for node in sub.get(n_key, []):
            nid = get_node_id(node)
            node["marker"] = marker_map.get(nid, "")
        intra_items.append(
            {
                "page_type_id": pt,
                "page_type_name": p["page_type_name"],
                "node_count": len(sub[n_key]),
                "edge_count": len(sub[e_key]),
                "utg_subgraph": sub,
            }
        )

    # Inter-page transition graph: remove all intra-page-type edges from original UTG.
    inter_edges = []
    touched_nodes = set()
    for e in edges:
        a, b = get_edge_endpoints(e)
        if not a or not b:
            continue
        ta = node_to_type.get(a)
        tb = node_to_type.get(b)
        if ta is None or tb is None:
            continue
        if ta != tb:
            inter_edges.append(copy.deepcopy(e))
            touched_nodes.add(a)
            touched_nodes.add(b)

    inter_nodes = [copy.deepcopy(n) for n in nodes if get_node_id(n) in touched_nodes]
    inter_graph = {n_key: inter_nodes, e_key: inter_edges}

    # Write files.
    write_json(
        out_dir / "page_type_clusters.json",
        {
            "app_dir": str(app_dir),
            "cluster_by": "edge_by_edge_sameDetect_relation_then_same_page_connected_components",
            "ocr_enabled": False,
            "page_type_count": len(page_types),
            "page_types": page_types,
            "node_to_page_type": node_to_type,
        },
    )

    write_json(
        out_dir / "edge_relation_judgements.json",
        {
            "app_dir": str(app_dir),
            "edge_count": len(edges),
            "judged_edge_count": sum(1 for x in edge_judgements if x.get("judged", False)),
            "edge_judgements": edge_judgements,
        },
    )

    write_json(
        out_dir / "intra_page_type_subgraphs.json",
        {
            "app_dir": str(app_dir),
            "schema": {"node_key": n_key, "edge_key": e_key},
            "page_type_count": len(intra_items),
            "subgraphs": intra_items,
        },
    )

    intra_dir = out_dir / "intra_page"
    related_dir = intra_dir
    for item in intra_items:
        pt = item["page_type_id"]
        f_json = intra_dir / f"{pt}.json"
        write_json(f_json, item["utg_subgraph"])

    for p in page_types:
        pt = p["page_type_id"]
        node_set = set(p["node_ids"])
        related_edges = []
        touched_nodes = set()

        for e in edges:
            a, b = get_edge_endpoints(e)
            if not a or not b:
                continue
            a_in = a in node_set
            b_in = b in node_set
            if a_in and not b_in and node_to_type.get(b, "") and node_to_type.get(b, "") != pt:
                related_edges.append(copy.deepcopy(e))
                touched_nodes.add(a)
                touched_nodes.add(b)

        related_nodes = [copy.deepcopy(n) for n in nodes if get_node_id(n) in touched_nodes]
        for node in related_nodes:
            nid = get_node_id(node)
            node["page"] = node_to_type.get(nid, "")
            node["marker"] = node_marker_map.get(nid, "")
        related_graph = {n_key: related_nodes, e_key: related_edges}
        related_file = related_dir / f"{pt}_transition.json"
        write_json(related_file, related_graph)

    inter_graph_file = out_dir / "inter_page_transition_graph.json"
    write_json(inter_graph_file, inter_graph)


    return {
        "page_type_count": len(page_types),
        "intra_graph_count": len(intra_items),
        "inter_edge_count": len(inter_edges),
        "out_dir": str(out_dir),
    }


def main():
    # glitchvideoeditor/glitchvideoeffect
    # com/jiffy,musicplayer,valleythreesixfive

    # app_classification_lists = ["com.aptener.bluarmor.app", "com.freetymekiyan.apas",
    #                             "com.hashmusic.musicplayer","com.language.learnspainish","com.magazinecloner.aestheticmedicine",
    #                             "com.moymer.falou","com.nordicusability.jiffy",
    #                             "glitchvideoeditor.videoeffects.glitchvideoeffect", "science.brief.app",
    #                             "thecouponsapp.coupon" ]
    app_dir = r"D:\ui2code\InteractionSequence\data\com\jiffy"
    
    utg_path = app_dir / "utg.json"
    out_dir = app_dir
    # out_dir = app_dir / "page_type_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not app_dir.exists():
        raise FileNotFoundError(f"app_dir does not exist: {app_dir}")
    if not utg_path.exists():
        raise FileNotFoundError(f"utg file does not exist: {utg_path}")

    data = load_utg(utg_path)
    n_key, e_key = detect_schema(data)

    result = build_outputs(
        app_dir=app_dir,
        utg_data=data,
        n_key=n_key,
        e_key=e_key,
        out_dir=out_dir,
    )

    print(f"UTG: {utg_path}")
    print(f"schema: {n_key}/{e_key}")
    print(f"page_type_count: {result['page_type_count']}")
    print(f"intra_subgraph_count: {result['intra_graph_count']}")
    print(f"inter_transition_edge_count: {result['inter_edge_count']}")
    print(f"output_dir: {result['out_dir']}")


if __name__ == "__main__":
    main()
