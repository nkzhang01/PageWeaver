#!/usr/bin/env python3
"""
Merge duplicate UTG nodes based on identical state screenshots.

Rules:
1. Only `states/*` images (referenced by nodes[].image) are considered.
2. Nodes whose state image bytes are identical (SHA256) are treated as duplicates.
3. Duplicate nodes are redirected to one canonical node.
4. Edges are rewritten to canonical node ids, then merged by (from, to).
5. A new UTG file is written (original file is not overwritten by default).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

UTG_PREFIX = "var utg = \n"
DEFAULT_APP_DIR = (
    Path(__file__).resolve().parent / "mobileViewData" / "com.hashmusic.musicplayer"
)
SUPPORTED_HASH_ALGOS = ("md5", "sha256")


def file_digest(file_path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_utg(utg_path: Path) -> dict:
    text = utg_path.read_text(encoding="utf-8")
    marker = "var utg ="
    pos = text.find(marker)
    if pos >= 0:
        json_text = text[pos + len(marker) :].strip()
    else:
        json_text = text.strip()
    return json.loads(json_text)


def dump_utg(utg_path: Path, data: dict) -> None:
    content = UTG_PREFIX + json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    utg_path.write_text(content, encoding="utf-8")


def build_node_redirect(
    app_dir: Path, nodes: List[dict], hash_algo: str
) -> Tuple[Dict[str, str], List[dict], List[str]]:
    """
    Returns:
      redirect: duplicate_node_id -> canonical_node_id
      groups: human-readable duplicate group info
      missing_images: node ids whose state image is missing on disk
    """
    grouped: Dict[str, List[dict]] = defaultdict(list)
    missing_images: List[str] = []

    for node in nodes:
        node_id = node.get("id")
        image_rel = node.get("image")
        if not isinstance(node_id, str) or not isinstance(image_rel, str):
            continue

        rel_path = Path(image_rel)
        if not rel_path.parts or rel_path.parts[0] != "states":
            continue

        abs_path = app_dir / rel_path
        if not abs_path.exists():
            missing_images.append(node_id)
            continue

        digest = file_digest(abs_path, hash_algo)
        grouped[digest].append(node)

    redirect: Dict[str, str] = {}
    groups: List[dict] = []

    for digest, group_nodes in grouped.items():
        if len(group_nodes) <= 1:
            continue

        sorted_group = sorted(group_nodes, key=lambda x: str(x.get("id", "")))
        canonical = sorted_group[0]
        canonical_id = canonical["id"]
        duplicate_nodes = sorted_group[1:]

        for dup in duplicate_nodes:
            dup_id = dup["id"]
            redirect[dup_id] = canonical_id

        groups.append(
            {
                "digest": digest,
                "canonical_node_id": canonical_id,
                "canonical_image": canonical.get("image"),
                "all_node_ids": [n["id"] for n in sorted_group],
                "duplicate_node_ids": [n["id"] for n in duplicate_nodes],
                "all_images": [n.get("image") for n in sorted_group],
            }
        )

    groups.sort(
        key=lambda x: (-len(x["all_node_ids"]), x["canonical_node_id"], x["digest"])
    )
    missing_images.sort()
    return redirect, groups, missing_images


def event_key(event: dict) -> str:
    event_id = event.get("event_id")
    event_type = event.get("event_type")
    event_str = event.get("event_str")
    view_images = tuple(event.get("view_images", []))
    return json.dumps(
        {
            "event_id": event_id,
            "event_type": event_type,
            "event_str": event_str,
            "view_images": view_images,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_edge_label(events: List[dict]) -> str:
    event_ids = [ev.get("event_id") for ev in events if ev.get("event_id") is not None]
    if event_ids and len(event_ids) == len(events):
        return ", ".join(str(x) for x in event_ids)
    return str(len(events))


def build_edge_title(events: List[dict]) -> str:
    rows = []
    for ev in events:
        eid = ev.get("event_id", "")
        text = ev.get("event_str")
        if not isinstance(text, str):
            text = json.dumps(ev, ensure_ascii=False)
        rows.append(
            f"<tr><th>{html.escape(str(eid))}</th><td>{html.escape(text)}</td></tr>"
        )
    return "<table class=\"table\">\n" + "\n".join(rows) + "\n</table>"


def normalize_events(events: List[dict]) -> Tuple[List[dict], int]:
    unique_events: List[dict] = []
    seen = set()
    valid_count = 0

    for ev in events:
        if not isinstance(ev, dict):
            continue
        valid_count += 1
        k = event_key(ev)
        if k in seen:
            continue
        unique_events.append(deepcopy(ev))
        seen.add(k)

    return unique_events, valid_count


def rewrite_and_merge_edges(edges: List[dict], redirect: Dict[str, str]) -> Tuple[List[dict], dict]:
    merged: Dict[Tuple[str, str], dict] = {}
    meta_by_key: Dict[Tuple[str, str], dict] = {}
    endpoint_rewrites = 0
    valid_source_edge_count = 0
    source_event_count_before_dedup = 0

    for edge in edges:
        old_from = edge.get("from")
        old_to = edge.get("to")
        if not isinstance(old_from, str) or not isinstance(old_to, str):
            continue
        valid_source_edge_count += 1

        raw_events = edge.get("events", [])
        if not isinstance(raw_events, list):
            raw_events = []
        normalized_events, valid_event_count = normalize_events(raw_events)
        source_event_count_before_dedup += valid_event_count

        new_from = redirect.get(old_from, old_from)
        new_to = redirect.get(old_to, old_to)

        rewritten_endpoints = 0
        if new_from != old_from:
            endpoint_rewrites += 1
            rewritten_endpoints += 1
        if new_to != old_to:
            endpoint_rewrites += 1
            rewritten_endpoints += 1

        key = (new_from, new_to)
        source_info = {
            "edge_id": edge.get("id"),
            "from": old_from,
            "to": old_to,
            "event_count": valid_event_count,
            "rewritten_endpoints": rewritten_endpoints,
        }

        if key not in merged:
            new_edge = deepcopy(edge)
            new_edge["from"] = new_from
            new_edge["to"] = new_to
            new_edge["id"] = f"{new_from}-->{new_to}"
            new_edge["events"] = normalized_events
            merged[key] = new_edge
            meta_by_key[key] = {
                "new_from": new_from,
                "new_to": new_to,
                "source_edges": [source_info],
                "source_event_count": valid_event_count,
            }
            continue

        base = merged[key]
        existing_events = base.get("events", [])
        if not isinstance(existing_events, list):
            existing_events = []

        seen = {event_key(ev) for ev in existing_events if isinstance(ev, dict)}
        for ev in normalized_events:
            k = event_key(ev)
            if k in seen:
                continue
            existing_events.append(ev)
            seen.add(k)

        base["events"] = existing_events
        meta = meta_by_key[key]
        meta["source_edges"].append(source_info)
        meta["source_event_count"] += valid_event_count

    new_edges = list(merged.values())
    for edge in new_edges:
        events = edge.get("events", [])
        if not isinstance(events, list):
            events = []
            edge["events"] = events
        edge["label"] = build_edge_label(events)
        edge["title"] = build_edge_title(events)
        edge["id"] = f"{edge['from']}-->{edge['to']}"

    source_edges_merged_away = max(valid_source_edge_count - len(new_edges), 0)
    event_count_after_dedup = sum(
        len(e.get("events", [])) for e in new_edges if isinstance(e.get("events", []), list)
    )
    removed_duplicate_events = max(
        source_event_count_before_dedup - event_count_after_dedup, 0
    )

    merge_groups: List[dict] = []
    for key, meta in meta_by_key.items():
        source_edges = meta["source_edges"]
        if len(source_edges) <= 1:
            continue
        merged_event_count = len(merged[key].get("events", []))
        source_event_count = meta["source_event_count"]
        merge_groups.append(
            {
                "new_edge_id": f"{meta['new_from']}-->{meta['new_to']}",
                "new_from": meta["new_from"],
                "new_to": meta["new_to"],
                "source_edge_count": len(source_edges),
                "source_event_count": source_event_count,
                "merged_event_count": merged_event_count,
                "removed_duplicate_events": max(source_event_count - merged_event_count, 0),
                "rewritten_source_edge_count": sum(
                    1 for s in source_edges if s["rewritten_endpoints"] > 0
                ),
                "source_edges": source_edges,
            }
        )

    merge_groups.sort(
        key=lambda x: (
            -x["source_edge_count"],
            -x["removed_duplicate_events"],
            x["new_edge_id"],
        )
    )

    edge_report = {
        "summary": {
            "original_edge_count": len(edges),
            "valid_source_edge_count": valid_source_edge_count,
            "new_edge_count": len(new_edges),
            "endpoint_rewrites": endpoint_rewrites,
            "source_edges_merged_away": source_edges_merged_away,
            "merged_edge_groups": len(merge_groups),
            "source_event_count_before_dedup": source_event_count_before_dedup,
            "event_count_after_dedup": event_count_after_dedup,
            "removed_duplicate_events": removed_duplicate_events,
        },
        "merge_groups": merge_groups,
    }

    return new_edges, edge_report


def refresh_utg_stats(data: dict) -> None:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    data["num_nodes"] = len(nodes) if isinstance(nodes, list) else 0
    data["num_edges"] = len(edges) if isinstance(edges, list) else 0
    data["num_transitions"] = data["num_edges"]

    if isinstance(edges, list):
        data["num_effective_events"] = sum(
            len(e.get("events", [])) for e in edges if isinstance(e.get("events", []), list)
        )
    else:
        data["num_effective_events"] = 0

    if isinstance(nodes, list):
        activities = {n.get("activity") for n in nodes if isinstance(n.get("activity"), str)}
        data["num_reached_activities"] = len(activities)
    else:
        data["num_reached_activities"] = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge duplicate UTG state nodes based on identical state screenshots."
    )
    parser.add_argument(
        "--app-dir",
        default=str(DEFAULT_APP_DIR),
        help="App dataset root directory (contains utg.js and states/).",
    )
    parser.add_argument(
        "--utg-file",
        default="utg.js",
        help="Source UTG filename under --app-dir.",
    )
    parser.add_argument(
        "--output-utg",
        default="utg_dedup_states.js",
        help="Output UTG filename under --app-dir.",
    )
    parser.add_argument(
        "--report-file",
        default="utg_dedup_states_report.json",
        help="Report filename under --app-dir.",
    )
    parser.add_argument(
        "--hash-algo",
        default="md5",
        choices=SUPPORTED_HASH_ALGOS,
        help="Image hash algorithm for duplicate detection (default: md5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_dir = Path(args.app_dir).resolve()
    utg_path = app_dir / args.utg_file
    output_utg_path = app_dir / args.output_utg
    report_path = app_dir / args.report_file

    if not app_dir.exists():
        raise FileNotFoundError(f"App directory does not exist: {app_dir}")
    if not utg_path.exists():
        raise FileNotFoundError(f"UTG file does not exist: {utg_path}")

    data = load_utg(utg_path)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Invalid UTG format: nodes/edges must be lists.")

    redirect, groups, missing_images = build_node_redirect(
        app_dir, nodes, args.hash_algo
    )
    duplicate_node_ids = set(redirect.keys())

    new_nodes = [deepcopy(n) for n in nodes if n.get("id") not in duplicate_node_ids]
    new_edges, edge_report = rewrite_and_merge_edges(edges, redirect)

    data["nodes"] = new_nodes
    data["edges"] = new_edges
    refresh_utg_stats(data)

    dump_utg(output_utg_path, data)

    report = {
        "app_dir": str(app_dir),
        "hash_algorithm": args.hash_algo,
        "source_utg": str(utg_path),
        "output_utg": str(output_utg_path),
        "old_node_count": len(nodes),
        "new_node_count": len(new_nodes),
        "removed_node_count": len(duplicate_node_ids),
        "old_edge_count": len(edges),
        "new_edge_count": len(new_edges),
        "edge_endpoint_rewrites": edge_report["summary"]["endpoint_rewrites"],
        "duplicate_groups": len(groups),
        "missing_state_images": missing_images,
        "redirect_map": redirect,
        "groups": groups,
        "edge_merge_summary": edge_report["summary"],
        "edge_merge_groups": edge_report["merge_groups"],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Source UTG: {utg_path}")
    print(f"Output UTG: {output_utg_path}")
    print(f"Old nodes: {len(nodes)}")
    print(f"New nodes: {len(new_nodes)}")
    print(f"Removed duplicate nodes: {len(duplicate_node_ids)}")
    print(f"Old edges: {len(edges)}")
    print(f"New edges: {len(new_edges)}")
    print(f"Edge endpoint rewrites: {edge_report['summary']['endpoint_rewrites']}")
    print(f"Merged edge groups: {edge_report['summary']['merged_edge_groups']}")
    print(f"Duplicate groups: {len(groups)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
