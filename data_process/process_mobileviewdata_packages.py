import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


INPUT_PACKAGE_PATH = Path(__file__).resolve().parent / "mobileViewData"
OUTPUT_ROOT_PATH = INPUT_PACKAGE_PATH.parent / "output"
# Optional: set to a package directory name to process only one package.
# Example: "com.nordicusability.jiffy"
ONLY_PACKAGE_NAME = "glitchvideoeditor.videoeffects.glitchvideoeffect"
# Optional: when processing a single package, set explicit crop size [top, bottom].
# Example: [100, 200]
ONLY_PACKAGE_CROP_SIZE = [0,560]
UTG_PREFIX = "var utg = \n"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_utg(utg_path):
    text = utg_path.read_text(encoding="utf-8", errors="ignore")
    marker = "var utg ="
    pos = text.find(marker)
    if pos >= 0:
        text = text[pos + len(marker) :].strip()
    return json.loads(text)


def write_utg_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_utg_js(path, data):
    path.write_text(UTG_PREFIX + json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def find_app_dirs(root_dir):
    apps = set()
    for utg_name in ("utg.js", "utg.json"):
        for utg_path in root_dir.rglob(utg_name):
            app_dir = utg_path.parent.resolve()
            states_dir = app_dir / "states"
            if not states_dir.exists() or not states_dir.is_dir():
                continue
            apps.add(app_dir)
    return sorted(set(apps), key=lambda p: str(p))


def detect_source_utg_path(app_dir):
    for name in ("utg.js", "utg.json"):
        p = app_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def split_package_type_and_app_name(package_name):
    name = str(package_name).strip()
    if not name:
        return "unknown", "unknown"
    parts = [p for p in name.split(".") if p]
    if not parts:
        return name, name
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[-1]


def collect_package_entries(package_path):
    package_path = package_path.resolve()
    entries = []

    # Case 1: package_path is a single package directory.
    if (package_path / "states").is_dir() and detect_source_utg_path(package_path):
        pkg_type, app_name = split_package_type_and_app_name(package_path.name)
        return [
            {
                "package_dir": package_path,
                "package_name": package_path.name,
                "package_type": pkg_type,
                "app_name": app_name,
            }
        ]

    # Case 2: package_path is mobileViewData root, collect direct child packages.
    for child in sorted(package_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if ONLY_PACKAGE_NAME and child.name != ONLY_PACKAGE_NAME:
            continue
        if not (child / "states").is_dir():
            continue
        if detect_source_utg_path(child) is None:
            continue
        pkg_type, app_name = split_package_type_and_app_name(child.name)
        entries.append(
            {
                "package_dir": child.resolve(),
                "package_name": child.name,
                "package_type": pkg_type,
                "app_name": app_name,
            }
        )
    return entries


def parse_only_package_crop_size():
    if not ONLY_PACKAGE_CROP_SIZE:
        return None, None
    if not isinstance(ONLY_PACKAGE_CROP_SIZE, (list, tuple)) or len(ONLY_PACKAGE_CROP_SIZE) < 2:
        raise ValueError("ONLY_PACKAGE_CROP_SIZE must be [top, bottom].")
    top = int(ONLY_PACKAGE_CROP_SIZE[0])
    bottom = int(ONLY_PACKAGE_CROP_SIZE[1])
    if top < 0 or bottom < 0:
        raise ValueError("ONLY_PACKAGE_CROP_SIZE values must be non-negative.")
    return top, bottom


def resolve_image_path(app_dir, image_rel, node_id):
    candidates = []
    if isinstance(image_rel, str) and image_rel.strip():
        p = Path(image_rel.strip())
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(app_dir / p)
            candidates.append(app_dir / "states" / p.name)

    if isinstance(node_id, str) and node_id.strip():
        node_name = Path(node_id).name
        stem = Path(node_name).stem
        for ext in IMAGE_EXTS:
            candidates.append(app_dir / "states" / f"{stem}{ext}")

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def estimate_crop_parameters(sample_paths):
    if not sample_paths:
        return 70, 140, {"mode": "fallback_no_samples"}

    row_means = []
    row_varxs = []
    height = None
    width = None
    used = 0

    for p in sample_paths:
        try:
            with Image.open(p) as img_raw:
                arr = np.array(img_raw.convert("L"))
        except Exception:
            continue

        if arr.ndim != 2 or arr.size == 0:
            continue

        h, w = arr.shape
        if height is None:
            height, width = h, w
        if h != height or w != width:
            continue

        row_means.append(arr.mean(axis=1))
        row_varxs.append(arr.var(axis=1))
        used += 1

    if used == 0 or height is None:
        return 70, 140, {"mode": "fallback_decode_failed"}

    row_means = np.stack(row_means, axis=0)
    row_varxs = np.stack(row_varxs, axis=0)

    mean_varx = np.mean(row_varxs, axis=0)
    std_across_imgs = np.std(row_means, axis=0)

    mean_row = np.mean(row_means, axis=0)
    p45_var = float(np.percentile(mean_varx, 45))
    p45_std = float(np.percentile(std_across_imgs, 45))

    top_max = max(int(height * 0.12), 20)
    bottom_max = max(int(height * 0.16), 20)
    top_min = max(int(height * 0.02), 8)
    bottom_min = max(int(height * 0.04), 16)

    raw_top = 0
    for r in range(min(top_max, height)):
        is_extreme = mean_row[r] <= 45 or mean_row[r] >= 210
        if mean_varx[r] <= p45_var and std_across_imgs[r] <= p45_std and is_extreme:
            raw_top += 1
        else:
            break

    raw_bottom = 0
    for k in range(1, min(bottom_max, height) + 1):
        r = height - k
        is_extreme = mean_row[r] <= 45 or mean_row[r] >= 210
        if mean_varx[r] <= p45_var and std_across_imgs[r] <= p45_std and is_extreme:
            raw_bottom += 1
        else:
            break

    # Dynamic fallback is still package-specific (derived from each package statistics).
    top_ratio = 0.03 + min(p45_std / 4000.0, 0.015) + min(p45_var / 20000.0, 0.01)
    bottom_ratio = 0.06 + min(p45_std / 3000.0, 0.03) + min(p45_var / 10000.0, 0.03)
    ratio_top = int(round(height * top_ratio))
    ratio_bottom = int(round(height * bottom_ratio))

    crop_top = raw_top if raw_top >= top_min else ratio_top
    crop_bottom = raw_bottom if raw_bottom >= bottom_min else ratio_bottom

    # Clamp to avoid over-cropping.
    crop_top = max(top_min, min(crop_top, top_max))
    crop_bottom = max(bottom_min, min(crop_bottom, bottom_max))

    if crop_top + crop_bottom >= int(height * 0.5):
        crop_top = ratio_top
        crop_bottom = ratio_bottom

    meta = {
        "mode": "auto",
        "sample_count": used,
        "image_width": width,
        "image_height": height,
        "raw_top": int(raw_top),
        "raw_bottom": int(raw_bottom),
        "threshold_var_p45": p45_var,
        "threshold_std_p45": p45_std,
        "fallback_top_ratio": top_ratio,
        "fallback_bottom_ratio": bottom_ratio,
    }
    return int(crop_top), int(crop_bottom), meta


def crop_and_save(src_path, dst_path, crop_top, crop_bottom):
    with Image.open(src_path) as img_raw:
        img = img_raw.convert("RGB")
        w, h = img.size
        if h <= crop_top + crop_bottom:
            return False, f"skip_too_small:{w}x{h}"
        cropped = img.crop((0, crop_top, w, h - crop_bottom))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dst_path)
    return True, ""


def cleanup_unused_output_images(output_app_dir, keep_image_names):
    keep = {str(x).strip() for x in keep_image_names if str(x).strip()}
    removed = []
    failed = []
    for p in output_app_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.name in keep:
            continue
        try:
            p.unlink()
            removed.append(p.name)
        except Exception:
            failed.append(p.name)
            continue
    return removed, failed


def image_digest(image_path):
    with Image.open(image_path) as img_raw:
        img = img_raw.convert("RGB")
        h = hashlib.md5()
        h.update(str(img.size).encode("utf-8"))
        h.update(img.tobytes())
        return h.hexdigest()


def extract_last_id_tail(line):
    text = str(line or "").strip()
    if not text:
        return ""

    matches = re.findall(r"id[:/]\s*([^\s,;>\]]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    return text


def normalize_node_content(node):
    raw = node.get("content")
    if (raw is None or raw == "") and isinstance(node.get("props"), dict):
        raw = node["props"].get("content", "")

    lines = [x.strip() for x in str(raw or "").splitlines() if x.strip()]
    if not lines:
        return ""

    selected = lines[-2:] if len(lines) >= 2 else lines
    picked = [extract_last_id_tail(x) for x in selected]
    picked = [x for x in picked if x]
    return "\n".join(picked)


def infer_node_name(node, fallback_id):
    label = str(node.get("label", "")).strip()
    if label and label != str(node.get("id", "")).strip():
        return label

    activity = str(node.get("activity", "")).strip()
    if activity:
        return activity.split(".")[-1].strip() or fallback_id

    return Path(fallback_id).stem or fallback_id


def infer_node_type(node):
    activity = str(node.get("activity", "")).strip().lower()
    if "login" in activity:
        return "login page"
    if "detail" in activity:
        return "detail page"
    if "list" in activity:
        return "list page"
    return "screen"


def extract_event_strings(edge):
    out = []
    events = edge.get("events")
    if isinstance(events, list):
        for ev in events:
            if isinstance(ev, dict):
                s = ev.get("event_str")
                if not s and ev.get("event_type"):
                    s = str(ev.get("event_type"))
            else:
                s = str(ev)
            if isinstance(s, str) and s.strip():
                out.append(s.strip())

    if not out and isinstance(edge.get("conditions"), list):
        for c in edge["conditions"]:
            s = str(c).strip()
            if s:
                out.append(s)

    if not out:
        for key in ("label", "title"):
            s = edge.get(key)
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
                break
    return out


def edge_has_non_empty_view_images(edge):
    events = edge.get("events")
    if not isinstance(events, list) or not events:
        return False

    for ev in events:
        if not isinstance(ev, dict):
            continue
        views = ev.get("view_images")
        if isinstance(views, list):
            for v in views:
                if isinstance(v, str) and v.strip():
                    return True
                if v is not None and not isinstance(v, str):
                    return True
        elif isinstance(views, str):
            if views.strip():
                return True
        elif views:
            return True
    return False


def parse_edge_label_value(edge):
    label = edge.get("label")
    if label is None:
        return float("inf")
    text = str(label).strip()
    if not text:
        return float("inf")
    try:
        return float(text)
    except Exception:
        m = re.search(r"\d+(\.\d+)?", text)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass
    return float("inf")


def simplify_condition(event_text):
    text = " ".join(str(event_text or "").split())
    if not text:
        return ""

    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if m:
        event_type = m.group(1)
    else:
        event_type = text.split("(", 1)[0].strip() or "Event"

    payload = ""
    m = re.search(r"\](.*?)\[/", text)
    if m:
        payload = m.group(1).strip()

    if not payload:
        m = re.search(r"alt='([^']+)'", text) or re.search(r'alt="([^"]+)"', text)
        if m:
            payload = m.group(1).strip()

    if not payload:
        m = re.search(r"intent='([^']+)'", text) or re.search(r'intent="([^"]+)"', text)
        if m:
            intent = m.group(1).strip()
            parts = intent.split()
            comp = parts[-1] if parts else intent
            if "/" in comp:
                comp = comp.split("/")[-1]
            payload = comp.strip(".").split(".")[-1].strip()

    payload = re.sub(r"\s+", " ", payload).strip().lower()
    if payload:
        return f"{event_type}({payload})"
    return f"{event_type}()"


def build_actions(to_node_id):
    return [
        "transfer interaction context",
        f"update current screen to {to_node_id}",
    ]


def process_app(
    app_dir,
    output_app_dir,
    fixed_crop_top=None,
    fixed_crop_bottom=None,
    package_type="",
    app_name="",
    remove_duplicate_images=False,
):
    utg_path = detect_source_utg_path(app_dir)
    if utg_path is None:
        raise FileNotFoundError(f"utg.js/utg.json not found in {app_dir}")

    output_utg_json = "utg.json"
    output_utg_js = "utg_processed.js"

    data = load_utg(utg_path)
    output_app_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        n_key, e_key = "nodes", "edges"
    elif isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        n_key, e_key = "N", "E"
    else:
        raise ValueError(f"Unsupported UTG schema in {utg_path}")

    source_nodes = data.get(n_key, [])
    source_edges = data.get(e_key, [])
    source_node_count = len(source_nodes)
    source_edge_count = len(source_edges)

    # Step 1: remove edges whose view_images are empty/missing.
    edges = []
    removed_empty_view_edge_ids = []
    for edge in source_edges:
        if edge_has_non_empty_view_images(edge):
            edges.append(edge)
        else:
            removed_empty_view_edge_ids.append(str(edge.get("id", "")))

    # Step 2: remove nodes that have no remaining valid edges.
    connected_old_node_ids = set()
    for edge in edges:
        a = str(edge.get("from", "")).strip()
        b = str(edge.get("to", "")).strip()
        if a:
            connected_old_node_ids.add(a)
        if b:
            connected_old_node_ids.add(b)

    nodes = []
    for node in source_nodes:
        nid = str(node.get("id", "")).strip()
        if not nid:
            continue
        if nid in connected_old_node_ids:
            nodes.append(node)

    removed_node_by_edge_filter_count = max(source_node_count - len(nodes), 0)

    sample_paths = []
    seen_sample = set()
    for node in nodes:
        nid = str(node.get("id", "")).strip()
        if not nid:
            continue
        p = resolve_image_path(app_dir, node.get("image"), nid)
        if p is None:
            continue
        sp = str(p)
        if sp in seen_sample:
            continue
        seen_sample.add(sp)
        sample_paths.append(p)
        if len(sample_paths) >= 40:
            break

    if fixed_crop_top is None or fixed_crop_bottom is None:
        crop_top, crop_bottom, crop_meta = estimate_crop_parameters(sample_paths)
    else:
        crop_top, crop_bottom = int(fixed_crop_top), int(fixed_crop_bottom)
        crop_meta = {
            "mode": "fixed",
            "fixed_crop_top": crop_top,
            "fixed_crop_bottom": crop_bottom,
            "sample_count": len(sample_paths),
        }

    node_records = []
    image_cache = {}
    crop_failures = []

    for node in nodes:
        old_id = str(node.get("id", "")).strip()
        if not old_id:
            continue

        src_img = resolve_image_path(app_dir, node.get("image"), old_id)
        if src_img is None:
            crop_failures.append({"node_id": old_id, "reason": "image_not_found"})
            continue

        dst_image_name = f"{src_img.stem}.png"
        dst_img = output_app_dir / dst_image_name

        if str(src_img) not in image_cache:
            ok, reason = crop_and_save(src_img, dst_img, crop_top=crop_top, crop_bottom=crop_bottom)
            if not ok:
                crop_failures.append({"node_id": old_id, "reason": reason, "image": str(src_img)})
                continue
            image_cache[str(src_img)] = dst_img
        else:
            dst_img = image_cache[str(src_img)]

        digest = image_digest(dst_img)
        node_records.append(
            {
                "old_id": old_id,
                "node": node,
                "src_image": src_img,
                "cropped_image": dst_img,
                "image_name": dst_img.name,
                "digest": digest,
            }
        )

    digest_groups = defaultdict(list)
    for r in node_records:
        digest_groups[r["digest"]].append(r)

    old_to_new = {}
    canonical_by_image = {}
    dedup_groups = []
    duplicate_image_files = set()

    for digest, group in digest_groups.items():
        group = sorted(group, key=lambda x: (x["image_name"], x["old_id"]))
        canonical = group[0]
        canonical_image = canonical["image_name"]
        canonical_by_image[canonical_image] = canonical

        all_old_ids = []
        all_images = []
        for rec in group:
            old_to_new[rec["old_id"]] = canonical_image
            all_old_ids.append(rec["old_id"])
            all_images.append(rec["image_name"])
            if rec["image_name"] != canonical_image:
                duplicate_image_files.add(rec["cropped_image"])

        if len(group) > 1:
            dedup_groups.append(
                {
                    "digest": digest,
                    "canonical_image": canonical_image,
                    "canonical_old_id": canonical["old_id"],
                    "node_count": len(group),
                    "all_old_ids": all_old_ids,
                    "all_images": all_images,
                }
            )

    if remove_duplicate_images:
        for dup_path in sorted(duplicate_image_files, key=lambda p: str(p)):
            if dup_path.exists():
                dup_path.unlink()

    new_nodes = []
    for image_name in sorted(canonical_by_image.keys()):
        rec = canonical_by_image[image_name]
        src_node = rec["node"]
        new_nodes.append(
            {
                "id": image_name,
                "name": infer_node_name(src_node, image_name),
                "type": infer_node_type(src_node),
                "props": {"content": normalize_node_content(src_node)},
            }
        )

    edge_bucket = {}
    main_page_node_id = ""
    main_page_edge_label_value = float("inf")
    for edge in edges:
        old_from = str(edge.get("from", "")).strip()
        old_to = str(edge.get("to", "")).strip()
        if not old_from or not old_to:
            continue
        if old_from not in old_to_new or old_to not in old_to_new:
            continue

        new_from = old_to_new[old_from]
        new_to = old_to_new[old_to]
        key = (new_from, new_to)

        label_value = parse_edge_label_value(edge)
        if label_value < main_page_edge_label_value:
            main_page_edge_label_value = label_value
            main_page_node_id = new_from

        if key not in edge_bucket:
            edge_bucket[key] = {
                "conditions": [],
                "source_edges": [],
            }

        conditions = []
        for event_str in extract_event_strings(edge):
            cond = simplify_condition(event_str)
            if cond:
                conditions.append(cond)
        if not conditions:
            conditions = ["UnknownEvent()"]

        edge_bucket[key]["conditions"].extend(conditions)
        edge_bucket[key]["source_edges"].append(str(edge.get("id", "")))

    new_edges = []
    for (new_from, new_to), info in sorted(edge_bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        conds = unique_keep_order(info["conditions"])
        new_edges.append(
            {
                "id": f"{new_from}-->{new_to}",
                "from": new_from,
                "to": new_to,
                "conditions": conds,
                "actions": build_actions(new_to),
            }
        )

    # Step 3: put main page node first in node list:
    # choose the "from" node of the smallest-label edge.
    if main_page_node_id:
        idx = -1
        for i, node in enumerate(new_nodes):
            if str(node.get("id", "")) == main_page_node_id:
                idx = i
                break
        if idx > 0:
            node_main = new_nodes.pop(idx)
            new_nodes.insert(0, node_main)

    # Remove images for deleted nodes in output directory.
    keep_image_names = [str(n.get("id", "")).strip() for n in new_nodes]
    removed_output_images, failed_removed_output_images = cleanup_unused_output_images(
        output_app_dir, keep_image_names
    )

    output_data = {"N": new_nodes, "E": new_edges}
    write_utg_json(output_app_dir / output_utg_json, output_data)
    if output_utg_js:
        write_utg_js(output_app_dir / output_utg_js, output_data)

    report = {
        "app_dir": str(app_dir),
        "output_app_dir": str(output_app_dir),
        "package_type": package_type,
        "app_name": app_name,
        "source_utg": str(utg_path),
        "output_utg_json": str(output_app_dir / output_utg_json),
        "output_utg_js": str(output_app_dir / output_utg_js) if output_utg_js else "",
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "crop_meta": crop_meta,
        "source_node_count": source_node_count,
        "source_edge_count": source_edge_count,
        "filtered_source_node_count": len(nodes),
        "filtered_source_edge_count": len(edges),
        "removed_empty_view_image_edge_count": len(removed_empty_view_edge_ids),
        "removed_empty_view_image_edge_ids_sample": removed_empty_view_edge_ids[:50],
        "removed_node_by_edge_filter_count": removed_node_by_edge_filter_count,
        "main_page_node_id": main_page_node_id,
        "main_page_edge_label_value": main_page_edge_label_value if main_page_node_id else None,
        "cropped_node_count": len(node_records),
        "new_node_count": len(new_nodes),
        "new_edge_count": len(new_edges),
        "dedup_group_count": len(dedup_groups),
        "removed_duplicate_node_count": max(len(node_records) - len(new_nodes), 0),
        "removed_output_image_count": len(removed_output_images),
        "removed_output_image_names_sample": removed_output_images[:50],
        "failed_remove_output_image_count": len(failed_removed_output_images),
        "failed_remove_output_image_names_sample": failed_removed_output_images[:50],
        "crop_failures": crop_failures,
        "dedup_groups": dedup_groups,
    }
    report_path = output_app_dir / "utg_process_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report



def main():
    package_path = INPUT_PACKAGE_PATH.resolve()
    output_root = OUTPUT_ROOT_PATH.resolve()
    if not package_path.exists():
        raise FileNotFoundError(f"package path not found: {package_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    package_entries = collect_package_entries(package_path)
    if not package_entries:
        print("No app directories with states + utg.js/utg.json found.")
        return

    single_override_top, single_override_bottom = parse_only_package_crop_size()
    has_single_crop_override = single_override_top is not None and single_override_bottom is not None

    all_reports = []
    package_list = []
    package_crop_size_list = []

    for entry in package_entries:
        app_dir = entry["package_dir"]
        pkg_type = entry["package_type"]
        app_name = entry["app_name"]
        package_name = entry["package_name"]
        try:
            output_app_dir = output_root / pkg_type / app_name
            use_single_override = has_single_crop_override and (
                (ONLY_PACKAGE_NAME and package_name == ONLY_PACKAGE_NAME)
                or (len(package_entries) == 1)
            )
            report = process_app(
                app_dir=app_dir,
                output_app_dir=output_app_dir,
                fixed_crop_top=single_override_top if use_single_override else None,
                fixed_crop_bottom=single_override_bottom if use_single_override else None,
                package_type=pkg_type,
                app_name=app_name,
                remove_duplicate_images=False,
            )
            all_reports.append(report)
            package_list.append(package_name)
            package_crop_size_list.append([report["crop_top"], report["crop_bottom"]])
            print(
                f"[OK] {package_name} -> {pkg_type}/{app_name}: nodes {report['source_node_count']} -> {report['new_node_count']}, "
                f"edges {report['source_edge_count']} -> {report['new_edge_count']}, "
                f"dedup_groups={report['dedup_group_count']}, "
                f"crop=({report['crop_top']},{report['crop_bottom']})"
            )
        except Exception as e:
            print(f"[FAIL] {package_name}: {e}")

    summary = {
        "package_path": str(package_path),
        "output_root": str(output_root),
        "app_classification_list": sorted(unique_keep_order([x["package_type"] for x in package_entries])),
        "package_list": package_list,
        "package_crop_size_list": package_crop_size_list,
        "output_structure": "output/<package_type>/<app_name>/",
        "crop_mode": "single_fixed_override" if has_single_crop_override else "auto_per_package",
        "only_package_name": ONLY_PACKAGE_NAME,
        "only_package_crop_size": ONLY_PACKAGE_CROP_SIZE,
        "processed_app_count": len(all_reports),
        "apps": all_reports,
    }
    summary_path = output_root / "utg_batch_process_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
