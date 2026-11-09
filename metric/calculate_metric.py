import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import math
# import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import re
import torch
from torch.nn.functional import cosine_similarity
import clip
import easyocr
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_metrics"

Method = {
    "E2E": {
        "root": (
            PROJECT_ROOT
            / "results"
            / "appium_screenshots"
            / "baselines"
            / "E2E"
        ),
    },
    "E2E-prompted": {
        "root": (
            PROJECT_ROOT
            / "results"
            / "appium_screenshots"
            / "baselines"
            / "E2E-prompted"
        ),
    },
    "GPA": {
        "root": (
            PROJECT_ROOT
            / "results"
            / "appium_screenshots"
            / "baselines"
            / "GPA"
        ),
    },
    "PageWeaver_GT": {
        "root": PROJECT_ROOT / "results" / "appium_screenshots" / "PageWeaver_GT",
    },
}


GROUNDTRUTH_CLUSTER_FILE = "UTG_Label.json"
GENERATED_UTG_FILE = "UTG.json"
RUN_REPORT_FILE = "run_report.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_SCREENSHOT_COUNT = 3
DIGITS = 4
ENABLE_CLIP = True
ENABLE_TEXT = True


_clip_model = None
_clip_preprocess = None
_ocr_reader = None
_ocr_cache: dict[Path, list[str]] = {}




def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def validate_paths() -> None:
    """Fail early with clear messages when required metric inputs are missing."""
    errors: List[str] = []
    warnings: List[str] = []

    if not DATA_ROOT.is_dir():
        errors.append(f"DATA_ROOT does not exist or is not a directory: {DATA_ROOT}")
    else:
        dataset_apps = [
            app_dir
            for category_dir in DATA_ROOT.iterdir()
            if category_dir.is_dir()
            for app_dir in category_dir.iterdir()
            if app_dir.is_dir()
        ]
        if not dataset_apps:
            errors.append(f"No app directories found under DATA_ROOT: {DATA_ROOT}")
        else:
            missing_gt = [
                str((app_dir / GROUNDTRUTH_CLUSTER_FILE).relative_to(PROJECT_ROOT))
                for app_dir in dataset_apps
                if not (app_dir / GROUNDTRUTH_CLUSTER_FILE).is_file()
            ]
            no_images = [
                str(app_dir.relative_to(PROJECT_ROOT))
                for app_dir in dataset_apps
                if screenshot_count(app_dir) == 0
            ]
            if missing_gt:
                warnings.append(
                    f"{len(missing_gt)} dataset apps are missing {GROUNDTRUTH_CLUSTER_FILE}; "
                    f"examples: {missing_gt[:5]}"
                )
            if no_images:
                warnings.append(
                    f"{len(no_images)} dataset apps have no screenshots; "
                    f"examples: {no_images[:5]}"
                )

    for baseline, config in Method.items():
        root = config["root"]
        if not root.is_dir():
            errors.append(f"Method root for {baseline} does not exist: {root}")
            continue
        app_dirs = [
            app_dir
            for category_dir in root.iterdir()
            if category_dir.is_dir()
            for app_dir in category_dir.iterdir()
            if app_dir.is_dir()
        ]
        if not app_dirs:
            warnings.append(f"Method root for {baseline} has no app directories: {root}")
            continue
        utg_count = sum(1 for app_dir in app_dirs if (app_dir / GENERATED_UTG_FILE).is_file())
        image_app_count = sum(1 for app_dir in app_dirs if screenshot_count(app_dir) > 0)
        if utg_count == 0:
            warnings.append(
                f"Method {baseline} has app directories but no {GENERATED_UTG_FILE}: {root}"
            )
        if image_app_count == 0:
            warnings.append(f"Method {baseline} has app directories but no screenshots: {root}")

    if errors:
        message = "\n".join(f"[PATH ERROR] {error}" for error in errors)
        if warnings:
            message += "\n" + "\n".join(f"[PATH WARN] {warning}" for warning in warnings)
        raise FileNotFoundError(message)

    print(f"[PATH OK] DATA_ROOT={DATA_ROOT}")
    for baseline, config in Method.items():
        print(f"[PATH OK] {baseline} root={config['root']}")
    for warning in warnings:
        print(f"[PATH WARN] {warning}")


def to_4f(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(f"{float(value):.{DIGITS}f}")


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.{DIGITS}f}"
    if value is None:
        return ""
    return value


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def screenshot_count(app_dir: Path) -> int:
    if not app_dir.is_dir():
        return 0
    return sum(
        1
        for path in app_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def app_is_eligible_for_metrics(app_dir: Path) -> bool:
    return screenshot_count(app_dir) >= MIN_SCREENSHOT_COUNT


def normalize_page_name(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    name = Path(text).name
    stem = Path(name).stem if Path(name).suffix else name
    if stem.endswith("Screen") and len(stem) > len("Screen"):
        stem = stem[:-len("Screen")]
    return stem


def image_original_name(screenshot_stem: str, original_stems: set) -> Optional[str]:
    if screenshot_stem in original_stems:
        return screenshot_stem
    normalized = normalize_page_name(screenshot_stem)
    if normalized in original_stems:
        return normalized
    return None


def ssim_similarity(image_path1: str, image_path2: str) -> float:
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    if image1 is None:
        raise FileNotFoundError(image_path1)
    if image2 is None:
        raise FileNotFoundError(image_path2)
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    score, _ = ssim(image1, image2_resized, full=True)
    return score


def mae(img1_path: str, img2_path: str) -> float:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB").resize(img1.size, Image.Resampling.LANCZOS)
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    mae = np.mean(np.abs(arr1 - arr2)) / 255.0
    return mae


def get_clip_model():
    global _clip_model, _clip_preprocess
    if not ENABLE_CLIP or clip is None or torch is None or cosine_similarity is None:
        return None, None, None
    if _clip_model is None or _clip_preprocess is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _clip_model, _clip_preprocess, device


def clip_similarity_score(image_path1: str, image_path2: str) -> Optional[float]:
    model, preprocess, device = get_clip_model()
    if model is None:
        return None
    img1 = preprocess(Image.open(image_path1).convert("RGB")).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(image_path2).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features1 = model.encode_image(img1)
        features2 = model.encode_image(img2)
    features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
    features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)
    return float(cosine_similarity(features1, features2).item())
def get_ocr_reader():
    global _ocr_reader
    if not ENABLE_TEXT or easyocr is None:
        return None
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(
            ["ch_sim", "en"],
            gpu=torch is not None and torch.cuda.is_available(),
            verbose=False,
        )
    return _ocr_reader


def tokenize_ocr_text(text: str) -> list[str]:
    # Treat each CJK character as a token while preserving Latin words/numbers.
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())


def image_text_tokens(path: Path) -> list[str]:
    reader = get_ocr_reader()
    if reader is None:
        return []
    resolved = path.resolve()
    if resolved not in _ocr_cache:
        fragments = reader.readtext(str(path), detail=0, paragraph=False)
        _ocr_cache[resolved] = tokenize_ocr_text(" ".join(map(str, fragments)))
    return _ocr_cache[resolved]


def bleu_text_similarity(original: Path, generated: Path) -> float:
    if not ENABLE_TEXT:
        return None
    reference = image_text_tokens(original)
    hypothesis = image_text_tokens(generated)
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    if sentence_bleu is None or SmoothingFunction is None:
        from difflib import SequenceMatcher
        return float(SequenceMatcher(None, reference, hypothesis).ratio())
    return float(sentence_bleu(
        [reference],
        hypothesis,
        smoothing_function=SmoothingFunction().method1,
    ))



def get_page_score(original_img: Path, screenshot_img: Path) -> Dict[str, Optional[float]]:
    text_score = to_4f(bleu_text_similarity(original_img, screenshot_img))
    ssim_score = to_4f(ssim_similarity(str(original_img), str(screenshot_img)))
    clip_score = to_4f(clip_similarity_score(str(original_img), str(screenshot_img)))
    mae_score = to_4f(mae(str(original_img), str(screenshot_img)))
    return {
        "clip_similarity": clip_score,
        "text_similarity": text_score,
        "structure_similarity": ssim_score,
        "mae": mae_score,
    }


def build_cluster_map(cluster_data: dict) -> Tuple[Dict[str, str], int]:
    cluster_map: Dict[str, str] = {}
    duplicate_states = 0
    pages = cluster_data.get("pages", [])
    if not isinstance(pages, list):
        return cluster_map, duplicate_states

    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_id = str(page.get("page_id") or f"P{page_index}").strip()
        states = page.get("states", [])
        if not isinstance(states, list):
            continue
        for state in states:
            if not isinstance(state, dict):
                continue
            state_id = normalize_page_name(state.get("mockup"))
            if not state_id:
                continue
            if state_id in cluster_map and cluster_map[state_id] != page_id:
                duplicate_states += 1
                continue
            cluster_map[state_id] = page_id
    return cluster_map, duplicate_states


def build_utg_gold_cluster_map(utg_data: dict) -> Tuple[Dict[str, str], int]:
    cluster_map: Dict[str, str] = {}
    duplicate_states = 0
    nodes = utg_data.get("N")
    if not isinstance(nodes, list):
        nodes = utg_data.get("nodes", [])
    if not isinstance(nodes, list):
        return cluster_map, duplicate_states

    for node in nodes:
        if not isinstance(node, dict):
            continue
        state_id = normalize_page_name(node.get("id") or node.get("name"))
        if not state_id:
            continue
        cluster = str(node.get("page_cluster", "")).strip() or "__MISSING_PAGE_CLUSTER__"
        if state_id in cluster_map and cluster_map[state_id] != cluster:
            duplicate_states += 1
            continue
        cluster_map[state_id] = cluster
    return cluster_map, duplicate_states


def build_generated_utg_cluster_map(utg_data: dict) -> Tuple[Dict[str, str], int]:
    cluster_map: Dict[str, str] = {}
    duplicate_states = 0
    nodes = utg_data.get("N")
    if not isinstance(nodes, list):
        nodes = utg_data.get("nodes", [])
    if not isinstance(nodes, list):
        return cluster_map, duplicate_states

    for node in nodes:
        if not isinstance(node, dict):
            continue
        state_id = normalize_page_name(node.get("id") or node.get("name"))
        page = str(node.get("page", "")).strip()
        if not state_id or not page:
            continue
        if state_id in cluster_map and cluster_map[state_id] != page:
            duplicate_states += 1
            continue
        cluster_map[state_id] = page
    return cluster_map, duplicate_states


def build_edge_set(utg_data: dict) -> set[Tuple[str, str]]:
    edges = utg_data.get("E")
    if not isinstance(edges, list):
        edges = utg_data.get("edges", [])
    if not isinstance(edges, list):
        return set()
    return {
        (str(edge.get("from", "")).strip(), str(edge.get("to", "")).strip())
        for edge in edges
        if isinstance(edge, dict)
        and str(edge.get("from", "")).strip()
        and str(edge.get("to", "")).strip()
    }


def split_edges_by_gold_cluster(
    edges: set[Tuple[str, str]],
    gold_cluster_map: Dict[str, str],
) -> tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    """Split gold edges into intra-page and page-navigation edges.

    The split is based on whether the two endpoint states belong to the same
    groundtruth page cluster. Edges whose endpoint cluster is unavailable are
    treated as page-navigation edges so they are not over-counted as intra-page
    interactions.
    """
    intra_page_edges: set[Tuple[str, str]] = set()
    page_navigation_edges: set[Tuple[str, str]] = set()
    for source_id, target_id in edges:
        source_cluster = gold_cluster_map.get(normalize_page_name(source_id))
        target_cluster = gold_cluster_map.get(normalize_page_name(target_id))
        if source_cluster and target_cluster and source_cluster == target_cluster:
            intra_page_edges.add((source_id, target_id))
        else:
            page_navigation_edges.add((source_id, target_id))
    return intra_page_edges, page_navigation_edges


def build_success_edge_set(run_report: dict) -> set[Tuple[str, str]]:
    results = run_report.get("results", [])
    if not isinstance(results, list):
        return set()
    return {
        (str(result.get("from", "")).strip(), str(result.get("to", "")).strip())
        for result in results
        if isinstance(result, dict)
        and str(result.get("status", "")).lower() == "success"
        and str(result.get("from", "")).strip()
        and str(result.get("to", "")).strip()
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def comb2(value: int) -> int:
    return value * (value - 1) // 2 if value >= 2 else 0


def contingency_counts(y_true: List[str], y_pred: List[str]) -> Tuple[Counter, Counter, Counter]:
    return Counter(y_true), Counter(y_pred), Counter(zip(y_true, y_pred))


def entropy(label_count: Counter, total: int) -> float:
    value = 0.0
    for count in label_count.values():
        probability = count / total if total else 0
        if probability:
            value -= probability * math.log(probability)
    return value


def mutual_information(true_count: Counter, pred_count: Counter, contingency: Counter, total: int) -> float:
    value = 0.0
    if total == 0:
        return value
    for (true_label, pred_label), count in contingency.items():
        numerator = count * total
        denominator = true_count[true_label] * pred_count[pred_label]
        value += (count / total) * math.log(numerator / denominator)
    return value


def adjusted_rand_index(true_count: Counter, pred_count: Counter, contingency: Counter, total: int) -> float:
    if total < 2:
        return 0.0
    sum_comb_contingency = sum(comb2(count) for count in contingency.values())
    sum_comb_true = sum(comb2(count) for count in true_count.values())
    sum_comb_pred = sum(comb2(count) for count in pred_count.values())
    total_pairs = comb2(total)
    expected_index = safe_div(sum_comb_true * sum_comb_pred, total_pairs)
    max_index = 0.5 * (sum_comb_true + sum_comb_pred)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0 if sum_comb_contingency == max_index else 0.0
    return (sum_comb_contingency - expected_index) / denominator


def information_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    total = len(y_true)
    true_count, pred_count, contingency = contingency_counts(y_true, y_pred)
    h_true = entropy(true_count, total)
    h_pred = entropy(pred_count, total)
    mi = mutual_information(true_count, pred_count, contingency, total)
    homogeneity = 1.0 if h_true == 0.0 else 1.0 - safe_div(h_true - mi, h_true)
    completeness = 1.0 if h_pred == 0.0 else 1.0 - safe_div(h_pred - mi, h_pred)
    nmi = 1.0 if h_true == 0.0 and h_pred == 0.0 else safe_div(mi, (h_true + h_pred) / 2.0)
    return {
        "ari": adjusted_rand_index(true_count, pred_count, contingency, total),
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": f1_from_pr(homogeneity, completeness),
    }


def pairwise_stats(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    true_count, pred_count, contingency = contingency_counts(y_true, y_pred)
    tp = sum(comb2(count) for count in contingency.values())
    pred_same = sum(comb2(count) for count in pred_count.values())
    true_same = sum(comb2(count) for count in true_count.values())
    fp = pred_same - tp
    fn = true_same - tp
    total_pairs = comb2(len(y_true))
    return {
        "pairwise_precision": safe_div(tp, tp + fp),
        "pairwise_recall": safe_div(tp, tp + fn),
        "pairwise_f1": f1_from_pr(safe_div(tp, tp + fp), safe_div(tp, tp + fn)),
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "pair_total": total_pairs,
    }


def evaluate_clustering(gold_map: Dict[str, str], pred_map: Dict[str, str]) -> Dict[str, float]:
    state_ids = sorted(gold_map)
    y_true = [gold_map[state_id] for state_id in state_ids]
    y_pred = [pred_map.get(state_id, "__MISSING_PRED_STATE__") for state_id in state_ids]
    info = information_metrics(y_true, y_pred) if state_ids else {
        "ari": 0.0,
        "nmi": 0.0,
        "homogeneity": 0.0,
        "completeness": 0.0,
        "v_measure": 0.0,
    }
    pairwise = pairwise_stats(y_true, y_pred) if state_ids else {
        "pairwise_precision": 0.0,
        "pairwise_recall": 0.0,
        "pairwise_f1": 0.0,
        "pair_tp": 0,
        "pair_fp": 0,
        "pair_fn": 0,
        "pair_total": 0,
    }
    return {
        **info,
        **pairwise,
        "gold_state_count": len(gold_map),
        "pred_state_count": len(pred_map),
        "missing_pred_state_count": sum(1 for state_id in state_ids if state_id not in pred_map),
        "extra_pred_state_count": sum(1 for state_id in pred_map if state_id not in gold_map),
        "gold_cluster_count": len(set(gold_map.values())),
        "pred_cluster_count": len(set(pred_map.values())),
    }


def evaluate_implemented_clustering(
    gold_map: Dict[str, str],
    pred_map: Dict[str, str],
) -> Optional[Dict[str, float]]:
    """Evaluate clustering quality only on state IDs present in both maps."""
    matched_ids = sorted(set(gold_map) & set(pred_map))
    if not matched_ids:
        return None
    matched_gold = {state_id: gold_map[state_id] for state_id in matched_ids}
    matched_pred = {state_id: pred_map[state_id] for state_id in matched_ids}
    return evaluate_clustering(matched_gold, matched_pred)


def iter_apps() -> Iterable[Tuple[str, str, Path]]:
    for category_dir in sorted(DATA_ROOT.iterdir(), key=lambda path: path.name):
        if not category_dir.is_dir():
            continue

        for app_dir in sorted(category_dir.iterdir(), key=lambda path: path.name):
            if app_dir.is_dir():
                yield category_dir.name, app_dir.name, app_dir


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})


def page_summary_rows(page_rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in page_rows:
        groups[(row["baseline"], row["category"], row["app"])].append(row)

    metric_keys = [
        "clip_similarity",
        "text_similarity",
        "structure_similarity",
        "mae",
    ]
    app_rows = []
    for (baseline, category, app), group in sorted(groups.items()):
        item = {
            "baseline": baseline,
            "category": category,
            "app": app,
            "matched_page_count": len(group),
        }
        for key in metric_keys:
            item[f"{key}_mean"] = to_4f(mean(row.get(key) for row in group))
        app_rows.append(item)

    baseline_groups = defaultdict(list)
    for row in app_rows:
        baseline_groups[row["baseline"]].append(row)

    rows = []
    for baseline, group in sorted(baseline_groups.items()):
        item = {
            "baseline": baseline,
            "app_count": len(group),
            "matched_page_count": sum(int(row["matched_page_count"]) for row in group),
        }
        for key in metric_keys:
            item[f"{key}_mean"] = to_4f(mean(row.get(f"{key}_mean") for row in group))
        rows.append(item)
    return rows


def page_coverage_summary_rows(page_coverage_rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in page_coverage_rows:
        groups[row["baseline"]].append(row)

    rows = []
    for baseline, group in sorted(groups.items()):
        gold_count = sum(int(row["gold_page_count"]) for row in group)
        matched_count = sum(int(row["matched_page_count"]) for row in group)
        rows.append({
            "baseline": baseline,
            "app_count": len(group),
            "gold_page_count": gold_count,
            "matched_page_count": matched_count,
            "missing_page_count": gold_count - matched_count,
            "page_coverage_micro": to_4f(safe_div(matched_count, gold_count)),
            "page_coverage_macro": to_4f(mean(row["page_coverage"] for row in group)),
        })
    return rows


def cluster_summary_rows(cluster_rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in cluster_rows:
        if isinstance(row.get("ari"), (int, float)):
            groups[row["baseline"]].append(row)

    metric_keys = [
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "implemented_ari",
        "implemented_nmi",
        "implemented_homogeneity",
        "implemented_completeness",
        "implemented_v_measure",
        "implemented_pairwise_precision",
        "implemented_pairwise_recall",
        "implemented_pairwise_f1",
        "cluster_missing_page_rate",
        "cluster_state_coverage",
    ]
    rows = []
    for baseline, group in sorted(groups.items()):
        tp = sum(int(row["pair_tp"]) for row in group)
        fp = sum(int(row["pair_fp"]) for row in group)
        fn = sum(int(row["pair_fn"]) for row in group)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        implemented_tp = sum(
            int(row["implemented_pair_tp"])
            for row in group
            if row.get("implemented_pair_tp") is not None
        )
        implemented_fp = sum(
            int(row["implemented_pair_fp"])
            for row in group
            if row.get("implemented_pair_fp") is not None
        )
        implemented_fn = sum(
            int(row["implemented_pair_fn"])
            for row in group
            if row.get("implemented_pair_fn") is not None
        )
        implemented_precision = safe_div(
            implemented_tp, implemented_tp + implemented_fp
        )
        implemented_recall = safe_div(
            implemented_tp, implemented_tp + implemented_fn
        )
        gold_page_count = sum(int(row["gold_state_count"]) for row in group)
        implemented_page_count = sum(
            int(row["implemented_state_count"]) for row in group
        )
        missing_page_count = gold_page_count - implemented_page_count
        item = {
            "baseline": baseline,
            "app_count": len({(row["category"], row["app"]) for row in group}),
            "result_count": len(group),
            "cluster_gold_page_count": gold_page_count,
            "cluster_implemented_page_count": implemented_page_count,
            "cluster_missing_page_count": missing_page_count,
            "cluster_missing_page_rate_micro": to_4f(
                safe_div(missing_page_count, gold_page_count)
            ),
            "cluster_state_coverage_micro": to_4f(
                safe_div(implemented_page_count, gold_page_count)
            ),
        }
        for key in metric_keys:
            item[f"{key}_mean"] = to_4f(mean(row.get(key) for row in group))
        item["pairwise_micro_precision"] = to_4f(precision)
        item["pairwise_micro_recall"] = to_4f(recall)
        item["pairwise_micro_f1"] = to_4f(f1_from_pr(precision, recall))
        item["pair_tp"] = tp
        item["pair_fp"] = fp
        item["pair_fn"] = fn
        item["implemented_pairwise_micro_precision"] = to_4f(
            implemented_precision
        )
        item["implemented_pairwise_micro_recall"] = to_4f(implemented_recall)
        item["implemented_pairwise_micro_f1"] = to_4f(
            f1_from_pr(implemented_precision, implemented_recall)
        )
        item["implemented_pair_tp"] = implemented_tp
        item["implemented_pair_fp"] = implemented_fp
        item["implemented_pair_fn"] = implemented_fn
        rows.append(item)
    return rows


def interaction_summary_rows(interaction_rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in interaction_rows:
        groups[row["baseline"]].append(row)

    rows = []
    for baseline, group in sorted(groups.items()):
        gold_count = sum(int(row["gold_edge_count"]) for row in group)
        generated_count = sum(int(row["generated_edge_count"]) for row in group)
        implemented_count = sum(int(row["implemented_edge_count"]) for row in group)
        intra_required_count = sum(
            int(row["intra_page_required_interaction_count"]) for row in group
        )
        intra_completed_count = sum(
            int(row["intra_page_completed_interaction_count"]) for row in group
        )
        navigation_required_count = sum(
            int(row["page_navigation_required_interaction_count"]) for row in group
        )
        navigation_completed_count = sum(
            int(row["page_navigation_completed_interaction_count"]) for row in group
        )
        completion_rate = safe_div(implemented_count, gold_count)
        intra_completion_rate = safe_div(intra_completed_count, intra_required_count)
        navigation_completion_rate = safe_div(
            navigation_completed_count,
            navigation_required_count,
        )
        precision = safe_div(implemented_count, generated_count)
        recall = completion_rate
        rows.append({
            "baseline": baseline,
            "app_count": len(group),
            "required_interaction_count": gold_count,
            "completed_interaction_count": implemented_count,
            "interaction_completion_rate": to_4f(completion_rate),
            "intra_page_required_interaction_count": intra_required_count,
            "intra_page_completed_interaction_count": intra_completed_count,
            "intra_page_interaction_completion_rate": to_4f(intra_completion_rate),
            "page_navigation_required_interaction_count": navigation_required_count,
            "page_navigation_completed_interaction_count": navigation_completed_count,
            "page_navigation_interaction_completion_rate": to_4f(
                navigation_completion_rate
            ),
            "gold_edge_count": gold_count,
            "generated_edge_count": generated_count,
            "implemented_edge_count": implemented_count,
            "missing_edge_count": gold_count - implemented_count,
            "interaction_precision_micro": to_4f(precision),
            "interaction_recall_micro": to_4f(recall),
            "interaction_f1_micro": to_4f(f1_from_pr(precision, recall)),
            "interaction_precision_macro": to_4f(
                mean(row["interaction_precision"] for row in group)
            ),
            "interaction_recall_macro": to_4f(
                mean(row["interaction_recall"] for row in group)
            ),
            "interaction_f1_macro": to_4f(mean(row["interaction_f1"] for row in group)),
        })
    return rows


def availability_summary_rows(availability_rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in availability_rows:
        groups[row["baseline"]].append(row)

    rows = []
    for baseline, group in sorted(groups.items()):
        status_counts = Counter(row["result_status"] for row in group)
        failed_app_count = sum(int(row["failed_app"]) for row in group)
        app_folder_count = len(group)
        rows.append({
            "baseline": baseline,
            "app_folder_count": app_folder_count,
            "eligible_app_count": app_folder_count - failed_app_count,
            "failed_app_count": failed_app_count,
            "failed_app_rate": to_4f(safe_div(failed_app_count, app_folder_count)),
            "minimum_screenshot_count": MIN_SCREENSHOT_COUNT,
            "report_only_count": status_counts["report_only"],
            "excluded_from_metrics_count": sum(
                int(row["excluded_from_metrics"]) for row in group
            ),
            "report_and_images_no_utg_count": status_counts["report_and_images_no_utg"],
            "utg_and_images_count": status_counts["utg_and_images"],
            "utg_only_count": status_counts["utg_only"],
            "images_only_count": status_counts["images_only"],
            "empty_count": status_counts["empty"],
        })
    return rows


def attach_failure_metrics(
    summary_rows: List[dict],
    availability_summaries: List[dict],
) -> List[dict]:
    failure_by_baseline = {
        row["baseline"]: row for row in availability_summaries
    }
    for row in summary_rows:
        availability = failure_by_baseline.get(row["baseline"], {})
        row.update({
            "total_app_count": availability.get("app_folder_count", 0),
            "eligible_app_count": availability.get("eligible_app_count", 0),
            "failed_app_count": availability.get("failed_app_count", 0),
            "failed_app_rate": availability.get("failed_app_rate", 0.0),
            "minimum_screenshot_count": MIN_SCREENSHOT_COUNT,
        })
    return summary_rows


def calculate_page_metrics() -> List[dict]:
    rows = []
    app_items = list(iter_apps())
    iterator = tqdm(app_items, desc="page metrics") if tqdm else app_items
    for category, app, original_app_dir in iterator:
        original_images = {path.stem: path for path in original_app_dir.glob("*.png")}
        if not original_images:
            continue
        original_stems = set(original_images)
        for baseline, config in Method.items():
            screenshot_app_dir = config["root"] / category / app
            if not app_is_eligible_for_metrics(screenshot_app_dir):
                continue
            for screenshot_img in sorted(screenshot_app_dir.glob("*.png"), key=lambda path: path.name):
                original_stem = image_original_name(screenshot_img.stem, original_stems)
                if not original_stem:
                    continue
                original_img = original_images[original_stem]
                try:
                    scores = get_page_score(original_img, screenshot_img)
                    error = ""
                except Exception as exc:
                    scores = {
                        "clip_similarity": None,
                        "text_similarity": None,
                        "structure_similarity": None,
                        "mae": None,
                    }
                    error = str(exc)
                rows.append({
                    "baseline": baseline,
                    "category": category,
                    "app": app,
                    "original_mockup": original_stem,
                    "screenshot": screenshot_img.name,
                    "original_image": str(original_img.relative_to(PROJECT_ROOT)),
                    "screenshot_image": str(screenshot_img.relative_to(PROJECT_ROOT)),
                    **scores,
                    "error": error,
                })
    return rows


def calculate_result_availability_metrics() -> List[dict]:
    rows = []
    for category, app, _original_app_dir in iter_apps():
        for baseline, config in Method.items():
            generated_app_dir = config["root"] / category / app
            if not generated_app_dir.is_dir():
                continue
            has_report = (generated_app_dir / RUN_REPORT_FILE).is_file()
            has_utg = (generated_app_dir / GENERATED_UTG_FILE).is_file()
            image_count = screenshot_count(generated_app_dir)
            failed_app = image_count < MIN_SCREENSHOT_COUNT
            has_images = image_count > 0
            if has_report and not has_utg and not has_images:
                status = "report_only"
            elif has_report and has_images and not has_utg:
                status = "report_and_images_no_utg"
            elif has_utg and has_images:
                status = "utg_and_images"
            elif has_utg:
                status = "utg_only"
            elif has_images:
                status = "images_only"
            else:
                status = "empty"
            rows.append({
                "baseline": baseline,
                "category": category,
                "app": app,
                "result_status": status,
                "report_only": int(status == "report_only"),
                "failed_app": int(failed_app),
                "failure_reason": (
                    f"screenshot_count_below_{MIN_SCREENSHOT_COUNT}"
                    if failed_app
                    else ""
                ),
                "minimum_screenshot_count": MIN_SCREENSHOT_COUNT,
                "excluded_from_metrics": int(failed_app),
                "has_report": int(has_report),
                "has_utg": int(has_utg),
                "screenshot_count": image_count,
                "folder": str(generated_app_dir.relative_to(PROJECT_ROOT)),
            })
    return rows


def calculate_page_coverage_metrics(page_rows: List[dict]) -> List[dict]:
    matched_by_app = defaultdict(set)
    for row in page_rows:
        matched_by_app[(row["baseline"], row["category"], row["app"])].add(
            row["original_mockup"]
        )

    rows = []
    for category, app, original_app_dir in iter_apps():
        original_states = {path.stem for path in original_app_dir.glob("*.png")}
        if not original_states:
            continue
        for baseline, config in Method.items():
            generated_app_dir = config["root"] / category / app
            if not app_is_eligible_for_metrics(generated_app_dir):
                continue
            matched_states = matched_by_app.get((baseline, category, app), set())
            missing_states = sorted(original_states - matched_states)
            rows.append({
                "baseline": baseline,
                "category": category,
                "app": app,
                "gold_page_count": len(original_states),
                "matched_page_count": len(matched_states),
                "missing_page_count": len(missing_states),
                "page_coverage": to_4f(safe_div(len(matched_states), len(original_states))),
                "missing_pages": "|".join(missing_states),
            })
    return rows


def calculate_cluster_metrics() -> List[dict]:
    rows = []
    for category, app, original_app_dir in iter_apps():
        gt_path = original_app_dir / GROUNDTRUTH_CLUSTER_FILE
        if not gt_path.exists():
            continue
        gold_map, duplicate_gold = build_utg_gold_cluster_map(load_json(gt_path))
        if not gold_map:
            continue
        for baseline, config in Method.items():
            generated_app_dir = config["root"] / category / app
            if not app_is_eligible_for_metrics(generated_app_dir):
                continue
            pred_path = generated_app_dir / GENERATED_UTG_FILE
            if not pred_path.exists():
                continue
            pred_map, duplicate_pred = build_generated_utg_cluster_map(load_json(pred_path))
            matched_state_count = len(set(gold_map) & set(pred_map))
            if not pred_map:
                status = "missing_page_fields"
            elif not matched_state_count:
                status = "no_state_id_overlap"
            else:
                status = "ok"
            metric = evaluate_clustering(gold_map, pred_map) if pred_map else None
            implemented_metric = evaluate_implemented_clustering(gold_map, pred_map)
            missing_page_count = len(gold_map) - matched_state_count
            missing_page_rate = safe_div(missing_page_count, len(gold_map))
            metric_values = {
                "ari": "",
                "nmi": "",
                "homogeneity": "",
                "completeness": "",
                "v_measure": "",
                "pairwise_precision": "",
                "pairwise_recall": "",
                "pairwise_f1": "",
                "pair_tp": "",
                "pair_fp": "",
                "pair_fn": "",
                "pair_total": "",
                "gold_state_count": len(gold_map),
                "pred_state_count": len(pred_map),
                "matched_state_count": matched_state_count,
                "missing_pred_state_count": len(gold_map),
                "extra_pred_state_count": 0,
                "gold_cluster_count": len(set(gold_map.values())),
                "pred_cluster_count": 0,
                "implemented_state_count": matched_state_count,
                "cluster_missing_page_count": missing_page_count,
                "cluster_missing_page_rate": to_4f(missing_page_rate),
                "cluster_state_coverage": to_4f(
                    safe_div(matched_state_count, len(gold_map))
                ),
                "implemented_ari": None,
                "implemented_nmi": None,
                "implemented_homogeneity": None,
                "implemented_completeness": None,
                "implemented_v_measure": None,
                "implemented_pairwise_precision": None,
                "implemented_pairwise_recall": None,
                "implemented_pairwise_f1": None,
                "implemented_pair_tp": None,
                "implemented_pair_fp": None,
                "implemented_pair_fn": None,
                "implemented_pair_total": None,
            }
            if metric:
                metric_values.update({
                    key: to_4f(value) if isinstance(value, float) else value
                    for key, value in metric.items()
                })
            if implemented_metric:
                for key in [
                    "ari",
                    "nmi",
                    "homogeneity",
                    "completeness",
                    "v_measure",
                    "pairwise_precision",
                    "pairwise_recall",
                    "pairwise_f1",
                    "pair_tp",
                    "pair_fp",
                    "pair_fn",
                    "pair_total",
                ]:
                    value = implemented_metric[key]
                    metric_values[f"implemented_{key}"] = (
                        to_4f(value) if isinstance(value, float) else value
                    )
            rows.append({
                "baseline": baseline,
                "category": category,
                "app": app,
                "status": status,
                "groundtruth_file": str(gt_path.relative_to(PROJECT_ROOT)),
                "prediction_file": (
                    str(pred_path.relative_to(PROJECT_ROOT)) if pred_path.exists() else ""
                ),
                **metric_values,
                "duplicate_gold_state_count": duplicate_gold,
                "duplicate_pred_state_count": duplicate_pred,
            })
    return rows


def calculate_interaction_metrics() -> List[dict]:
    rows = []
    for category, app, original_app_dir in iter_apps():
        gold_path = original_app_dir / GROUNDTRUTH_CLUSTER_FILE
        if not gold_path.exists():
            continue
        gold_utg = load_json(gold_path)
        gold_edges = build_edge_set(gold_utg)
        gold_cluster_map, _duplicate_gold = build_utg_gold_cluster_map(gold_utg)
        intra_page_gold_edges, page_navigation_gold_edges = split_edges_by_gold_cluster(
            gold_edges,
            gold_cluster_map,
        )
        for baseline, config in Method.items():
            generated_app_dir = config["root"] / category / app
            if not app_is_eligible_for_metrics(generated_app_dir):
                continue

            generated_utg_path = generated_app_dir / GENERATED_UTG_FILE
            run_report_path = generated_app_dir / RUN_REPORT_FILE
            has_images = True
            if generated_utg_path.exists():
                generated_edges = build_edge_set(load_json(generated_utg_path))
                prediction_file = generated_utg_path
                source = "generated_utg"
            elif run_report_path.exists() and has_images:
                generated_edges = build_success_edge_set(load_json(run_report_path))
                prediction_file = run_report_path
                source = "successful_run_report_edges"
            else:
                continue

            implemented = gold_edges & generated_edges
            missing = gold_edges - generated_edges
            extra = generated_edges - gold_edges
            intra_page_implemented = implemented & intra_page_gold_edges
            page_navigation_implemented = implemented & page_navigation_gold_edges
            precision = safe_div(len(implemented), len(generated_edges))
            recall = safe_div(len(implemented), len(gold_edges))
            intra_page_completion_rate = safe_div(
                len(intra_page_implemented),
                len(intra_page_gold_edges),
            )
            page_navigation_completion_rate = safe_div(
                len(page_navigation_implemented),
                len(page_navigation_gold_edges),
            )
            rows.append({
                "baseline": baseline,
                "category": category,
                "app": app,
                "source": source,
                "groundtruth_file": str(gold_path.relative_to(PROJECT_ROOT)),
                "prediction_file": (
                    str(prediction_file.relative_to(PROJECT_ROOT)) if prediction_file else ""
                ),
                "required_interaction_count": len(gold_edges),
                "completed_interaction_count": len(implemented),
                "interaction_completion_rate": to_4f(recall),
                "intra_page_required_interaction_count": len(intra_page_gold_edges),
                "intra_page_completed_interaction_count": len(intra_page_implemented),
                "intra_page_interaction_completion_rate": to_4f(
                    intra_page_completion_rate
                ),
                "page_navigation_required_interaction_count": len(page_navigation_gold_edges),
                "page_navigation_completed_interaction_count": len(page_navigation_implemented),
                "page_navigation_interaction_completion_rate": to_4f(
                    page_navigation_completion_rate
                ),
                "gold_edge_count": len(gold_edges),
                "generated_edge_count": len(generated_edges),
                "implemented_edge_count": len(implemented),
                "missing_edge_count": len(missing),
                "extra_edge_count": len(extra),
                "interaction_precision": to_4f(precision),
                "interaction_recall": to_4f(recall),
                "interaction_f1": to_4f(f1_from_pr(precision, recall)),
                "implemented_edges": "|".join(
                    f"{source_id}-->{target_id}" for source_id, target_id in sorted(implemented)
                ),
                "missing_edges": "|".join(
                    f"{source_id}-->{target_id}" for source_id, target_id in sorted(missing)
                ),
                "extra_edges": "|".join(
                    f"{source_id}-->{target_id}" for source_id, target_id in sorted(extra)
                ),
            })
    return rows


def combined_app_rows(
    page_rows: List[dict],
    availability_rows: List[dict],
    page_coverage_rows: List[dict],
    cluster_rows: List[dict],
    interaction_rows: List[dict],
) -> List[dict]:
    page_groups = defaultdict(list)
    for row in page_rows:
        page_groups[(row["baseline"], row["category"], row["app"])].append(row)

    cluster_by_app = {
        (row["baseline"], row["category"], row["app"]): row
        for row in cluster_rows
    }
    page_coverage_by_app = {
        (row["baseline"], row["category"], row["app"]): row
        for row in page_coverage_rows
    }
    availability_by_app = {
        (row["baseline"], row["category"], row["app"]): row
        for row in availability_rows
    }
    interaction_by_app = {
        (row["baseline"], row["category"], row["app"]): row
        for row in interaction_rows
    }
    keys = sorted(
        set(availability_by_app)
        | set(page_groups)
        | set(page_coverage_by_app)
        | set(cluster_by_app)
        | set(interaction_by_app)
    )
    rows = []
    for baseline, category, app in keys:
        pages = page_groups.get((baseline, category, app), [])
        availability = availability_by_app.get((baseline, category, app), {})
        page_coverage = page_coverage_by_app.get((baseline, category, app), {})
        cluster = cluster_by_app.get((baseline, category, app), {})
        interaction = interaction_by_app.get((baseline, category, app), {})
        row = {
            "baseline": baseline,
            "category": category,
            "app": app,
            "result_status": availability.get("result_status", ""),
            "report_only": availability.get("report_only", ""),
            "failed_app": availability.get("failed_app", ""),
            "failure_reason": availability.get("failure_reason", ""),
            "excluded_from_metrics": availability.get("excluded_from_metrics", ""),
            "has_report": availability.get("has_report", ""),
            "has_utg": availability.get("has_utg", ""),
            "screenshot_count": availability.get("screenshot_count", ""),
            "minimum_screenshot_count": availability.get(
                "minimum_screenshot_count", ""
            ),
            "gold_page_count": page_coverage.get("gold_page_count", ""),
            "matched_page_count": page_coverage.get("matched_page_count", len(pages)),
            "missing_page_count": page_coverage.get("missing_page_count", ""),
            "page_coverage": page_coverage.get("page_coverage", ""),
            "clip_similarity_mean": to_4f(mean(item.get("clip_similarity") for item in pages)),
            "text_similarity_mean": to_4f(mean(item.get("text_similarity") for item in pages)),
            "structure_similarity_mean": to_4f(mean(item.get("structure_similarity") for item in pages)),
            "mae_mean": to_4f(mean(item.get("mae") for item in pages)),
            "cluster_status": cluster.get("status", ""),
        }
        for key in [
            "ari",
            "nmi",
            "homogeneity",
            "completeness",
            "v_measure",
            "pairwise_precision",
            "pairwise_recall",
            "pairwise_f1",
            "gold_state_count",
            "pred_state_count",
            "matched_state_count",
            "missing_pred_state_count",
            "extra_pred_state_count",
            "implemented_state_count",
            "cluster_missing_page_count",
            "cluster_missing_page_rate",
            "cluster_state_coverage",
            "implemented_ari",
            "implemented_nmi",
            "implemented_homogeneity",
            "implemented_completeness",
            "implemented_v_measure",
            "implemented_pairwise_precision",
            "implemented_pairwise_recall",
            "implemented_pairwise_f1",
        ]:
            row[key] = cluster.get(key, "")
        for key in [
            "required_interaction_count",
            "completed_interaction_count",
            "interaction_completion_rate",
            "intra_page_required_interaction_count",
            "intra_page_completed_interaction_count",
            "intra_page_interaction_completion_rate",
            "page_navigation_required_interaction_count",
            "page_navigation_completed_interaction_count",
            "page_navigation_interaction_completion_rate",
            "gold_edge_count",
            "generated_edge_count",
            "implemented_edge_count",
            "missing_edge_count",
            "extra_edge_count",
            "interaction_precision",
            "interaction_recall",
            "interaction_f1",
        ]:
            row[key] = interaction.get(key, "")
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_paths()

    page_rows = calculate_page_metrics()
    availability_rows = calculate_result_availability_metrics()
    page_coverage_rows = calculate_page_coverage_metrics(page_rows)
    cluster_rows = calculate_cluster_metrics()
    interaction_rows = calculate_interaction_metrics()
    app_rows = combined_app_rows(
        page_rows,
        availability_rows,
        page_coverage_rows,
        cluster_rows,
        interaction_rows,
    )

    availability_summaries = availability_summary_rows(availability_rows)
    page_summaries = attach_failure_metrics(
        page_summary_rows(page_rows), availability_summaries
    )
    page_coverage_summaries = attach_failure_metrics(
        page_coverage_summary_rows(page_coverage_rows), availability_summaries
    )
    cluster_summaries = attach_failure_metrics(
        cluster_summary_rows(cluster_rows), availability_summaries
    )
    interaction_summaries = attach_failure_metrics(
        interaction_summary_rows(interaction_rows), availability_summaries
    )

    write_csv(OUTPUT_DIR / "page_metrics.csv", page_rows)
    write_csv(OUTPUT_DIR / "page_metric_summary.csv", page_summaries)
    write_csv(OUTPUT_DIR / "result_availability_metrics.csv", availability_rows)
    write_csv(
        OUTPUT_DIR / "result_availability_summary.csv",
        availability_summaries,
    )
    write_csv(OUTPUT_DIR / "page_coverage_metrics.csv", page_coverage_rows)
    write_csv(
        OUTPUT_DIR / "page_coverage_summary.csv",
        page_coverage_summaries,
    )
    write_csv(OUTPUT_DIR / "cluster_metrics.csv", cluster_rows)
    write_csv(OUTPUT_DIR / "cluster_metric_summary.csv", cluster_summaries)
    write_csv(OUTPUT_DIR / "interaction_metrics.csv", interaction_rows)
    write_csv(
        OUTPUT_DIR / "interaction_metric_summary.csv",
        interaction_summaries,
    )
    write_csv(OUTPUT_DIR / "app_metrics.csv", app_rows)

    print(f"[DONE] page rows: {len(page_rows)}")
    print(
        "[DONE] report-only folders: "
        f"{sum(int(row['report_only']) for row in availability_rows)}"
    )
    for summary in availability_summaries:
        print(
            f"[AVAILABILITY] {summary['baseline']}: "
            f"failed_apps={summary['failed_app_count']}, "
            f"failure_rate={summary['failed_app_rate']}, "
            f"report_only={summary['report_only_count']}, "
            f"report_and_images_no_utg={summary['report_and_images_no_utg_count']}, "
            f"utg_and_images={summary['utg_and_images_count']}, "
            f"images_only={summary['images_only_count']}"
        )
    print(f"[DONE] page coverage rows: {len(page_coverage_rows)}")
    print(f"[DONE] cluster rows: {len(cluster_rows)}")
    print(f"[DONE] interaction rows: {len(interaction_rows)}")
    print(f"[DONE] app rows: {len(app_rows)}")
    print(f"[GT] {GROUNDTRUTH_CLUSTER_FILE} node.page_cluster")
    print(f"[OUT] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
