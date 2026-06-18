from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path

import httpx
from openai import OpenAI
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data_Find"
PROMPT_PATH = PROJECT_ROOT / "page_cluster" / "utils" / "Baselines-Clustering" / "baseline_UTG+Mockups.txt"
BASELINE_NAME = "baseline_UTG+Mockups"
STATE_Main_Path = PROJECT_ROOT / "page_cluster" /"baseline"
STATE_PATH = STATE_Main_Path / "baseline_states" / BASELINE_NAME / "state.json"
# MODEL = "claude-opus-4-7"
MODEL = "gpt-5"
CATEGORY_FILTER = None
APP_FILTER = None
MAX_IMAGE_SIDE = 768
SLEEP_SECONDS = 0
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def create_client():
    return OpenAI(
        base_url="http://47.76.75.25:9000/v1",
        api_key="sk-lrcGmVpnlczcF5lGB1CfFeAa60Cb4eAb87A982F025009e69",
        http_client=httpx.Client(
            base_url="http://47.76.75.25:9000/v1",
            follow_redirects=True,
        ),
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_state():
    if STATE_PATH.exists():
        return json.loads(read_text(STATE_PATH))
    return {}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare_prompt_template(prompt_template: str) -> str:
    return (
        prompt_template
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("{{UTG}}", "{UTG}")
    )


def normalize_node_id(node_id) -> str:
    name = Path(str(node_id or "").split("?")[0].split("#")[0]).name
    lower_name = name.lower()
    for suffix in IMAGE_SUFFIXES:
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_utg_process(app_dir: Path):
    return json.loads(read_text(app_dir / "UTG_process.json"))


def get_mockup_ids(utg) -> list[str]:
    nodes = utg.get("N", utg.get("nodes", []))
    mockup_ids = []
    for node in nodes:
        if isinstance(node, dict):
            node_id = str(node.get("id", "")).strip()
        else:
            node_id = str(node).strip()
        if node_id:
            mockup_ids.append(node_id)
    return mockup_ids


def resolve_image_path(app_dir: Path, mockup_id: str):
    normalized = normalize_node_id(mockup_id)
    candidates = [mockup_id] + [f"{normalized}{suffix}" for suffix in IMAGE_SUFFIXES]
    for candidate in dict.fromkeys(candidates):
        path = app_dir / candidate
        if path.exists() and path.is_file():
            return path
    return None


def encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    image_type = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    with Image.open(path) as image:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if image_type == "jpeg" and image.mode in {"RGBA", "P"}:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG" if image_type == "jpeg" else "PNG")
    return f"data:image/{image_type};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def extract_json(text: str):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def iter_apps():
    for category_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        if CATEGORY_FILTER and category_dir.name != CATEGORY_FILTER:
            continue
        for app_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            if APP_FILTER and app_dir.name != APP_FILTER:
                continue
            if (app_dir / "UTG_process.json").exists():
                yield category_dir.name, app_dir.name, app_dir


def cluster_app(client, prompt_template: str, category: str, app: str, app_dir: Path):
    utg = load_utg_process(app_dir)
    prompt = prompt_template.format(
        UTG=json.dumps(utg, ensure_ascii=False, indent=2),
    )
    content = [{"type": "text", "text": prompt}]
    for mockup_id in get_mockup_ids(utg):
        image_path = resolve_image_path(app_dir, mockup_id)
        if image_path is None:
            print(f"[WARN] missing image: {category}/{app}/{mockup_id}")
            continue
        content.append({"type": "text", "text": f"Mockup ID: {mockup_id}"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(image_path)}})

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
    )
    result = extract_json(response.choices[0].message.content or "")

    output_dir = app_dir / "Cluster"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{BASELINE_NAME}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {category}/{app} -> {output_path}")


def main():
    client = create_client()
    prompt_template = prepare_prompt_template(read_text(PROMPT_PATH))
    state = load_state()
    for category, app, app_dir in iter_apps():
        if category not in state:
            state[category] = {}
        if app not in state[category]:
            state[category][app] = 0
            save_state(state)
        if state[category][app] == 1:
            print(f"[SKIP] {category}/{app}: already completed")
            continue
        try:
            cluster_app(client, prompt_template, category, app, app_dir)
            state[category][app] = 1
            save_state(state)
        except Exception as error:
            print(f"[FAIL] {category}/{app}: {error}")
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
