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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "code_results" / "baselines" / "E2E-prompted"
STATE_PATH = OUTPUT_ROOT / "state.json"
PROMPT_PATH = PROJECT_ROOT / "code_generate" / "utils" / "prompt_templates" / "Baselines-End2End" / "E2E-prompted.txt"
CATEGORY_FILTER = None
APP_FILTER = None
MAX_IMAGE_SIDE = 768
SLEEP_SECONDS = 0
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

API_BASE = ""
MODEL_NAME = ""
API_KEY = ""


def create_client():
    api_key = API_KEY.strip()
    if not api_key:
        raise RuntimeError("API_KEY is empty. Please fill your model API key in API_KEY.")

    return OpenAI(
        base_url=API_BASE,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=API_BASE,
            follow_redirects=True,
        ),
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


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


def load_UTG(app_dir: Path):
    return json.loads(read_text(app_dir / "UTG.json"))


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


def iter_apps():
    for category_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        if CATEGORY_FILTER and category_dir.name != CATEGORY_FILTER:
            continue
        for app_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            if APP_FILTER and app_dir.name != APP_FILTER:
                continue
            if (app_dir / "UTG.json").exists():
                yield category_dir.name, app_dir.name, app_dir


def strip_code_fence(code: str) -> str:
    code = code.strip()
    match = re.fullmatch(r"```(?:[\w.+-]+)?\s*(.*?)\s*```", code, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    return code.rstrip() + "\n"


def safe_output_path(output_dir: Path, file_name: str) -> Path:
    file_name = file_name.strip().replace("\\", "/")
    base_name = Path(file_name).name
    if not base_name or base_name in {".", ".."} or ":" in base_name:
        raise ValueError(f"Invalid file name: {file_name}")
    return output_dir / base_name


def save_code_files(response_text: str, output_dir: Path):
    pattern = re.compile(r"@@@(?P<file_name>[^@\r\n]+?)@@@\s*", re.DOTALL)
    matches = list(pattern.finditer(response_text))
    if not matches:
        raise ValueError("No @@@File Name@@@ blocks found in API response.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, match in enumerate(matches):
        file_name = match.group("file_name").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response_text)
        code = strip_code_fence(response_text[start:end])
        output_path = safe_output_path(output_dir, file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code, encoding="utf-8")
        saved_paths.append(output_path)
    return saved_paths


def generate_app(client, prompt_template: str, category: str, app: str, app_dir: Path):
    utg = load_UTG(app_dir)
    prompt = prompt_template.format(UTG=json.dumps(utg, ensure_ascii=False, indent=2))
    content = [{"type": "text", "text": prompt}]
    for mockup_id in get_mockup_ids(utg):
        image_path = resolve_image_path(app_dir, mockup_id)
        if image_path is None:
            print(f"[WARN] missing image: {category}/{app}/{mockup_id}")
            continue
        content.append({"type": "text", "text": f"Mockup ID: {mockup_id}"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(image_path)}})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
    )
    response_text = response.choices[0].message.content or ""
    output_dir = OUTPUT_ROOT / category / app
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "res.txt").write_text(response_text, encoding="utf-8")
    saved_paths = save_code_files(response_text, output_dir)
    print(f"[SAVE] {category}/{app}: {len(saved_paths)} files -> {output_dir}")


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
            generate_app(client, prompt_template, category, app, app_dir)
            state[category][app] = 1
            save_state(state)
        except Exception as error:
            print(f"[FAIL] {category}/{app}: {error}")
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
