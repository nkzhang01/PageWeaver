from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data_Find"
COMPONENT_ROOT = PROJECT_ROOT / "results" / "extract_results"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "code_results" / "baseline_claude" / "baseline_UTG+Component"
STATE_PATH = OUTPUT_ROOT / "state.json"
PROMPT_PATH = PROJECT_ROOT / "code_generaction" / "utils" / "prompt_templates" / "Baselines-End2End" / "baseline_UTG+Component.txt"
# MODEL = "gpt-5"
MODEL = "claude-opus-4-7"
CATEGORY_FILTER = None
APP_FILTER = None
SLEEP_SECONDS = 0


def create_client():
    return OpenAI(
        base_url="http://47.76.75.25:9000/v1",
        api_key="sk-tlvJixOFH5cY8PPEzdAJqRQwTAHhQEmPsMxNpCTrhNiVM8Ra",
        http_client=httpx.Client(
            base_url="http://47.76.75.25:9000/v1",
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
        .replace("{{Component_Descriptions}}", "{Component_Descriptions}")
        .replace("{{UTG}}", "{UTG}")
    )


def load_component_descriptions(component_dir: Path) -> str:
    parts = []
    for type_path in sorted(component_dir.glob("*_typeProcess.txt")):
        content = read_text(type_path).strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def load_utg_process(app_dir: Path):
    return json.loads(read_text(app_dir / "UTG_process.json"))


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


def iter_apps():
    for category_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        if CATEGORY_FILTER and category_dir.name != CATEGORY_FILTER:
            continue
        for app_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            if APP_FILTER and app_dir.name != APP_FILTER:
                continue
            if (app_dir / "UTG_process.json").exists():
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
    utg = load_utg_process(app_dir)
    component_dir = COMPONENT_ROOT / category / app
    component_descriptions = load_component_descriptions(component_dir)
    if not component_descriptions:
        raise FileNotFoundError(f"No *_typeProcess.txt files found in {component_dir}")

    prompt = prompt_template.format(
        Component_Descriptions=component_descriptions,
        UTG=json.dumps(utg, ensure_ascii=False, indent=2),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
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
