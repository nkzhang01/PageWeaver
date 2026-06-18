from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data_Find"
COMPONENT_ROOT = PROJECT_ROOT / "results" / "extract_results"
PROMPT_PATH = PROJECT_ROOT / "page_cluster" / "utils" / "Baselines-Clustering" / "baseline_Component.txt"
BASELINE_NAME = "baseline_Component"
STATE_Main_Path = PROJECT_ROOT / "page_cluster" /"baseline"
STATE_PATH = STATE_Main_Path / "baseline_states" / BASELINE_NAME / "state.json"
# MODEL = "claude-opus-4-7"
MODEL = "gpt-5"
CATEGORY_FILTER = None
APP_FILTER = None
SLEEP_SECONDS = 0


def create_client():
    return OpenAI(
        base_url="http://47.76.75.25:9000/v1",
        api_key="sk-lRr1ugMYXkV3M2uP3e58260f30A644B9Bd9d2a20A17cB13d",
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
        .replace("{{Component_Descriptions}}", "{Component_Descriptions}")
    )


def load_component_descriptions(component_dir: Path) -> str:
    parts = []
    for type_path in sorted(component_dir.glob("*_typeProcess.txt")):
        content = read_text(type_path).strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


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
            if (app_dir / "UTG.json").exists() or (app_dir / "UTG_process.json").exists():
                yield category_dir.name, app_dir.name, app_dir


def cluster_app(client, prompt_template: str, category: str, app: str, app_dir: Path):
    component_dir = COMPONENT_ROOT / category / app
    component_descriptions = load_component_descriptions(component_dir)
    if not component_descriptions:
        raise FileNotFoundError(f"No *_typeProcess.txt files found in {component_dir}")

    prompt = prompt_template.format(Component_Descriptions=component_descriptions)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
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
