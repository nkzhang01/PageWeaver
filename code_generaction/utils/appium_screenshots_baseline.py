from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
APPIUM_IMPORT_ERROR: Exception | None = None
# except Exception as exc:  # pragma: no cover - import guard for environments without Appium
#     webdriver = None  # type: ignore[assignment]
#     UiAutomator2Options = None  # type: ignore[assignment]
#     AppiumBy = None  # type: ignore[assignment]
#     WebDriverWait = None  # type: ignore[assignment]
#     EC = None  # type: ignore[assignment]
#     NoSuchElementException = Exception  # type: ignore[assignment]
#     TimeoutException = Exception  # type: ignore[assignment]
#     WebDriverException = Exception  # type: ignore[assignment]
#     APPIUM_IMPORT_ERROR = exc

try:
    from compiler.RN_compiler import run_gradlew_assemble
except ImportError:
    from code_generaction.utils.compiler.RN_compiler import run_gradlew_assemble

try:
    from label_component import (
        OUTPUT_FILENAME as LABEL_OUTPUT_FILENAME,
        get_output_res_path,
        process_res_file,
        split_res_sections,
    )
except ImportError:
    from code_generaction.utils.label_component import (
        OUTPUT_FILENAME as LABEL_OUTPUT_FILENAME,
        get_output_res_path,
        process_res_file,
        split_res_sections,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_NAME = "baseline_UTG+Mockups"
GENERATED_CODE_ROOT = REPO_ROOT / "results" / "code_results" / "baseline" / BASELINE_NAME
OUTPUT_ROOT = REPO_ROOT / "results" / "appium_screenshots_baseline"/BASELINE_NAME

# Edit these values directly before running this file.
PROJECT_PATH = Path(r"D:\testAndroid\AwesomeProject")
BATCH_MODE = True
CATEGORY = None
APP_NAME = None
APPIUM_SERVER_URL = "http://127.0.0.1:4723"
DEVICE_NAME = "emulator-5554"
APP_PACKAGE = "com.awesomeproject"
APP_ACTIVITY = "com.awesomeproject.MainActivity"
DEVICE_SERIAL: str | None = None
NO_RESET = True
CLEAR_COMPONENTS = True
WRITE_LABEL_FILE = True
SKIP_COMPILE = True
SKIP_EXISTING_OUTPUT = True
STARTUP_RENDER_WAIT_SECONDS = 5.0
SETTLE_SECONDS = 2.0
PRE_INTERACTION_RENDER_WAIT_SECONDS = 2.0
RENDER_WAIT_POLL_SECONDS = 0.2
RENDER_WAIT_STABLE_POLLS = 2
ELEMENT_TIMEOUT_SECONDS = 5
CURRENT_PAGE_WAIT_SECONDS = 2.0
BACK_RECOVERY_SETTLE_SECONDS = 1.0

INTERACTIVE_ID_RE = re.compile(
    r"^(?P<page>[A-Za-z0-9_]+)(?:\.(?P<component>[A-Za-z0-9_]+))?\.(?P<type>[A-Za-z0-9_]+)\.(?P<index>\d+)$"
)
INVALID_LABEL_PAGE_NAMES = {"UnknownPage"}


@dataclass(frozen=True)
class InteractionTarget:
    component_id: str
    page_name: str
    component_name: str | None
    component_type: str
    index: int
    text: str
    locator_strategy: str
    locator_value: str


class CompileFailure(RuntimeError):
    def __init__(self, stage: str, detail: str):
        super().__init__(detail)
        self.stage = stage
        self.detail = detail


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def resolve_device_serial() -> str | None:
    serial = normalize_text(DEVICE_SERIAL)
    if serial:
        return serial
    fallback = normalize_text(DEVICE_NAME)
    return fallback or None


def adb_base_command() -> list[str]:
    command = ["adb"]
    device_serial = resolve_device_serial()
    if device_serial:
        command.extend(["-s", device_serial])
    return command


def run_capture(command: list[str], timeout: int = 30) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    return (result.stdout or "") + "\n" + (result.stderr or "")


def ensure_generated_code_dir(category: str, app_name: str) -> Path:
    generated_code_dir = GENERATED_CODE_ROOT / category / app_name
    if not generated_code_dir.exists():
        raise FileNotFoundError(f"Generated code directory not found: {generated_code_dir}")
    return generated_code_dir


def has_existing_app_output(category: str, app_name: str) -> bool:
    output_dir = OUTPUT_ROOT / category / app_name
    return (output_dir / "run_report.json").exists() or (output_dir / "UTG.json").exists()


def discover_app_targets() -> list[tuple[str, str]]:
    if not GENERATED_CODE_ROOT.exists():
        raise FileNotFoundError(f"Generated code root not found: {GENERATED_CODE_ROOT}")

    requested_category = normalize_text(CATEGORY)
    requested_app_name = normalize_text(APP_NAME)
    targets: list[tuple[str, str]] = []

    if requested_category and requested_app_name:
        if SKIP_EXISTING_OUTPUT and has_existing_app_output(requested_category, requested_app_name):
            print(f"[skip] existing output: {requested_category}/{requested_app_name}")
            return []
        return [(requested_category, requested_app_name)]

    category_dirs = [GENERATED_CODE_ROOT / requested_category] if requested_category else sorted(
        path for path in GENERATED_CODE_ROOT.iterdir() if path.is_dir()
    )
    for category_dir in category_dirs:
        if not category_dir.exists() or not category_dir.is_dir():
            continue
        for app_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            if requested_app_name and app_dir.name != requested_app_name:
                continue
            if (app_dir / "res.txt").exists():
                if SKIP_EXISTING_OUTPUT and has_existing_app_output(category_dir.name, app_dir.name):
                    print(f"[skip] existing output: {category_dir.name}/{app_dir.name}")
                    continue
                targets.append((category_dir.name, app_dir.name))
    return targets


def copy_generated_code_to_project(generated_code_dir: Path, project_path: Path, clear_components: bool) -> None:
    components_dir = project_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    if clear_components:
        for child in components_dir.iterdir():
            if child.is_file() and child.suffix.lower() in {".js", ".jsx"}:
                child.unlink()

    copied_count = 0
    for source_path in sorted(generated_code_dir.iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in {".js", ".jsx"}:
            continue
        destination_path = project_path / source_path.name if source_path.name == "App.js" else components_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_count += 1
        print(f"[copy] {source_path.name} -> {destination_path}")

    if copied_count == 0:
        raise RuntimeError(f"No .js/.jsx files found in: {generated_code_dir}")


def materialize_labeled_res_to_project(res_label_path: Path, project_path: Path, clear_components: bool) -> None:
    if not res_label_path.exists():
        raise FileNotFoundError(f"Labeled res file not found: {res_label_path}")

    sections = split_res_sections(res_label_path.read_text(encoding="utf-8", errors="ignore"))
    components_dir = project_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    if clear_components:
        for child in components_dir.iterdir():
            if child.is_file() and child.suffix.lower() in {".js", ".jsx"}:
                child.unlink()

    for section in sections:
        file_name = Path(section.name).name
        destination_path = project_path / file_name if file_name == "App.js" else components_dir / file_name
        destination_path.write_text(section.code.rstrip() + "\n", encoding="utf-8")
        print(f"[label-copy] {section.name} -> {destination_path}")


def extract_packages_from_res_file(res_path: Path) -> list[str]:
    if not res_path.exists():
        return []

    packages: set[str] = set()
    code = res_path.read_text(encoding="utf-8", errors="ignore")
    for line in code.splitlines():
        line = line.strip()
        if not line.startswith("import ") or " from " not in line:
            continue
        package_name = line.split(" from ", 1)[1].rstrip(";").strip().strip("'\"")
        if package_name.startswith((".", "/", "\\")):
            continue
        if package_name.startswith("@"):
            parts = package_name.split("/")
            if len(parts) >= 2:
                packages.add("/".join(parts[:2]))
        else:
            packages.add(package_name.split("/")[0])

    excluded = {
        "react",
        "react-native",
        "@react-navigation/native",
        "@react-navigation/stack",
        "@react-navigation/native-stack",
        "react-native-gesture-handler",
        "react-native-safe-area-context",
        "react-native-screens",
        "@react-native-community/slider",
    }
    return sorted(package for package in packages if package and package not in excluded)


def find_latest_apk(project_path: Path) -> Path:
    apk_root = project_path / "android" / "app" / "build" / "outputs" / "apk"
    apks = sorted(apk_root.rglob("*.apk"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not apks:
        raise FileNotFoundError(f"No APK found under: {apk_root}")
    debug_apks = [path for path in apks if "debug" in path.name.lower() or "debug" in str(path.parent).lower()]
    return debug_apks[0] if debug_apks else apks[0]


def install_apk(apk_path: Path) -> None:
    command = adb_base_command() + ["install", "-r", "-d", str(apk_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install APK: {result.stderr or result.stdout}")
    print(f"[apk] installed: {apk_path}")


def compile_project(project_path: Path, category: str, app_name: str, res_path: Path) -> Path:
    packages = extract_packages_from_res_file(res_path)
    print(f"[compile] add packages: {packages}")
    compile_state, compile_error = run_gradlew_assemble(str(project_path), category, app_name, packages)
    if not compile_state:
        raise CompileFailure(
            stage=f"compile:{res_path.name}",
            detail=f"React Native compile failed: {compile_error}",
        )
    apk_path = find_latest_apk(project_path)
    print(f"[apk] built: {apk_path}")
    try:
        install_apk(apk_path)
    except Exception as exc:
        raise CompileFailure(stage=f"install:{res_path.name}", detail=str(exc)) from exc
    return apk_path


def build_driver() -> webdriver.Remote:
    if APPIUM_IMPORT_ERROR is not None:
        raise ImportError(f"Appium dependencies are not available: {APPIUM_IMPORT_ERROR}") from APPIUM_IMPORT_ERROR
    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.automation_name = "UiAutomator2"
    options.device_name = DEVICE_NAME
    options.udid = DEVICE_NAME
    options.no_reset = NO_RESET
    options.new_command_timeout = 300
    options.app_package = APP_PACKAGE
    if APP_ACTIVITY:
        options.app_activity = APP_ACTIVITY
    print(f"[appium] connecting: {APPIUM_SERVER_URL}")
    return webdriver.Remote(APPIUM_SERVER_URL, options=options)


def restart_app(driver: webdriver.Remote) -> None:
    try:
        driver.terminate_app(APP_PACKAGE)
    except Exception:
        pass
    driver.activate_app(APP_PACKAGE)
    print("[appium] app restarted")
    if STARTUP_RENDER_WAIT_SECONDS > 0:
        wait_for_page_render(driver, STARTUP_RENDER_WAIT_SECONDS, "startup render")


def interruptible_sleep(seconds: float, step: float = 0.2) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(step, remaining))


def get_render_snapshot(driver: webdriver.Remote) -> str:
    page_source = getattr(driver, "page_source", "")
    if not isinstance(page_source, str):
        page_source = str(page_source)
    return hashlib.md5(page_source.encode("utf-8", errors="ignore")).hexdigest()


def wait_for_page_render(
    driver: webdriver.Remote,
    max_wait_seconds: float | None = None,
    reason: str = "page render",
) -> None:
    wait_seconds = PRE_INTERACTION_RENDER_WAIT_SECONDS if max_wait_seconds is None else max_wait_seconds
    if wait_seconds <= 0:
        return

    print(f"[appium] waiting up to {wait_seconds}s for {reason}")
    deadline = time.monotonic() + wait_seconds
    last_snapshot: str | None = None
    stable_polls = 0

    while time.monotonic() < deadline:
        try:
            snapshot = get_render_snapshot(driver)
        except Exception as exc:
            print(f"[appium] render probe failed during {reason}: {exc}; falling back to sleep")
            interruptible_sleep(wait_seconds)
            return

        if snapshot == last_snapshot:
            stable_polls += 1
        else:
            last_snapshot = snapshot
            stable_polls = 0

        if stable_polls >= RENDER_WAIT_STABLE_POLLS:
            print(f"[appium] render settled for {reason}")
            return

        interruptible_sleep(min(RENDER_WAIT_POLL_SECONDS, max(0.0, deadline - time.monotonic())))

    print(f"[appium] render wait timed out after {wait_seconds}s for {reason}")


def screenshot_name_from_page(page_name: str, index: int) -> str:
    safe_name = normalize_text(page_name).replace(".", "_")
    return f"{safe_name}_{index}.png"


def get_label_components(label_value: str) -> tuple[str | None, str | None, str | None, int | None]:
    match = INTERACTIVE_ID_RE.match(label_value)
    if not match:
        return None, None, None, None
    component_name = match.group("component")
    component_type = match.group("type")
    index = int(match.group("index"))
    return match.group("page"), component_name, component_type, index


def candidate_id_values(element) -> list[str]:
    values = []
    for attr_name in ("resource-id", "content-desc", "contentDescription"):
        try:
            value = normalize_text(element.get_attribute(attr_name))
        except Exception:
            value = ""
        if value:
            if "/" in value:
                values.append(value.split("/")[-1])
            values.append(value)
    return values


def extract_descendant_texts(element) -> list[str]:
    texts: list[str] = []
    try:
        descendants = element.find_elements(AppiumBy.XPATH, ".//*")
    except Exception:
        return texts

    for child in descendants:
        try:
            text_value = normalize_text(child.get_attribute("text"))
        except Exception:
            text_value = ""
        if text_value:
            texts.append(text_value)
    return texts


def extract_target_text(element, component_id: str) -> str:
    try:
        text_value = normalize_text(element.get_attribute("text"))
    except Exception:
        text_value = ""
    if text_value:
        return text_value

    descendant_texts = extract_descendant_texts(element)
    if descendant_texts:
        return descendant_texts[0]

    page_name, component_name, component_type, _ = get_label_components(component_id)
    fallback = component_name or component_type or page_name or component_id
    return normalize_text(fallback)


def find_labeled_elements(driver: webdriver.Remote) -> list[InteractionTarget]:
    candidates = []
    seen_ids: set[str] = set()
    seen_elements: set[str] = set()

    for xpath_query, locator_label in (
        ('//*[@resource-id]', "resource-id"),
        ('//*[@content-desc]', "content-desc"),
    ):
        try:
            elements = driver.find_elements(AppiumBy.XPATH, xpath_query)
        except Exception:
            continue

        for element in elements:
            try:
                element_key = getattr(element, "id", None) or repr(element)
            except Exception:
                element_key = repr(element)
            if element_key in seen_elements:
                continue

            component_id = ""
            locator_value = ""
            locator_strategy = ""
            for attr_name, strategy_name in (("resource-id", "resource-id"), ("content-desc", "accessibility")):
                try:
                    raw_value = normalize_text(element.get_attribute(attr_name))
                except Exception:
                    raw_value = ""
                if not raw_value:
                    continue
                locator_candidate = raw_value.split("/")[-1] if "/" in raw_value else raw_value
                if INTERACTIVE_ID_RE.match(locator_candidate):
                    component_id = locator_candidate
                    locator_value = locator_candidate
                    locator_strategy = strategy_name
                    break

            if not component_id or component_id in seen_ids:
                continue

            page_name, component_name, component_type, index = get_label_components(component_id)
            if not page_name or not component_type or index is None:
                continue
            if page_name in INVALID_LABEL_PAGE_NAMES:
                continue
            seen_elements.add(element_key)
            seen_ids.add(component_id)

            candidates.append(
                InteractionTarget(
                    component_id=component_id,
                    page_name=page_name,
                    component_name=component_name,
                    component_type=component_type,
                    index=index,
                    text=extract_target_text(element, component_id),
                    locator_strategy=locator_strategy,
                    locator_value=locator_value,
                )
            )
    return sorted(candidates, key=lambda item: (item.page_name, item.component_type, item.index, item.component_id))


def get_current_page_info(driver: webdriver.Remote) -> tuple[str, str | None]:
    labeled_elements = find_labeled_elements(driver)
    if labeled_elements:
        return labeled_elements[0].page_name, labeled_elements[0].component_id

    all_candidates = []
    try:
        elements = driver.find_elements(AppiumBy.XPATH, "//*[@resource-id or @content-desc]")
    except Exception:
        elements = []

    for element in elements:
        for value in candidate_id_values(element):
            page_name, _, _, _ = get_label_components(value)
            if page_name and page_name not in INVALID_LABEL_PAGE_NAMES:
                all_candidates.append((page_name, value))
    if all_candidates:
        page_name, component_id = all_candidates[0]
        return page_name, component_id
    raise RuntimeError("Cannot determine current page from labeled components or root container.")


def find_element_for_target(driver: webdriver.Remote, target: InteractionTarget):
    locators = [
        (AppiumBy.XPATH, f'//*[@resource-id="{target.locator_value}"]'),
        (AppiumBy.XPATH, f'//*[@content-desc="{target.locator_value}"]'),
        (AppiumBy.XPATH, f'//*[@resource-id="{APP_PACKAGE}:id/{target.locator_value}"]'),
    ]

    last_error: Exception | None = None
    for by, value in locators:
        try:
            return WebDriverWait(driver, ELEMENT_TIMEOUT_SECONDS).until(
                EC.presence_of_element_located((by, value))
            )
        except Exception as exc:
            last_error = exc
            continue
    raise NoSuchElementException(f"Cannot find labeled interaction element: {target.component_id}") from last_error


def next_screenshot_index(output_dir: Path, page_name: str) -> int:
    prefix = normalize_text(page_name).replace(".", "_") + "_"
    max_index = 0
    if output_dir.exists():
        for path in output_dir.glob(f"{prefix}*.png"):
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.png", path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def screenshot_hash_bytes(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def node_id_from_image_name(image_name: str) -> str:
    return Path(image_name).stem


def capture_or_reuse_screenshot(
    driver: webdriver.Remote,
    output_dir: Path,
    page_name: str,
    known_hashes: dict[str, dict[str, str]],
) -> tuple[str, Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = driver.get_screenshot_as_png()
    image_hash = screenshot_hash_bytes(image_bytes)
    existing = known_hashes.get(image_hash)
    if existing:
        screenshot_path = output_dir / existing["image_name"]
        print(f"[screenshot] reuse {screenshot_path}")
        return existing["node_id"], screenshot_path, False

    image_name = screenshot_name_from_page(page_name, next_screenshot_index(output_dir, page_name))
    screenshot_path = output_dir / image_name
    screenshot_path.write_bytes(image_bytes)
    known_hashes[image_hash] = {
        "node_id": node_id_from_image_name(image_name),
        "image_name": image_name,
    }
    print(f"[screenshot] {screenshot_path}")
    return node_id_from_image_name(image_name), screenshot_path, True


def condition_text_for_target(target: InteractionTarget) -> str:
    return f"Click({target.text or target.component_id})"


def state_signature_from_targets(page_name: str, targets: list[InteractionTarget]) -> str:
    target_ids = sorted(target.component_id for target in targets)
    return f"{page_name}::{'|'.join(target_ids)}"


def get_runtime_state_signature(driver: webdriver.Remote) -> tuple[str, str]:
    page_name, _ = get_current_page_info(driver)
    targets = [target for target in find_labeled_elements(driver) if target.page_name == page_name]
    return page_name, state_signature_from_targets(page_name, targets)


def runtime_state_matches(driver: webdriver.Remote, expected_page: str, expected_state_signature: str) -> bool:
    try:
        page_name, state_signature = get_runtime_state_signature(driver)
    except Exception:
        return False
    return page_name == expected_page and state_signature == expected_state_signature


def update_utg_node(nodes_by_id: dict[str, dict], node_id: str, page_name: str) -> None:
    if node_id in nodes_by_id:
        return
    nodes_by_id[node_id] = {
        "id": node_id,
        "page": page_name,
    }


def replay_path(driver: webdriver.Remote, path_targets: list[InteractionTarget]) -> None:
    restart_app(driver)
    for target in path_targets:
        wait_for_page_render(driver, reason=f"replay before {target.component_id}")
        element = find_element_for_target(driver, target)
        element.click()
        interruptible_sleep(SETTLE_SECONDS)


def adb_back() -> None:
    subprocess.run(
        ["adb", "-s", DEVICE_NAME, "shell", "input", "keyevent", "4"],
        check=False,
    )


def add_recovery_transition(
    recovery_graph: dict[str, list[tuple[str, InteractionTarget]]],
    from_state_signature: str,
    to_state_signature: str,
    target: InteractionTarget,
) -> None:
    existing = recovery_graph.setdefault(from_state_signature, [])
    if any(next_state == to_state_signature and item.component_id == target.component_id for next_state, item in existing):
        return
    existing.append((to_state_signature, target))


def find_shortest_recovery_path(
    recovery_graph: dict[str, list[tuple[str, InteractionTarget]]],
    start_state_signature: str,
    target_state_signature: str,
) -> list[InteractionTarget] | None:
    if start_state_signature == target_state_signature:
        return []

    queue = deque([(start_state_signature, [])])
    visited = {start_state_signature}
    while queue:
        state_signature, path = queue.popleft()
        for next_state_signature, target in recovery_graph.get(state_signature, []):
            if next_state_signature in visited:
                continue
            next_path = path + [target]
            if next_state_signature == target_state_signature:
                return next_path
            visited.add(next_state_signature)
            queue.append((next_state_signature, next_path))
    return None


def recover_to_state(
    driver: webdriver.Remote,
    path_targets: list[InteractionTarget],
    recovery_graph: dict[str, list[tuple[str, InteractionTarget]]],
    start_state_signature: str,
    expected_page: str,
    expected_state_signature: str,
) -> None:
    if runtime_state_matches(driver, expected_page, expected_state_signature):
        return

    try:
        driver.back()
        interruptible_sleep(BACK_RECOVERY_SETTLE_SECONDS)
        if runtime_state_matches(driver, expected_page, expected_state_signature):
            print(f"[recover] driver.back to {expected_page}")
            return
    except Exception as exc:
        print(f"[recover] driver.back failed: {exc}")

    adb_back()
    interruptible_sleep(BACK_RECOVERY_SETTLE_SECONDS)
    if runtime_state_matches(driver, expected_page, expected_state_signature):
        print(f"[recover] adb back to {expected_page}")
        return

    shortest_path = find_shortest_recovery_path(recovery_graph, start_state_signature, expected_state_signature)
    if shortest_path is None:
        shortest_path = path_targets
        print(f"[recover] replay current path to {expected_page}: {len(shortest_path)} clicks")
    else:
        print(f"[recover] replay shortest path to {expected_page}: {len(shortest_path)} clicks")
    replay_path(driver, shortest_path)


def explore_app(driver: webdriver.Remote, output_dir: Path) -> dict:
    # State uses the visible interaction layout, so small screenshot byte changes do not re-click the same target.
    visited_interactions: set[str] = set()
    state_pending_interactions: dict[str, set[str]] = {}
    completed_states: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    nodes_by_id: dict[str, dict] = {}
    screenshot_hashes: dict[str, dict[str, str]] = {}
    recovery_graph: dict[str, list[tuple[str, InteractionTarget]]] = {}
    edges: list[dict] = []
    results: list[dict] = []

    restart_app(driver)
    start_page, _ = get_current_page_info(driver)
    start_targets = [target for target in find_labeled_elements(driver) if target.page_name == start_page]
    start_state_signature = state_signature_from_targets(start_page, start_targets)
    start_node_id, _, _ = capture_or_reuse_screenshot(driver, output_dir, start_page, screenshot_hashes)
    update_utg_node(nodes_by_id, start_node_id, start_page)

    def dfs(current_page: str, current_node_id: str, current_path: list[InteractionTarget]) -> None:
        wait_for_page_render(driver, reason=f"state {current_page}")
        page_name, _ = get_current_page_info(driver)
        current_node_id_runtime, _, _ = capture_or_reuse_screenshot(driver, output_dir, page_name, screenshot_hashes)
        update_utg_node(nodes_by_id, current_node_id_runtime, page_name)
        effective_from_node_id = current_node_id_runtime or current_node_id

        targets = [target for target in find_labeled_elements(driver) if target.page_name == page_name]
        if not targets:
            completed_states.add(state_signature_from_targets(page_name, targets))
            return

        state_signature = state_signature_from_targets(page_name, targets)
        if state_signature in completed_states:
            print(f"[dfs] skip completed state: {page_name}")
            return

        pending_ids = state_pending_interactions.setdefault(
            state_signature,
            {target.component_id for target in targets},
        )
        pending_targets = [
            target
            for target in targets
            if target.component_id in pending_ids
            and f"{state_signature}::{target.component_id}" not in visited_interactions
        ]
        if not pending_targets:
            completed_states.add(state_signature)
            print(f"[dfs] completed state: {page_name}")
            return

        for target in pending_targets:
            interaction_visit_key = f"{state_signature}::{target.component_id}"
            if interaction_visit_key in visited_interactions:
                continue

            visited_interactions.add(interaction_visit_key)
            pending_ids.discard(target.component_id)
            print(f"[dfs] {page_name} -> click {target.component_id} ({target.text})")
            try:
                wait_for_page_render(driver, reason=f"before click {target.component_id}")
                element = find_element_for_target(driver, target)
                element.click()
                interruptible_sleep(SETTLE_SECONDS)

                next_page, _ = get_current_page_info(driver)
                next_targets = [item for item in find_labeled_elements(driver) if item.page_name == next_page]
                next_state_signature = state_signature_from_targets(next_page, next_targets)
                add_recovery_transition(recovery_graph, state_signature, next_state_signature, target)
                next_node_id, screenshot_path, _ = capture_or_reuse_screenshot(driver, output_dir, next_page, screenshot_hashes)
                update_utg_node(nodes_by_id, next_node_id, next_page)

                condition_text = condition_text_for_target(target)
                edge_key = (effective_from_node_id, next_node_id, condition_text)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_id = f"{effective_from_node_id}-->{next_node_id}"
                    edges.append(
                        {
                            "id": edge_id,
                            "from": effective_from_node_id,
                            "to": next_node_id,
                            "conditions": [condition_text],
                        }
                    )
                results.append(
                    {
                        "component_id": target.component_id,
                        "page": page_name,
                        "to_page": next_page,
                        "to_node_id": next_node_id,
                        "text": target.text,
                        "status": "success",
                        "screenshot": str(screenshot_path),
                    }
                )

                next_path = current_path + [target]
                if next_state_signature in completed_states:
                    print(f"[dfs] next state already completed: {next_page}")
                else:
                    dfs(next_page, next_node_id, next_path)
                if pending_ids:
                    recover_to_state(driver, current_path, recovery_graph, start_state_signature, page_name, state_signature)
                else:
                    print(f"[recover] skip completed state: {page_name}")
            except Exception as exc:
                results.append(
                    {
                        "component_id": target.component_id,
                        "page": page_name,
                        "text": target.text,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"[dfs] failed: {target.component_id}: {exc}")
                if pending_ids:
                    recover_to_state(driver, current_path, recovery_graph, start_state_signature, page_name, state_signature)
                else:
                    print(f"[recover] skip completed state: {page_name}")

        if not pending_ids:
            completed_states.add(state_signature)
            print(f"[dfs] completed state: {page_name}")

    dfs(start_page, start_node_id, [])
    return {
        "N": list(nodes_by_id.values()),
        "E": edges,
        "results": results,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] {path}")


def run_single_app(category: str, app_name: str) -> None:
    generated_code_dir = ensure_generated_code_dir(category, app_name)
    output_dir = OUTPUT_ROOT / category / app_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.json"
    original_res_path = generated_code_dir / "res.txt"
    labeled_res_path = get_output_res_path(original_res_path)

    if not original_res_path.exists():
        raise FileNotFoundError(f"res.txt not found: {original_res_path}")

    compile_failures: list[dict] = []
    try:
        print(f"[label] generating {LABEL_OUTPUT_FILENAME} from {original_res_path}")
        process_res_file(original_res_path, write=WRITE_LABEL_FILE)
        if not labeled_res_path.exists():
            raise FileNotFoundError(f"Labeled res file not found after labeling: {labeled_res_path}")

        print(f"[flow] labeled code -> project: {labeled_res_path}")
        materialize_labeled_res_to_project(labeled_res_path, PROJECT_PATH, CLEAR_COMPONENTS)
        if SKIP_COMPILE:
            print("[compile] skipped")
        else:
            compile_project(PROJECT_PATH, category, app_name, labeled_res_path)
    except CompileFailure as exc:
        compile_failures.append({"stage": exc.stage, "detail": exc.detail})
        write_json(
            report_path,
            {
                "category": category,
                "app_name": app_name,
                "generated_code_dir": str(generated_code_dir),
                "original_res": str(original_res_path),
                "labeled_res": str(labeled_res_path),
                "compile_status": "failed",
                "compile_failures": compile_failures,
                "interaction_failures": [],
                "results": [],
            },
        )
        raise
    except Exception as exc:
        compile_failures.append({"stage": "prepare", "detail": str(exc)})
        write_json(
            report_path,
            {
                "category": category,
                "app_name": app_name,
                "generated_code_dir": str(generated_code_dir),
                "original_res": str(original_res_path),
                "labeled_res": str(labeled_res_path),
                "compile_status": "failed",
                "compile_failures": compile_failures,
                "interaction_failures": [],
                "results": [],
            },
        )
        raise

    driver = build_driver()
    try:
        utg_data = explore_app(driver, output_dir)
    finally:
        driver.quit()

    utg_path = output_dir / "UTG.json"
    write_json(utg_path, {"N": utg_data["N"], "E": utg_data["E"]})
    interaction_failures = [item for item in utg_data["results"] if item.get("status") == "failed"]
    write_json(
        report_path,
        {
            "category": category,
            "app_name": app_name,
            "generated_code_dir": str(generated_code_dir),
            "original_res": str(original_res_path),
            "labeled_res": str(labeled_res_path),
            "utg_path": str(utg_path),
            "compile_status": "skipped" if SKIP_COMPILE else "success",
            "compile_failures": compile_failures,
            "result_count": len(utg_data["results"]),
            "interaction_failure_count": len(interaction_failures),
            "interaction_failures": interaction_failures,
            "results": utg_data["results"],
        },
    )


def main() -> None:
    targets = discover_app_targets() if BATCH_MODE else [(normalize_text(CATEGORY), normalize_text(APP_NAME))]
    targets = [(category, app_name) for category, app_name in targets if category and app_name]
    if not targets:
        raise ValueError("No app targets found. Set CATEGORY/APP_NAME or check GENERATED_CODE_ROOT.")

    successful_apps: list[dict] = []
    compile_failed_apps: list[dict] = []
    interaction_failed_apps: list[dict] = []
    unexpected_failed_apps: list[dict] = []

    for index, (category, app_name) in enumerate(targets, start=1):
        print(f"[batch] ({index}/{len(targets)}) {category}/{app_name}")
        try:
            run_single_app(category, app_name)
            successful_apps.append({"category": category, "app_name": app_name})
        except KeyboardInterrupt:
            print(f"[batch] interrupted by user while processing {category}/{app_name}")
            raise
        except CompileFailure as exc:
            compile_failed_apps.append(
                {
                    "category": category,
                    "app_name": app_name,
                    "stage": exc.stage,
                    "detail": exc.detail,
                }
            )
            print(f"[batch] compile failed: {category}/{app_name} @ {exc.stage}: {exc.detail}")
        except Exception as exc:
            unexpected_failed_apps.append(
                {
                    "category": category,
                    "app_name": app_name,
                    "detail": str(exc),
                }
            )
            print(f"[batch] failed: {category}/{app_name}: {exc}")

        report_path = OUTPUT_ROOT / category / app_name / "run_report.json"
        if report_path.exists():
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            interaction_failures = report_data.get("interaction_failures", [])
            if interaction_failures:
                interaction_failed_apps.append(
                    {
                        "category": category,
                        "app_name": app_name,
                        "count": len(interaction_failures),
                        "items": interaction_failures,
                    }
                )

    batch_report = {
        "baseline_name": BASELINE_NAME,
        "target_count": len(targets),
        "success_count": len(successful_apps),
        "compile_failed_count": len(compile_failed_apps),
        "interaction_failed_count": len(interaction_failed_apps),
        "unexpected_failed_count": len(unexpected_failed_apps),
        "successful_apps": successful_apps,
        "compile_failed_apps": compile_failed_apps,
        "interaction_failed_apps": interaction_failed_apps,
        "unexpected_failed_apps": unexpected_failed_apps,
    }
    write_json(OUTPUT_ROOT / "_batch_report.json", batch_report)

    print("[summary] successful apps:")
    if successful_apps:
        for item in successful_apps:
            print(f"  - {item['category']}/{item['app_name']}")
    else:
        print("  - none")

    print("[summary] compile failed apps:")
    if compile_failed_apps:
        for item in compile_failed_apps:
            print(f"  - {item['category']}/{item['app_name']} @ {item['stage']}: {item['detail']}")
    else:
        print("  - none")

    print("[summary] interaction failed apps:")
    if interaction_failed_apps:
        for item in interaction_failed_apps:
            print(f"  - {item['category']}/{item['app_name']}: {item['count']} failed interactions")
            for failure in item["items"]:
                component_id = failure.get("component_id", "")
                page = failure.get("page", "")
                error = failure.get("error", "")
                print(f"    * {page} / {component_id}: {error}")
    else:
        print("  - none")

    print("[summary] unexpected failed apps:")
    if unexpected_failed_apps:
        for item in unexpected_failed_apps:
            print(f"  - {item['category']}/{item['app_name']}: {item['detail']}")
    else:
        print("  - none")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[appium] interrupted by user")
