from __future__ import annotations

import json
import os
import random
import re
import multiprocessing
import shutil
import subprocess
import sys
import time
import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
APPIUM_IMPORT_ERROR: Exception | None = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code_generate.utils.compiler.RN_compiler import run_gradlew_assemble

from utg_generate.label_component import (
    OUTPUT_FILENAME as LABEL_OUTPUT_FILENAME,
    get_output_res_path,
    process_res_file,
    split_res_sections,
)

BASELINE_NAMES = ["PageWeaver"]

BASELINE_NAME = ""

GENERATED_CODE_ROOT = None
OUTPUT_ROOT = None

DATA_FIND_ROOT = REPO_ROOT / "dataset"

# Edit these values directly before running this file.
PROJECT_PATH = Path(r"")
BATCH_MODE = True
CATEGORY = None
APP_NAME = None
APPIUM_SERVER_URL = "http://127.0.0.1:4723"
APPIUM_HOST = "127.0.0.1"
APPIUM_PORT = 4723
APPIUM_AUTO_RESTART_ON_CONNECTION_ERROR = True
APPIUM_START_COMMAND = ["appium", "--address", APPIUM_HOST, "--port", str(APPIUM_PORT)]
APPIUM_START_WAIT_SECONDS = 8.0
DEVICE_NAME = "emulator-5554"
APP_PACKAGE = "com.awesomeproject"
APP_ACTIVITY = "com.awesomeproject.MainActivity"
DEVICE_SERIAL: str | None = None
EMULATOR_AVD_NAME: str | None = "Pixel_6"
EMULATOR_AUTO_START = True
EMULATOR_NO_SNAPSHOT_LOAD = True
EMULATOR_COLD_START_ON_APPIUM_CONNECTION_ERROR = True
NO_RESET = True
CLEAR_COMPONENTS = True
WRITE_LABEL_FILE = True
SKIP_COMPILE = True
SKIP_EXISTING_OUTPUT = True
DEBUG_LABELS = False
RERUN_ONLY_MISMATCHED_APPS = True
CLEAR_OUTPUT_DIR_BEFORE_RERUN = True
STARTUP_RENDER_WAIT_SECONDS = 5.0
SETTLE_SECONDS = 2.0
PRE_INTERACTION_RENDER_WAIT_SECONDS = 2.0
RENDER_WAIT_POLL_SECONDS = 0.2
RENDER_WAIT_STABLE_POLLS = 2
UI_TREE_WAIT_SECONDS = 3.0
UI_TREE_WAIT_POLL_SECONDS = 0.2
UI_TREE_WAIT_STABLE_POLLS = 2
ELEMENT_TIMEOUT_SECONDS = 2.0
ELEMENT_LOCATE_POLL_SECONDS = 0.15
CURRENT_PAGE_WAIT_SECONDS = 2.0
APPIUM_RUN_RETRY_COUNT = 2
APPIUM_RUN_RETRY_DELAY_SECONDS = 5.0
APPIUM_RUN_RETRY_BACKOFF = 1.5
ADB_DEVICE_WAIT_SECONDS = 45.0
ADB_DEVICE_WAIT_POLL_SECONDS = 3.0
ADB_RESTART_ON_DEVICE_MISSING = True
LABEL_GENERATION_TIMEOUT_SECONDS = 60.0
EMULATOR_BOOT_WAIT_SECONDS = 120.0
EMULATOR_COLD_START_SETTLE_SECONDS = 5.0
RERUN_ONLY_ZERO_SCREENSHOT_APPS = True

INTERACTIVE_ID_RE = re.compile(
    r"^(?P<page>[A-Za-z0-9_]+)(?:\.(?P<component>[A-Za-z_][A-Za-z0-9_]*))?\.(?P<type>[A-Za-z_][A-Za-z0-9_]*)\.(?P<index>.+)$"
)
INVALID_LABEL_PAGE_NAMES = {"UnknownPage"}
APP_LABEL_PAGE_NAME = "App"
NON_INTERACTION_ROOT_COMPONENT_TYPES = {
    "ActivityIndicator",
    "FlatList",
    "Image",
    "ImageBackground",
    "KeyboardAvoidingView",
    "Modal",
    "RefreshControl",
    "SafeAreaView",
    "ScrollView",
    "SectionList",
    "StatusBar",
    "Text",
    "View",
    "VirtualizedList",
}

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

def process_res_file_worker(res_path: Path, write: bool, result_queue: multiprocessing.Queue) -> None:
    try:
        process_res_file(res_path, write=write)
    except BaseException as exc:
        result_queue.put((False, repr(exc)))
    else:
        result_queue.put((True, ""))

def process_res_file_with_timeout(res_path: Path, write: bool, timeout_seconds: float) -> None:
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=process_res_file_worker,
        args=(res_path, write, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join()
        raise CompileFailure(
            "label_timeout",
            f"code labeling exceeded {int(timeout_seconds)} seconds: {res_path}",
        )
    if process.exitcode not in (0, None):
        raise CompileFailure("label", f"code labeling process exited with code {process.exitcode}: {res_path}")
    try:
        success, detail = result_queue.get_nowait()
    except Exception:
        return
    if not success:
        raise CompileFailure("label", detail)

def print_debug_labels(
    title: str,
    page_name: str,
    node_id: str | None,
    state_signature: str | None,
    targets: list[InteractionTarget],
) -> None:
    if not DEBUG_LABELS:
        return

    print("\n" + "-" * 80)
    print(f"[LABEL DEBUG] {title}")
    print(f"[PAGE] {page_name}")
    print(f"[NODE] {node_id}")
    print(f"[STATE SIGNATURE] {state_signature}")
    print(f"[TARGET COUNT] {len(targets)}")
    print("-" * 80)

    for i, target in enumerate(targets, start=1):
        print(
            f"{i:02d}. "
            f"id={target.component_id} | "
            f"page={target.page_name} | "
            f"component={target.component_name} | "
            f"type={target.component_type} | "
            f"index={target.index} | "
            f"text={target.text} | "
            f"locator={target.locator_strategy}:{target.locator_value}"
        )

    print("-" * 80 + "\n")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def read_json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def count_top_level_images(directory: Path) -> int | None:
    if not directory.exists() or not directory.is_dir():
        return None
    return sum(
        1
        for child in directory.iterdir()
        if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

def build_mismatch_records() -> list[dict]:
    records: list[dict] = []
    if not DATA_FIND_ROOT.exists():
        raise FileNotFoundError(f"dataset root not found: {DATA_FIND_ROOT}")

    for category_dir in sorted(path for path in DATA_FIND_ROOT.iterdir() if path.is_dir()):
        baseline_category_dir = OUTPUT_ROOT / category_dir.name
        for app_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            baseline_app_dir = baseline_category_dir / app_dir.name
            data_find_count = count_top_level_images(app_dir)
            baseline_count = count_top_level_images(baseline_app_dir)
            match = baseline_count is not None and data_find_count is not None and baseline_count >= data_find_count
            records.append(
                {
                    "Category": category_dir.name,
                    "App": app_dir.name,
                    "DataFindCount": data_find_count,
                    "BaselineCount": baseline_count,
                    "Difference": None if baseline_count is None or data_find_count is None else baseline_count - data_find_count,
                    "Match": match,
                    "DataFindPath": str(app_dir),
                    "BaselinePath": str(baseline_app_dir),
                }
            )
    return records

def mismatch_records_to_csv(records: list[dict]) -> str:
    headers = ["Category", "App", "DataFindCount", "BaselineCount", "Difference", "Match", "DataFindPath", "BaselinePath"]
    lines = [",".join(headers)]
    for record in records:
        fields: list[str] = []
        for header in headers:
            value = record.get(header, "")
            text = "" if value is None else str(value)
            if any(char in text for char in [",", "\"", "\n", "\r"]):
                text = '"' + text.replace('"', '""') + '"'
            fields.append(text)
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"

def write_mismatch_reports() -> list[dict]:
    records = build_mismatch_records()
    write_text(MISMATCH_OUTPUT_JSON, json.dumps(records, ensure_ascii=False, indent=4))
    write_text(MISMATCH_OUTPUT_CSV, mismatch_records_to_csv(records))
    mismatched_count = sum(1 for record in records if not record["Match"])
    print(f"[mismatch] updated: total={len(records)}, mismatched={mismatched_count}")
    return records

def load_mismatch_targets() -> list[tuple[str, str]]:
    if not MISMATCH_OUTPUT_JSON.exists():
        records = write_mismatch_reports()
    else:
        try:
            records = read_json_file(MISMATCH_OUTPUT_JSON)
        except json.JSONDecodeError as exc:
            print(f"[mismatch] invalid json, rebuilding {MISMATCH_OUTPUT_JSON}: {exc}")
            records = write_mismatch_reports()
    return [
        (record["Category"], record["App"])
        for record in records
        if not record.get("Match")
    ]

def load_zero_screenshot_targets() -> list[tuple[str, str]]:
    if not MISMATCH_OUTPUT_JSON.exists():
        records = write_mismatch_reports()
    else:
        try:
            records = read_json_file(MISMATCH_OUTPUT_JSON)
        except json.JSONDecodeError as exc:
            print(f"[mismatch] invalid json, rebuilding {MISMATCH_OUTPUT_JSON}: {exc}")
            records = write_mismatch_reports()

    targets: list[tuple[str, str]] = []
    for record in records:
        baseline_count = record.get("BaselineCount")

        # BaselineCount 为 0 表示截图数量为 0
        # 如果目录不存在，BaselineCount 可能是 None，这里也可以按 0 处理
        if baseline_count in (0, None):
            targets.append((record["Category"], record["App"]))

    return targets

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

def find_process_ids_listening_on_port(port: int) -> set[int]:
    pids: set[int] = set()
    output = run_capture(["netstat", "-ano"], timeout=20)
    listen_marker = f":{port}"
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        state = parts[3].upper() if len(parts) >= 4 else ""
        pid_text = parts[-1]
        if listen_marker not in local_address or state != "LISTENING":
            continue
        try:
            pids.add(int(pid_text))
        except ValueError:
            continue
    return pids

def stop_appium_server() -> None:
    pids = find_process_ids_listening_on_port(APPIUM_PORT)
    if not pids:
        print(f"[appium] no process listening on port {APPIUM_PORT}")
        return
    for pid in sorted(pids):
        print(f"[appium] stopping pid {pid} on port {APPIUM_PORT}")
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def start_appium_server() -> None:
    print(f"[appium] starting server: {' '.join(APPIUM_START_COMMAND)}")
    subprocess.Popen(
        APPIUM_START_COMMAND,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    interruptible_sleep(APPIUM_START_WAIT_SECONDS)

def restart_appium_server() -> None:
    print("[appium] restarting server")
    stop_appium_server()
    interruptible_sleep(1.0)
    start_appium_server()

def android_sdk_emulator_path() -> Path | None:
    for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk_root = normalize_text(os.environ.get(env_name))
        if not sdk_root:
            continue
        emulator_name = "emulator.exe" if os.name == "nt" else "emulator"
        emulator_path = Path(sdk_root) / "emulator" / emulator_name
        if emulator_path.exists():
            return emulator_path
    return None

def emulator_command_base() -> list[str]:
    emulator_path = android_sdk_emulator_path()
    return [str(emulator_path)] if emulator_path else ["emulator"]

def list_available_avds() -> list[str]:
    output = run_capture(emulator_command_base() + ["-list-avds"], timeout=30)
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip() and not line.lower().startswith("info")
    ]

def resolve_emulator_avd_name() -> str | None:
    configured = normalize_text(EMULATOR_AVD_NAME)
    if configured:
        return configured

    avds = list_available_avds()
    if not avds:
        return None
    if len(avds) > 1:
        print(f"[emulator] EMULATOR_AVD_NAME not set; using first AVD: {avds[0]} (available: {avds})")
    return avds[0]

def start_emulator_no_snapshot() -> bool:
    avd_name = resolve_emulator_avd_name()
    if not avd_name:
        print("[emulator] no AVD available to auto-start")
        return False

    command = emulator_command_base() + ["-avd", avd_name]
    if EMULATOR_NO_SNAPSHOT_LOAD:
        command.append("-no-snapshot-load")
    print(f"[emulator] starting: {' '.join(command)}")
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return True

def stop_target_emulator() -> None:
    command = adb_base_command() + ["emu", "kill"]
    print(f"[emulator] stopping: {' '.join(command)}")
    try:
        output = run_capture(command, timeout=20)
        if output.strip():
            print(f"[emulator] stop output: {output.strip()}")
    except Exception as exc:
        print(f"[emulator] stop failed: {exc}")

def cold_start_emulator() -> None:
    if not EMULATOR_AUTO_START:
        print("[emulator] cold start skipped: EMULATOR_AUTO_START=False")
        return

    print("[emulator] cold start requested")
    stop_target_emulator()
    interruptible_sleep(EMULATOR_COLD_START_SETTLE_SECONDS)
    restart_adb_server()
    if not start_emulator_no_snapshot():
        print("[emulator] cold start failed: no emulator started")
        return
    ensure_device_available(wait_seconds=EMULATOR_BOOT_WAIT_SECONDS, restart_server=True)

def parse_adb_devices(output: str) -> dict[str, str]:
    devices: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("list of devices"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) >= 2:
            devices[parts[0]] = parts[1]
    return devices

def get_adb_devices() -> dict[str, str]:
    output = run_capture(["adb", "devices"], timeout=15)
    return parse_adb_devices(output)

def run_adb_shell_capture(args: list[str], timeout: int = 15) -> str:
    return run_capture(adb_base_command() + ["shell", *args], timeout=timeout)

def android_runtime_ready_status() -> tuple[bool, str]:
    try:
        boot_completed = normalize_text(run_adb_shell_capture(["getprop", "sys.boot_completed"], timeout=10))
    except Exception as exc:
        return False, f"getprop sys.boot_completed failed: {exc}"

    try:
        settings_service = normalize_text(run_adb_shell_capture(["service", "check", "settings"], timeout=10))
    except Exception as exc:
        return False, f"service check settings failed: {exc}"

    boot_ready = boot_completed.splitlines()[0].strip() == "1" if boot_completed else False
    settings_ready = "found" in settings_service.lower() and "not found" not in settings_service.lower()
    if boot_ready and settings_ready:
        return True, "android runtime ready"
    return False, f"boot_completed={boot_completed!r}; settings_service={settings_service!r}"

def restart_adb_server() -> None:
    print("[adb] restarting adb server")
    for command in (["adb", "kill-server"], ["adb", "start-server"]):
        output = run_capture(command, timeout=30)
        if output.strip():
            print(f"[adb] {' '.join(command)}: {output.strip()}")

def ensure_device_available(
    wait_seconds: float = ADB_DEVICE_WAIT_SECONDS,
    restart_server: bool = ADB_RESTART_ON_DEVICE_MISSING,
) -> None:
    target_serial = resolve_device_serial()
    effective_wait_seconds = max(
        wait_seconds,
        EMULATOR_BOOT_WAIT_SECONDS if EMULATOR_AUTO_START else wait_seconds,
    )
    deadline = time.monotonic() + max(0.0, effective_wait_seconds)
    restarted = False
    emulator_started = False
    last_status = ""

    while True:
        try:
            devices = get_adb_devices()
        except Exception as exc:
            devices = {}
            last_status = f"adb devices failed: {exc}"
        else:
            if target_serial:
                state = devices.get(target_serial)
                if state == "device":
                    runtime_ready, runtime_status = android_runtime_ready_status()
                    if runtime_ready:
                        print(f"[adb] device ready: {target_serial}")
                        return
                    last_status = f"{target_serial}=device; {runtime_status}"
                else:
                    last_status = f"{target_serial}={state or 'missing'}; devices={devices}"
            else:
                ready_devices = [serial for serial, state in devices.items() if state == "device"]
                if ready_devices:
                    runtime_ready, runtime_status = android_runtime_ready_status()
                    if runtime_ready:
                        print(f"[adb] device ready: {ready_devices[0]}")
                        return
                    last_status = f"{ready_devices[0]}=device; {runtime_status}"
                else:
                    last_status = f"no ready device; devices={devices}"

        if restart_server and not restarted:
            restart_adb_server()
            restarted = True

        if EMULATOR_AUTO_START and not emulator_started:
            missing_target = False
            if target_serial:
                missing_target = target_serial not in devices or devices.get(target_serial) in {"offline", "unknown"}
            else:
                missing_target = not any(state == "device" for state in devices.values())
            if missing_target:
                emulator_started = start_emulator_no_snapshot()

        if time.monotonic() >= deadline:
            raise RuntimeError(f"Android device/emulator unavailable: {last_status}")

        print(f"[adb] waiting for device: {last_status}")
        interruptible_sleep(ADB_DEVICE_WAIT_POLL_SECONDS)

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
        if SKIP_EXISTING_OUTPUT and not RERUN_ONLY_MISMATCHED_APPS and has_existing_app_output(requested_category, requested_app_name):
            print(f"[skip] existing output: {requested_category}/{requested_app_name}")
            return []
        return [(requested_category, requested_app_name)]
 # 放这里：优先筛 BaselineCount == 0 的 app
    if RERUN_ONLY_ZERO_SCREENSHOT_APPS:
        zero_targets = load_zero_screenshot_targets()

        if requested_category:
            zero_targets = [
                (category, app_name)
                for category, app_name in zero_targets
                if category == requested_category
            ]

        if requested_app_name:
            zero_targets = [
                (category, app_name)
                for category, app_name in zero_targets
                if app_name == requested_app_name
            ]

        print(f"[targets] zero-screenshot apps: {len(zero_targets)}")
        return zero_targets
    if RERUN_ONLY_MISMATCHED_APPS:
        mismatch_targets = load_mismatch_targets()
        if requested_category:
            mismatch_targets = [(category, app_name) for category, app_name in mismatch_targets if category == requested_category]
        if requested_app_name:
            mismatch_targets = [(category, app_name) for category, app_name in mismatch_targets if app_name == requested_app_name]
        print(f"[targets] mismatch-only apps: {len(mismatch_targets)}")
        return mismatch_targets

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

def materialize_labeled_res_to_project(res_label_path: Path, project_path: Path, clear_components: bool) -> None:
    if not res_label_path.exists():
        raise FileNotFoundError(f"Labeled res file not found: {res_label_path}")

    sections = split_res_sections(res_label_path.read_text(encoding="utf-8"))
    components_dir = project_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    if clear_components:
        for child in components_dir.iterdir():
            if child.is_file() and child.suffix.lower() in {".js", ".jsx"}:
                child.unlink()

    for section in sections:
        file_name = Path(section.name).name
        if Path(file_name).suffix.lower() != ".js":
            print(f"[label-copy] skip non-js section: {section.name}")
            continue
        destination_path = project_path / file_name if file_name == "App.js" else components_dir / file_name
        destination_path.write_text(section.code.rstrip() + "\n", encoding="utf-8")
        print(f"[label-copy] {section.name} -> {destination_path}")

def extract_packages_from_res_file(res_path: Path) -> list[str]:
    if not res_path.exists():
        return []

    packages: set[str] = set()
    code = res_path.read_text(encoding="utf-8")
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
    ensure_device_available()
    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.automation_name = "UiAutomator2"
    options.set_capability("enableMultiWindows", True)
    options.set_capability("ignoreUnimportantViews", False)
    options.device_name = DEVICE_NAME
    device_serial = resolve_device_serial()
    if device_serial:
        options.udid = device_serial
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

    # print(f"[appium] waiting up to {wait_seconds}s for {reason}")
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
            # print(f"[appium] render settled for {reason}")
            return

        interruptible_sleep(min(RENDER_WAIT_POLL_SECONDS, max(0.0, deadline - time.monotonic())))

    # print(f"[appium] render wait timed out after {wait_seconds}s for {reason}")

def get_ui_tree_snapshot(driver: webdriver.Remote) -> tuple[str, int, int, int]:
    page_source = getattr(driver, "page_source", "")
    if not isinstance(page_source, str):
        page_source = str(page_source)
    source_hash = hashlib.md5(page_source.encode("utf-8", errors="ignore")).hexdigest()
    return (
        source_hash,
        len(page_source),
        page_source.count("resource-id="),
        page_source.count("content-desc="),
    )

def wait_for_ui_tree_stable(
    driver: webdriver.Remote,
    reason: str = "before reading UI tree",
    max_wait_seconds: float | None = None,
) -> None:
    wait_seconds = UI_TREE_WAIT_SECONDS if max_wait_seconds is None else max_wait_seconds
    if wait_seconds <= 0:
        return

    deadline = time.monotonic() + wait_seconds
    last_snapshot: tuple[str, int, int, int] | None = None
    stable_polls = 0

    while time.monotonic() < deadline:
        try:
            snapshot = get_ui_tree_snapshot(driver)
        except Exception as exc:
            print(f"[appium] UI tree probe failed during {reason}: {exc}; falling back to sleep")
            interruptible_sleep(wait_seconds)
            return

        if snapshot == last_snapshot and snapshot[1] > 0:
            stable_polls += 1
        else:
            last_snapshot = snapshot
            stable_polls = 0

        if stable_polls >= UI_TREE_WAIT_STABLE_POLLS:
            return

        interruptible_sleep(min(UI_TREE_WAIT_POLL_SECONDS, max(0.0, deadline - time.monotonic())))

    print(f"[appium] UI tree wait timed out after {wait_seconds}s for {reason}")

def screenshot_name_from_page(page_name: str, index: int) -> str:
    safe_name = normalize_text(page_name).replace(".", "_")
    return f"{safe_name}_{index}.png"

def get_label_components(
        label_value: str
    ) -> tuple[
        str | None,
        str | None,
        str | None,
        str | None
    ]:
        match = INTERACTIVE_ID_RE.match(label_value)

        if not match:
            return None, None, None, None

        component_name = match.group("component")
        component_type = match.group("type")

        index = match.group("index")
        return (
            match.group("page"),
            component_name,
            component_type,
            index,
        )

def is_tab_screen_label_value(label_value: str | None) -> bool:
    page_name, component_name, component_type, _ = get_label_components(normalize_text(label_value))
    return bool(page_name and normalize_text(component_name).lower() == "tab" and component_type == "Screen")

def is_tab_screen_target(target: InteractionTarget) -> bool:
    return normalize_text(target.component_name).lower() == "tab" and target.component_type == "Screen"

def is_app_scoped_label_value(label_value: str | None) -> bool:
    page_name, _, _, _ = get_label_components(normalize_text(label_value))
    return page_name == APP_LABEL_PAGE_NAME

def is_app_scoped_target(target: InteractionTarget) -> bool:
    return target.page_name == APP_LABEL_PAGE_NAME

def is_non_interaction_root_target(target: InteractionTarget) -> bool:
    return target.component_name is None and target.component_type in NON_INTERACTION_ROOT_COMPONENT_TYPES

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

def collect_ui_identifier_samples(driver: webdriver.Remote, limit: int = 20) -> list[str]:
    samples: list[str] = []
    seen: set[str] = set()
    try:
        elements = driver.find_elements(AppiumBy.XPATH, "//*[@resource-id or @content-desc]")
    except Exception:
        return samples

    for element in elements:
        for value in candidate_id_values(element):
            if not value or value in seen:
                continue
            seen.add(value)
            samples.append(value)
            if len(samples) >= limit:
                return samples
    return samples

def get_driver_identity(driver: webdriver.Remote) -> str:
    parts: list[str] = []
    for label, attr_name in (("package", "current_package"), ("activity", "current_activity")):
        try:
            value = normalize_text(getattr(driver, attr_name))
        except Exception:
            value = ""
        if value:
            parts.append(f"{label}={value}")
    return ", ".join(parts) if parts else "package/activity unavailable"

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

def find_labeled_elements(driver: webdriver.Remote, ensure_stable: bool = True) -> list[InteractionTarget]:
    if ensure_stable:
        wait_for_ui_tree_stable(driver, "before finding labeled elements")

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
    deadline = time.monotonic() + max(CURRENT_PAGE_WAIT_SECONDS, UI_TREE_WAIT_SECONDS)
    while True:
        wait_for_ui_tree_stable(driver, "before determining current page")

        labeled_elements = find_labeled_elements(driver, ensure_stable=False)
        page_labeled_elements = [
            item
            for item in labeled_elements
            if not is_tab_screen_target(item) and not is_app_scoped_target(item)
        ]
        if page_labeled_elements:
            return page_labeled_elements[0].page_name, page_labeled_elements[0].component_id

        all_candidates = []
        try:
            elements = driver.find_elements(AppiumBy.XPATH, "//*[@resource-id or @content-desc]")
        except Exception:
            elements = []

        for element in elements:
            for value in candidate_id_values(element):
                if is_tab_screen_label_value(value):
                    continue
                if is_app_scoped_label_value(value):
                    continue
                page_name, _, _, _ = get_label_components(value)
                if page_name and page_name not in INVALID_LABEL_PAGE_NAMES:
                    all_candidates.append((page_name, value))
        if all_candidates:
            page_name, component_id = all_candidates[0]
            return page_name, component_id

        if time.monotonic() >= deadline:
            break
        interruptible_sleep(UI_TREE_WAIT_POLL_SECONDS)

    resource_id_count = len(driver.find_elements(AppiumBy.XPATH, "//*[@resource-id]"))
    content_desc_count = len(driver.find_elements(AppiumBy.XPATH, "//*[@content-desc]"))
    samples = collect_ui_identifier_samples(driver)
    identity = get_driver_identity(driver)
    print("resource-id count=", resource_id_count)
    print("content-desc count=", content_desc_count)
    print("ui identity=", identity)
    print("ui id samples=", samples)

    hint = ""
    if SKIP_COMPILE:
        hint = (
            " SKIP_COMPILE=True, so the project files may have been updated without rebuilding/"
            "installing the APK; rebuild/install the labeled project or set SKIP_COMPILE=False."
        )
    raise RuntimeError(
        "Cannot determine current page from labeled components or root container. "
        f"{identity}; resource-id count={resource_id_count}; content-desc count={content_desc_count}; "
        f"sample identifiers={samples}.{hint}"
    )

def build_target_locators(driver, target: InteractionTarget) -> list[tuple[str, str]]:
    app_package = ""
    try:
        app_package = normalize_text(driver.current_package)
    except Exception:
        pass

    locators: list[tuple[str, str]] = []

    def add_locator(locator: tuple[str, str]) -> None:
        if locator not in locators:
            locators.append(locator)

    if target.locator_strategy == "accessibility":
        add_locator((AppiumBy.ACCESSIBILITY_ID, target.locator_value or target.component_id))
    elif target.locator_strategy == "resource-id":
        raw_id = target.locator_value or target.component_id
        add_locator((AppiumBy.ID, raw_id))
        if app_package and not raw_id.startswith(f"{app_package}:id/"):
            add_locator((AppiumBy.ID, f"{app_package}:id/{raw_id}"))

    add_locator((AppiumBy.ACCESSIBILITY_ID, target.component_id))
    add_locator((AppiumBy.ID, target.component_id))
    if app_package:
        add_locator((AppiumBy.ID, f"{app_package}:id/{target.component_id}"))
        add_locator((AppiumBy.XPATH, f'//*[@resource-id="{app_package}:id/{target.component_id}"]'))
    add_locator((AppiumBy.XPATH, f'//*[@resource-id="{target.component_id}"]'))
    add_locator((AppiumBy.ANDROID_UIAUTOMATOR, f'new UiSelector().description("{target.component_id}")'))
    return locators

def find_element_for_target(driver, target):
    last_error = None
    locators = build_target_locators(driver, target)
    deadline = time.monotonic() + max(0.0, ELEMENT_TIMEOUT_SECONDS)

    while time.monotonic() <= deadline:
        for locator in locators:
            try:
                return driver.find_element(*locator)
            except Exception as exc:
                last_error = exc

        interruptible_sleep(ELEMENT_LOCATE_POLL_SECONDS, step=ELEMENT_LOCATE_POLL_SECONDS)

    raise NoSuchElementException(
        f"Cannot find {target.component_id}"
    ) from last_error

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

def is_tab_interaction(target: InteractionTarget | None) -> bool:
    if target is None:
        return False
    return "tab" in normalize_text(target.component_name).lower()

def sort_interaction_targets_for_exploration(targets: list[InteractionTarget]) -> list[InteractionTarget]:
    shuffled_targets = list(targets)
    random.shuffle(shuffled_targets)
    return shuffled_targets

def describe_interaction_targets(targets: list[InteractionTarget]) -> list[dict]:
    return [
        {
            "component_id": target.component_id,
            "text": target.text,
            "component_name": target.component_name,
            "component_type": target.component_type,
            "index": target.index,
        }
        for target in targets
    ]

def capture_or_reuse_screenshot(
    driver: webdriver.Remote,
    output_dir: Path,
    page_name: str,
    state_signature: str,
    known_states: dict[str, dict],
    tab_interaction: bool = False,
) -> tuple[str, Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)

    state_record = known_states.get(state_signature)
    images = state_record.get("images", {}) if state_record else {}

    image_bytes = driver.get_screenshot_as_png()
    image_hash = screenshot_hash_bytes(image_bytes)

    if image_hash in images:
        known_state = images[image_hash]
        known_screenshot_path = output_dir / known_state["image_name"]
        print(f"[screenshot] reuse state+image {known_screenshot_path}")
        return known_state["node_id"], known_screenshot_path, False

    image_name = screenshot_name_from_page(
        page_name,
        next_screenshot_index(output_dir, page_name),
    )
    screenshot_path = output_dir / image_name
    screenshot_path.write_bytes(image_bytes)

    node_id = node_id_from_image_name(image_name)
    if state_record is None:
        state_record = {"images": {}}
        known_states[state_signature] = state_record
        images = state_record["images"]

    images[image_hash] = {
        "node_id": node_id,
        "image_name": image_name,
        "image_hash": image_hash,
    }

    print(f"[screenshot] new state+image {screenshot_path}")
    return node_id, screenshot_path, True

def condition_text_for_target(target: InteractionTarget) -> str:
    return f"Click({target.text or target.component_id})"

def state_signature_from_targets(page_name: str, targets: list[InteractionTarget]) -> str:
    target_ids = sorted(target.component_id for target in targets if not is_tab_screen_target(target))
    return f"{page_name}::{'|'.join(target_ids)}"

def get_page_interaction_targets(page_name: str, targets: list[InteractionTarget]) -> list[InteractionTarget]:
    return [
        target
        for target in targets
        if (target.page_name == page_name or is_tab_screen_target(target) or is_app_scoped_target(target))
        and not is_non_interaction_root_target(target)
    ]

def get_runtime_state_signature(driver: webdriver.Remote) -> tuple[str, str]:
    page_name, _ = get_current_page_info(driver)
    targets = get_page_interaction_targets(page_name, find_labeled_elements(driver, ensure_stable=False))
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
    force_replay: bool = False,
) -> None:
    if force_replay:
        shortest_path = find_shortest_recovery_path(
            recovery_graph,
            start_state_signature,
            expected_state_signature,
        )
        if shortest_path is None:
            shortest_path = path_targets
            print(f"[recover] force replay fallback path to {expected_page}: {len(shortest_path)} clicks")
        else:
            print(f"[recover] force replay shortest path to {expected_page}: {len(shortest_path)} clicks")
        replay_path(driver, shortest_path)
        return

    if runtime_state_matches(driver, expected_page, expected_state_signature):
        return

    shortest_path = find_shortest_recovery_path(recovery_graph, start_state_signature, expected_state_signature)
    if shortest_path is None:
        shortest_path = path_targets
        print(f"[recover] replay fallback path to {expected_page}: {len(shortest_path)} clicks")
    else:
        print(f"[recover] replay shortest path to {expected_page}: {len(shortest_path)} clicks")
    replay_path(driver, shortest_path)

def explore_app(driver: webdriver.Remote, output_dir: Path) -> dict:
    # A new state is recorded when either the visible page signature changes or the screenshot state changes.
    visited_interactions: set[str] = set()
    executed_component_ids: set[str] = set()
    state_pending_interactions: dict[str, set[str]] = {}
    completed_states: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    nodes_by_id: dict[str, dict] = {}
    known_state_nodes: dict[str, dict[str, str]] = {}
    recovery_graph: dict[str, list[tuple[str, InteractionTarget]]] = {}
    canonical_paths_by_state: dict[str, list[InteractionTarget]] = {}
    edges: list[dict] = []
    results: list[dict] = []
    state_debug: list[dict] = []

    restart_app(driver)
    start_page, _ = get_current_page_info(driver)
    start_targets = get_page_interaction_targets(start_page, find_labeled_elements(driver, ensure_stable=True))
    start_state_signature = state_signature_from_targets(start_page, start_targets)
    start_node_id, _, _ = capture_or_reuse_screenshot(
        driver,
        output_dir,
        start_page,
        start_state_signature,
        known_state_nodes,
    )
    update_utg_node(nodes_by_id, start_node_id, start_page)
    print_debug_labels(
    "START STATE",
    start_page,
    start_node_id,
    start_state_signature,
    start_targets,
    )

    def remember_state_path(state_signature: str, path_targets: list[InteractionTarget]) -> None:
        old_path = canonical_paths_by_state.get(state_signature)
        if old_path is None or len(path_targets) < len(old_path):
            canonical_paths_by_state[state_signature] = list(path_targets)

    def recovery_path_for_state(
        state_signature: str,
        fallback_path: list[InteractionTarget],
    ) -> list[InteractionTarget]:
        return list(canonical_paths_by_state.get(state_signature, fallback_path))

    def mark_interaction_executed(interaction_key: str, component_id: str) -> None:
        visited_interactions.add(interaction_key)
        executed_component_ids.add(component_id)

    remember_state_path(start_state_signature, [])

    def dfs(
        current_page: str,
        current_node_id: str,
        current_path: list[InteractionTarget],
        current_tab_interaction: bool = False,
    ) -> None:
        wait_for_page_render(driver, reason=f"state {current_page}")
        page_name, _ = get_current_page_info(driver)

        targets = get_page_interaction_targets(page_name, find_labeled_elements(driver, ensure_stable=True))
        if not targets:
            empty_state_signature = state_signature_from_targets(page_name, targets)
            empty_state_context_key = empty_state_signature
            remember_state_path(empty_state_signature, current_path)
            current_node_id_runtime, _, _ = capture_or_reuse_screenshot(
                driver,
                output_dir,
                page_name,
                empty_state_signature,
                known_state_nodes,
            )
            update_utg_node(nodes_by_id, current_node_id_runtime, page_name)
            state_debug.append(
                {
                    "page": page_name,
                    "node_id": current_node_id_runtime,
                    "state_signature": empty_state_signature,
                    "state_context_key": empty_state_context_key,
                    "page_path": [],
                    "target_count": 0,
                    "targets": [],
                }
            )
            completed_states.add(empty_state_context_key)
            return

        state_signature = state_signature_from_targets(page_name, targets)
        remember_state_path(state_signature, current_path)
        current_node_id_runtime, _, _ = capture_or_reuse_screenshot(
            driver,
            output_dir,
            page_name,
            state_signature,
            known_state_nodes,
            tab_interaction=current_tab_interaction,
        )
        update_utg_node(nodes_by_id, current_node_id_runtime, page_name)
        effective_from_node_id = current_node_id_runtime or current_node_id
        print_debug_labels(
            "DFS CURRENT STATE",
            page_name,
            current_node_id_runtime,
            state_signature,
            targets,
        )
        
        state_context_key = state_signature

        state_debug.append(
            {
                "page": page_name,
                "node_id": current_node_id_runtime,
                "state_signature": state_signature,
                "state_context_key": state_context_key,
                "page_path": [],
                "target_count": len(targets),
                "targets": describe_interaction_targets(targets),
            }
        )

        if state_context_key in completed_states:
            print(f"[dfs] skip completed state: {page_name}")
            return

        pending_ids = state_pending_interactions.setdefault(
            state_context_key,
            {target.component_id for target in targets},
        )

        def can_try_target_on_current_path(target: InteractionTarget) -> bool:
            interaction_key = f"{state_context_key}::{target.component_id}"
            if target.component_id in executed_component_ids:
                return False
            if interaction_key in visited_interactions:
                return False
            return True
        pending_targets = sort_interaction_targets_for_exploration(
            [
                target
                for target in targets
                if target.component_id in pending_ids
                and can_try_target_on_current_path(target)
            ]
        )

        if not pending_targets:
            if pending_ids:
                for target in targets:
                    if target.component_id not in pending_ids:
                        continue
                    interaction_key = f"{state_context_key}::{target.component_id}"
                    mark_interaction_executed(interaction_key, target.component_id)
                pending_ids.clear()
                print(f"[dfs] no new clickable path for pending targets; mark executed: {page_name}")
            completed_states.add(state_context_key)
            print(f"[dfs] completed state: {page_name}")
            return

        for target in pending_targets:
            interaction_visit_key = f"{state_context_key}::{target.component_id}"
            if interaction_visit_key in visited_interactions:
                continue

            print(f"[dfs] {page_name} -> click {target.component_id} ({target.text})")
            try:
                wait_for_page_render(driver, reason=f"before click {target.component_id}")
                element = find_element_for_target(driver, target)
                element.click()
                interruptible_sleep(SETTLE_SECONDS)

                next_page, _ = get_current_page_info(driver)
                next_targets = get_page_interaction_targets(next_page, find_labeled_elements(driver, ensure_stable=True))
                next_state_signature = state_signature_from_targets(next_page, next_targets)
                print_debug_labels(
                    f"AFTER CLICK {target.component_id}",
                    next_page,
                    None,
                    next_state_signature,
                    next_targets,
                )
                next_tab_interaction = is_tab_interaction(target)
                next_node_id, screenshot_path, _ = capture_or_reuse_screenshot(
                    driver,
                    output_dir,
                    next_page,
                    next_state_signature,
                    known_state_nodes,
                    tab_interaction=next_tab_interaction,
                )
                update_utg_node(nodes_by_id, next_node_id, next_page)

                same_page_signature = next_page == page_name and next_state_signature == state_signature
                same_screenshot_state = next_node_id == effective_from_node_id
                no_observable_change = same_page_signature and same_screenshot_state
                next_path = current_path + [target]

                condition_text = condition_text_for_target(target)
                if not no_observable_change:
                    remember_state_path(next_state_signature, next_path)
                    add_recovery_transition(recovery_graph, state_signature, next_state_signature, target)
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
                        "status": "no_change" if no_observable_change else "success",
                        "screenshot": str(screenshot_path),
                    }
                )

                if no_observable_change:
                    mark_interaction_executed(interaction_visit_key, target.component_id)
                    pending_ids.discard(target.component_id)
                    print(
                        "[dfs] no observable change after "
                        f"{target.component_id}; continue with other targets"
                    )
                    continue

                mark_interaction_executed(interaction_visit_key, target.component_id)
                pending_ids.discard(target.component_id)

                next_state_context_key = next_state_signature
                if next_state_context_key in completed_states:
                    print(f"[dfs] next state already completed: {next_page}")
                else:
                    dfs(next_page, next_node_id, next_path, next_tab_interaction)
                if pending_ids:
                    recover_to_state(
                        driver,
                        recovery_path_for_state(state_signature, current_path),
                        recovery_graph,
                        start_state_signature,
                        page_name,
                        state_signature,
                    )
                else:
                    print(f"[recover] skip completed state: {page_name}")
            except Exception as exc:
                mark_interaction_executed(interaction_visit_key, target.component_id)
                pending_ids.discard(target.component_id)
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
                    recover_to_state(
                        driver,
                        recovery_path_for_state(state_signature, current_path),
                        recovery_graph,
                        start_state_signature,
                        page_name,
                        state_signature,
                    )
                else:
                    print(f"[recover] skip completed state: {page_name}")

        if pending_ids:
            for target in targets:
                if target.component_id not in pending_ids:
                    continue
                interaction_key = f"{state_context_key}::{target.component_id}"
                mark_interaction_executed(interaction_key, target.component_id)
            pending_ids.clear()
            completed_states.add(state_context_key)
            print(f"[dfs] remaining pending targets marked executed: {page_name}")
            print(f"[dfs] completed state: {page_name}")
            return

        if not pending_ids:
            completed_states.add(state_context_key)
            print(f"[dfs] completed state: {page_name}")

    dfs(start_page, start_node_id, [])
    return {
        "N": list(nodes_by_id.values()),
        "E": edges,
        "results": results,
        "state_debug": state_debug,
    }

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] {path}")

def is_retryable_appium_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "connection refused",
        "cannot connect",
        "connection reset",
        "connection aborted",
        "connection closed",
        "connectionerror",
        "read timed out",
        "timed out",
        "timeout",
        "max retries exceeded",
        "socket hang up",
        "econnreset",
        "invalid session",
        "session deleted",
        "session not created",
        "uiautomator2",
        "instrumentation",
        "proxy",
        "httpconnectionpool",
        "adbexec",
        "can't find service: settings",
        "hidden_api_policy",
        "settings delete global",
        "exited with code 20",
        "android device/emulator unavailable",
        "no devices/emulators found",
        "device offline",
        "offline",
        "unauthorized",
        "device unauthorized",
        "device not found",
        "emulator not found",
        "missing",
        "emulator",
    )
    return isinstance(exc, (WebDriverException, TimeoutException, ConnectionError, OSError)) and any(
        marker in message for marker in retry_markers
    )

def is_appium_connection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "127.0.0.1" in message
        and "4723" in message
        and (
            "/session" in message
            or "httpconnectionpool" in message
            or "failed to establish a new connection" in message
            or "connection refused" in message
            or "max retries exceeded" in message
        )
    )

def run_appium_exploration_once(output_dir: Path) -> dict:
    driver = build_driver()
    try:
        utg_data = explore_app(driver, output_dir)

        print("[cleanup] terminate app")
        try:
            driver.terminate_app(APP_PACKAGE)
        except Exception:
            pass

        return utg_data
    finally:
        try:
            driver.quit()
        except Exception as exc:
            print(f"[cleanup] driver quit failed: {exc}")

def run_appium_exploration_with_retry(output_dir: Path) -> tuple[dict, list[dict]]:
    retry_records: list[dict] = []
    max_attempts = max(1, APPIUM_RUN_RETRY_COUNT + 1)
    delay_seconds = APPIUM_RUN_RETRY_DELAY_SECONDS

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[retry] appium attempt {attempt}/{max_attempts}")
            return run_appium_exploration_once(output_dir), retry_records
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            retryable = is_retryable_appium_error(exc)
            cold_start = (
                retryable
                and EMULATOR_COLD_START_ON_APPIUM_CONNECTION_ERROR
                and is_appium_connection_error(exc)
            )
            restart_appium = (
                retryable
                and APPIUM_AUTO_RESTART_ON_CONNECTION_ERROR
                and is_appium_connection_error(exc)
            )
            retry_records.append(
                {
                    "attempt": attempt,
                    "retryable": retryable,
                    "restart_appium_server": restart_appium,
                    "cold_start_emulator": cold_start,
                    "error": str(exc),
                }
            )
            if not retryable or attempt >= max_attempts:
                raise

            print(f"[retry] appium connection/session failure on attempt {attempt}: {exc}")
            if restart_appium:
                restart_appium_server()
            if cold_start:
                cold_start_emulator()
            print(f"[retry] waiting {delay_seconds:.1f}s before retry")
            interruptible_sleep(delay_seconds)
            delay_seconds *= APPIUM_RUN_RETRY_BACKOFF

    raise RuntimeError("unreachable appium retry state")

def run_single_app(category: str, app_name: str) -> None:
    generated_code_dir = ensure_generated_code_dir(category, app_name)
    output_dir = OUTPUT_ROOT / category / app_name
    if CLEAR_OUTPUT_DIR_BEFORE_RERUN and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.json"
    original_res_path = generated_code_dir / "res.txt"
    labeled_res_path = get_output_res_path(original_res_path)

    if not original_res_path.exists():
        raise FileNotFoundError(f"res.txt not found: {original_res_path}")

    compile_failures: list[dict] = []
    try:
        print(f"[label] generating {LABEL_OUTPUT_FILENAME} from {original_res_path} (timeout={int(LABEL_GENERATION_TIMEOUT_SECONDS)}s)")
        process_res_file_with_timeout(original_res_path, WRITE_LABEL_FILE, LABEL_GENERATION_TIMEOUT_SECONDS)
        if not labeled_res_path.exists():
            raise FileNotFoundError(f"Labeled res file not found after labeling: {labeled_res_path}")

        print(f"[flow] labeled code -> project: {labeled_res_path}")
        materialize_labeled_res_to_project(labeled_res_path, PROJECT_PATH, CLEAR_COMPONENTS)
        if SKIP_COMPILE:
            print("[compile] skipped")
            print(
                "[compile] warning: project files were updated but APK was not rebuilt/installed; "
                "make sure the running app is loading this labeled project."
            )
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

    retry_records: list[dict] = []
    try:
        utg_data, retry_records = run_appium_exploration_with_retry(output_dir)
    except Exception as exc:
        failure_stage = "device_unavailable" if "android device/emulator unavailable" in str(exc).lower() else "appium"
        write_json(
            report_path,
            {
                "category": category,
                "app_name": app_name,
                "generated_code_dir": str(generated_code_dir),
                "original_res": str(original_res_path),
                "labeled_res": str(labeled_res_path),
                "compile_status": "skipped" if SKIP_COMPILE else "success",
                "compile_failures": compile_failures,
                "appium_retry_count": len(retry_records),
                "appium_retries": retry_records,
                "interaction_failures": [
                    {
                        "stage": failure_stage,
                        "error": str(exc),
                    }
                ],
                "results": [],
            },
        )
        raise

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
            "appium_retry_count": len(retry_records),
            "appium_retries": retry_records,
            "result_count": len(utg_data["results"]),
            "interaction_failure_count": len(interaction_failures),
            "interaction_failures": interaction_failures,
            "state_debug": utg_data.get("state_debug", []),
            "results": utg_data["results"],
        },
    )

def main(baseline_name: str) -> None:
    global BASELINE_NAME
    global GENERATED_CODE_ROOT
    global OUTPUT_ROOT
    global MISMATCH_OUTPUT_JSON
    global MISMATCH_OUTPUT_CSV

    BASELINE_NAME = baseline_name

    GENERATED_CODE_ROOT = (
        REPO_ROOT
        / "results"
        / "code_results"
        # / "baselines"
        / BASELINE_NAME
    )

    OUTPUT_ROOT = (
        REPO_ROOT
        / "results"
        / "appium_screenshots"
        # / "baselines"
        / BASELINE_NAME
    )
    MISMATCH_OUTPUT_JSON = OUTPUT_ROOT / "mismatch_vs_dataset.json"
    MISMATCH_OUTPUT_CSV = OUTPUT_ROOT / "mismatch_vs_dataset.csv"

    print("\n" + "=" * 80)
    print(f"[baseline] {BASELINE_NAME}")
    print("=" * 80)

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
        finally:
            records = write_mismatch_reports()
            current_record = next(
                (item for item in records if item["Category"] == category and item["App"] == app_name),
                None,
            )
            if current_record is not None:
                print(
                    "[mismatch] "
                    f"{category}/{app_name}: "
                    f"DataFindCount={current_record['DataFindCount']}, "
                    f"BaselineCount={current_record['BaselineCount']}, "
                    f"Difference={current_record['Difference']}, "
                    f"Match={current_record['Match']}"
                )

        report_path = OUTPUT_ROOT / category / app_name / "run_report.json"
        if report_path.exists():
            report_data = read_json_file(report_path)
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
        for baseline_name in BASELINE_NAMES:
            main(baseline_name)

    except KeyboardInterrupt:
        print("[appium] interrupted by user")
