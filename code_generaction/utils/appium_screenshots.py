import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =========================
# Configuration
# =========================
GRAPH_PATH = r"D:\ui2code\InteractionSequence\data_Find"
OUTPUT_DIR = r"D:\ui2code\InteractionSequence\results\appium_screenshots"
APPIUM_SERVER_URL = "http://127.0.0.1:4723"
# DEVICE_NAME = "Android"
# APP_PACKAGE = "com.MyApp"  # TODO: set your real package name
# APP_ACTIVITY = "com.MyApp.MainActivity"
DEVICE_NAME = "emulator-5556"
APP_PACKAGE = "com.myproject"
APP_ACTIVITY = "com.myproject.MainActivity"
NO_RESET = True

# Performance settings
STARTUP_WAIT_SECONDS = 10
SETTLE_SECONDS = 2
ELEMENT_TIMEOUT_SECONDS = 3
PRE_INTERACTION_RENDER_WAIT_SECONDS = 2

# Cache for successful locators
LOCATOR_CACHE = {}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_schema(data: dict) -> Tuple[str, str]:
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    return "", ""


def parse_graph(graph_path: Path):
    data = load_json(graph_path)
    n_key, e_key = detect_schema(data)
    if not n_key or not e_key:
        raise ValueError(f"Unsupported graph schema: {graph_path}")

    nodes = data.get(n_key, [])
    edges = data.get(e_key, [])
    if not nodes:
        raise ValueError("Graph has no nodes.")

    node_ids = set()
    for n in nodes:
        node_id = str(n.get("id", "")).strip()
        if node_id:
            node_ids.add(node_id)

    parsed_edges = []
    for e in edges:
        edge_id = str(e.get("id", "")).strip()
        src = str(e.get("from", "")).strip()
        dst = str(e.get("to", "")).strip()
        if not edge_id or not src or not dst:
            continue
        parsed_edges.append(
            {
                "id": edge_id,
                "from": src,
                "to": dst,
                "conditions": e.get("conditions", []),
            }
        )

    start_node = str(nodes[0].get("id", "")).strip()
    return start_node, node_ids, parsed_edges


# def build_driver(
#     appium_server_url: str,
#     device_name: str,
#     app_package: str,
#     app_activity: Optional[str],
#     no_reset: bool,
# ):
#     options = UiAutomator2Options()
#     options.platform_name = "Android"
#     options.automation_name = "UiAutomator2"
#     options.device_name = device_name
#     options.no_reset = no_reset
#     options.new_command_timeout = 300
#     options.app_package = app_package
#     if app_activity:
#         options.app_activity = app_activity
    
#     print(f"Connecting to Appium: {appium_server_url}")
#     driver = webdriver.Remote(appium_server_url, options=options)
#     print("✅ Driver connected")
#     return driver


def build_driver(
    appium_server_url: str,
    device_name: str,
    app_package: str,
    app_activity: Optional[str],
    no_reset: bool,
):
    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.automation_name = "UiAutomator2"

    # 指定具体模拟器，例如 emulator-5556
    options.device_name = device_name
    options.udid = device_name

    options.no_reset = no_reset
    options.new_command_timeout = 300
    options.app_package = app_package

    if app_activity:
        options.app_activity = app_activity
    
    print(f"Connecting to Appium: {appium_server_url}")
    driver = webdriver.Remote(appium_server_url, options=options)
    print("✅ Driver connected")
    return driver

def find_element_by_testid(driver, test_id: str, timeout: float = ELEMENT_TIMEOUT_SECONDS):
    """优化版：优先使用最可能成功的定位器"""
    
    # 检查缓存
    if test_id in LOCATOR_CACHE:
        cached_by, cached_value = LOCATOR_CACHE[test_id]
        try:
            element = WebDriverWait(driver, 1).until(
                EC.presence_of_element_located((cached_by, cached_value))
            )
            print(f"  🔍 [CACHED] 找到元素")
            return element
        except:
            del LOCATOR_CACHE[test_id]
    
    app_pkg = ""
    try:
        app_pkg = driver.current_package or ""
    except:
        pass
    
    # 根据之前的成功经验，XPATH 直接匹配是最有效的
    # 所以把它放在第一位
    locators = []
    
    # 优先级1: 直接 XPATH 匹配（从日志看这个成功了）
    locators.append((AppiumBy.XPATH, f'//*[@resource-id="{test_id}"]'))
    
    # 优先级2: 带包名的 XPATH
    if app_pkg:
        locators.append((AppiumBy.XPATH, f'//*[@resource-id="{app_pkg}:id/{test_id}"]'))
    
    # 优先级3: ID 定位
    locators.append((AppiumBy.ID, test_id))
    if app_pkg:
        locators.append((AppiumBy.ID, f"{app_pkg}:id/{test_id}"))
    
    # 优先级4: 转义版本
    escaped_id = test_id.replace('-->', '_')
    locators.append((AppiumBy.XPATH, f'//*[@resource-id="{escaped_id}"]'))
    locators.append((AppiumBy.ID, escaped_id))
    
    # 尝试每个定位器
    for by, value in locators:
        try:
            print(f"  🔍 尝试定位: {by}={value[:50]}...")
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            # 缓存成功的定位器
            LOCATOR_CACHE[test_id] = (by, value)
            print(f"  ✅ 找到元素")
            return element
        except TimeoutException:
            continue
        except Exception as e:
            continue
    
    raise NoSuchElementException(f"找不到元素: {test_id}")

def wait_and_click_by_testid(driver, test_id: str, timeout_s: int = ELEMENT_TIMEOUT_SECONDS):
    """点击 testID 元素"""
    # 检查缓存
    if test_id in LOCATOR_CACHE:
        cached_by, cached_value = LOCATOR_CACHE[test_id]
        try:
            element = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((cached_by, cached_value))
            )
            element.click()
            print(f"  ✅ [CACHED] Clicked: {test_id}")
            return
        except:
            del LOCATOR_CACHE[test_id]
    
    # 查找并点击
    element = find_element_by_testid(driver, test_id, timeout_s)
    element.click()
    
    # 缓存成功的定位方式（简化版，实际使用时需要记录 by, value）
    print(f"  ✅ Clicked: {test_id}")


def wait_and_input_by_testid(driver, test_id: str, text: str, timeout_s: int = ELEMENT_TIMEOUT_SECONDS):
    """在 testID 元素中输入文本"""
    element = find_element_by_testid(driver, test_id, timeout_s)
    element.click()
    element.clear()
    element.send_keys(text)
    print(f"  ✅ Input to: {test_id}")


def edge_requires_set_text(edge: dict) -> bool:
    conditions = edge.get("conditions", [])
    if not isinstance(conditions, list):
        return False
    for c in conditions:
        if "settextevent" in str(c).lower():
            return True
    return False


def wait_for_page_render(wait_s: float = PRE_INTERACTION_RENDER_WAIT_SECONDS):
    """Wait before interacting so the page has time to render."""
    if wait_s <= 0:
        return
    print(f"  Waiting {wait_s}s for page render...")
    time.sleep(wait_s)


def execute_edge_action(driver, edge: dict, settle_s: float):
    edge_id = str(edge.get("id", "")).strip()
    if not edge_id:
        raise RuntimeError("edge id is empty")
    
    print(f"  🎯 Executing: {edge_id}")
    wait_for_page_render()
    
    if edge_requires_set_text(edge):
        wait_and_input_by_testid(driver, edge_id, text="test_input")
    else:
        wait_and_click_by_testid(driver, edge_id)
    
    time.sleep(settle_s)


def restart_app(driver, app_package: str, settle_s: float):
    """重启应用"""
    try:
        driver.terminate_app(app_package)
    except:
        pass
    driver.activate_app(app_package)
    time.sleep(settle_s)
    print("🔄 App restarted")


def screenshot_name_from_node_id(node_id: str) -> str:
    node_id = str(node_id).strip()
    if node_id.lower().endswith(".png"):
        return node_id
    return f"{node_id}.png"


def recover_to_path(driver, app_package: str, path_edge_ids: List[str], edges_by_id: Dict[str, dict], settle_s: float):
    """恢复到指定路径"""
    restart_app(driver, app_package, STARTUP_WAIT_SECONDS)
    for edge_id in path_edge_ids:
        edge = edges_by_id.get(edge_id)
        if edge:
            execute_edge_action(driver, edge, settle_s)


def dump_all_elements(driver):
    """调试：打印所有元素"""
    print("\n=== Debug: All elements with resource-id ===")
    try:
        elements = driver.find_elements(AppiumBy.XPATH, "//*[@resource-id]")
        print(f"Found {len(elements)} elements with resource-id:")
        for elem in elements[:20]:
            rid = elem.get_attribute("resource-id")
            text = elem.get_attribute("text") or ""
            clickable = elem.get_attribute("clickable") or ""
            if rid and rid != "android:id/":
                print(f"  • {rid}")
                if text:
                    print(f"    text: {text}")
                if clickable == 'true':
                    print(f"    [CLICKABLE]")
    except Exception as e:
        print(f"Debug error: {e}")
    print("========================================\n")


def run_appium_screenshots():
    """主函数"""
    print("=" * 70)
    print("UTG-based Appium Screenshot Capture")
    print("=" * 70)
    
    # 转换为 Path 对象
    graph_path = Path(GRAPH_PATH)
    output_dir = Path(OUTPUT_DIR)
    
    # 解析图
    start_node, node_ids, edges = parse_graph(graph_path)
    print(f"📊 Graph loaded: {len(edges)} edges, {len(node_ids)} nodes")
    print(f"🏁 Start node: {start_node}")
    
    # 构建邻接表
    edges_by_id = {e["id"]: e for e in edges}
    adj = defaultdict(list)
    for e in edges:
        adj[e["from"]].append(e)
    
    # 创建 driver
    driver = build_driver(
        APPIUM_SERVER_URL,
        DEVICE_NAME,
        APP_PACKAGE,
        APP_ACTIVITY,
        NO_RESET
    )
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 调试：打印所有元素
    time.sleep(STARTUP_WAIT_SECONDS)  # 等待应用完全加载
    dump_all_elements(driver)
    
    # 统计
    visited_edges = set()
    results = []
    success_count = 0
    failed_count = 0
    
    # DFS 遍历执行
    def dfs_execute(current_node: str, current_path: List[str]):
        nonlocal success_count, failed_count
        
        for edge in adj.get(current_node, []):
            edge_id = edge["id"]
            
            if edge_id in visited_edges:
                continue
            
            visited_edges.add(edge_id)
            print(f"\n📌 [{len(visited_edges)}/{len(edges)}] Edge: {edge_id}")
            
            try:
                # 执行边操作
                execute_edge_action(driver, edge, SETTLE_SECONDS)
                
                # 截图（目标节点）
                screenshot_name = screenshot_name_from_node_id(edge["to"])
                screenshot_path = output_dir / screenshot_name
                driver.save_screenshot(str(screenshot_path))
                
                results.append({
                    "edge_id": edge_id,
                    "from": edge["from"],
                    "to": edge["to"],
                    "status": "success",
                    "screenshot": str(screenshot_path)
                })
                success_count += 1
                print(f"  📸 Screenshot saved: {screenshot_name}")
                
                # 递归遍历
                current_path.append(edge_id)
                dfs_execute(edge["to"], current_path)
                
                # 回溯：回到父节点
                if current_path:
                    current_path.pop()
                    parent_path = current_path.copy()
                    print(f"  ↩️  Backtracking to parent...")
                    recover_to_path(driver, APP_PACKAGE, parent_path, edges_by_id, SETTLE_SECONDS)
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append({
                    "edge_id": edge_id,
                    "from": edge["from"],
                    "to": edge["to"],
                    "status": "failed",
                    "error": str(e)
                })
                failed_count += 1
                
                # 尝试恢复
                try:
                    recover_to_path(driver, APP_PACKAGE, current_path, edges_by_id, SETTLE_SECONDS)
                except:
                    print(f"  ⚠️  Recovery failed, restarting...")
                    restart_app(driver, APP_PACKAGE, STARTUP_WAIT_SECONDS)
    
    try:
        # 启动应用
        restart_app(driver, APP_PACKAGE, STARTUP_WAIT_SECONDS)
        
        # 截取起始节点
        start_screenshot = output_dir / screenshot_name_from_node_id(start_node)
        driver.save_screenshot(str(start_screenshot))
        print(f"📸 Start node screenshot: {start_screenshot}")
        
        # 开始 DFS 遍历
        dfs_execute(start_node, [])
        
        # 报告未访问的边
        unreachable = [e for e in edges if e["id"] not in visited_edges]
        for edge in unreachable:
            results.append({
                "edge_id": edge["id"],
                "from": edge["from"],
                "to": edge["to"],
                "status": "unreachable",
                "error": "Edge not reachable from start node"
            })
            failed_count += 1
            print(f"⚠️  Unreachable: {edge['id']}")
        
    finally:
        driver.quit()
    
    # 保存报告
    report = {
        "graph_path": str(graph_path),
        "start_node": start_node,
        "total_edges": len(edges),
        "success": success_count,
        "failed": failed_count,
        "results": results,
        "locator_cache": LOCATOR_CACHE
    }
    
    report_path = output_dir / "run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_appium_screenshots()
