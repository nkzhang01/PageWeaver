import json
from openai import OpenAI
import subprocess
import shutil
import httpx
import base64
import re
import os
import sys
import openai
import glob
import time
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(CODEGEN_DIR, ".."))
if CODEGEN_DIR not in sys.path:
    sys.path.insert(0, CODEGEN_DIR)

from utils.compiler.Flutter_compiler import compile_and_run_flutter_project
from utils.compiler.hvigor_compiler import compile_and_run_arkts_project
from utils.compiler.RN_compiler import run_gradlew_assemble
from utils.connect_neo4j import search_neo4j_by_id


lang_index = 0
UI_class_list = ["RN","Flutter","ArkTS"]
UI_class = UI_class_list[lang_index]
UItype_list = ["React Native","Flutter","ArkTS"]
UItype = UItype_list[lang_index]
# project_template_folder_list = ["rn_project_folder","","flutter_project_folder","arkts_project_folder"]
project_template_folder_list = [r"D:\ui2code\testAndroid\AwesomeProject","flutter_project_folder","arkts_project_folder"]
project_template_folder = project_template_folder_list[lang_index]
if project_template_folder and not os.path.isabs(project_template_folder):
    project_template_folder = os.path.join(SCRIPT_DIR, project_template_folder)
picture_folder = os.path.join(ROOT_DIR, "data")
prompt_templates_root = os.path.join(CODEGEN_DIR, "utils", "prompt_templates")
prompt_folder = os.path.join(prompt_templates_root, UI_class)
# component_info_folder = os.path.join('D:\ui2code\DeclarUI\Scripts\results\extract_results',UI_class)
# output_folder = os.path.join('D:\ui2code\DeclarUI\Scripts\results\code_results','baseline_gpt')

component_info_folder = os.path.join(ROOT_DIR, "results", "extract_results")
output_folder = os.path.join(ROOT_DIR, "results", "code_results", "gpt_5")

if not os.path.isdir(prompt_folder):
    raise FileNotFoundError(f"Prompt folder not found: {prompt_folder}")


os.environ["OPENAI_API_BASE"] = 'http://47.76.75.25:9000/v1'
os.environ["OPENAI_API_KEY"] = "8iYCoxq7PnoMPqzjDb59B50288374c40A73fAeA721454dF3"  # gpt-5/claude-3.5-sonnet

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    filename='gpt_code_generate.log',   
    filemode='a',         
    format='%(asctime)s - %(levelname)s \n %(message)s'  
    )


def get_uitype_detail():
    """
    Get the detail of the UI type
    """
    type_detail = {}
    if UItype == "React Native":
        save_type = "js"
        entry_func_name = "App"
        extend_name = "Screen"
        nav_code_template = ["navigation.navigate('{nav_id}'","'{nav_id}'"]
    elif UItype == "Flutter":
        save_type = "dart"
        entry_func_name = "main"
        extend_name = "Screen"
        nav_code_template = ["Navigator.pushNamed(context, '{nav_id}'","Navigator.pushReplacementNamed(context, '{nav_id}'","Navigator.pushNamed(\ncontext,\n'{nav_id}'"]     
    elif UItype =="ArkTS":
        save_type = "ets"
        entry_func_name = "main_pages"
        extend_name = "Page"
        nav_code_template = ["pushUrl({{ url: 'pages/{nav_id}'"]
    else:
        save_type = "txt"
        entry_func_name = ""
        extend_name = ""
        nav_code_template =[""]
    type_detail["save_type"] = save_type
    type_detail["entry_func_name"] = entry_func_name
    type_detail["extend_name"] = extend_name
    type_detail['nav_code_template'] = nav_code_template
    return type_detail


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_json_schema(data):
    if isinstance(data.get("N"), list) and isinstance(data.get("E"), list):
        return "N", "E"
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "nodes", "edges"
    return None, None


def page_subgraph_dir(app_classification, app_name, kind):
    return os.path.join(picture_folder, app_classification, app_name, "page_subgraphs", kind)


def get_cluster_page_list(app_classification, app_name):
    """
    获取聚类页面列表（不是每张截图），从 page_subgraphs/state/*.json 读取。
    """
    state_dir = page_subgraph_dir(app_classification, app_name, "state")
    page_list = []

    if os.path.exists(state_dir) and os.path.isdir(state_dir):
        for fn in sorted(os.listdir(state_dir)):
            if not fn.endswith(".json"):
                continue
            page_list.append(os.path.splitext(fn)[0])

    if not page_list:
        raise FileNotFoundError(f"No PSG state subgraphs found in: {state_dir}")

    page_list = list(dict.fromkeys(page_list))
    return page_list


def normalize_screen_id_from_node_id(node_id):
    s = str(node_id or "").strip().replace("\\", "/")
    s = os.path.basename(s)
    for ext in [".png", ".jpg", ".jpeg"]:
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return os.path.splitext(s)[0]


def extract_home_screen_from_psg(app_classification, app_name, page_name):
    """
    从 PSG(state 子图)提取该聚类页面的代表截图 id（优先第一个节点）。
    """
    state_path = os.path.join(page_subgraph_dir(app_classification, app_name, "state"), f"{page_name}.json")
    if not os.path.exists(state_path):
        return page_name

    data = load_json_file(state_path)
    n_key, _ = detect_json_schema(data)
    if not n_key:
        return page_name

    nodes = data.get(n_key, [])
    if not nodes:
        return page_name

    # generateSubGraphy 已保证首页面在第一个节点
    first_node_id = nodes[0].get("id", "")
    home = normalize_screen_id_from_node_id(first_node_id)
    return home or page_name


def resolve_image_path_for_screen(app_classification, app_name, screen_id):
    app_dir = os.path.join(picture_folder, app_classification, app_name)
    candidates = [
        os.path.join(app_dir, f"{screen_id}.png"),
        os.path.join(app_dir, f"{screen_id}.jpg"),
        os.path.join(app_dir, f"{screen_id}.jpeg"),
        os.path.join(app_dir, "states", f"{screen_id}.png"),
        os.path.join(app_dir, "states", f"{screen_id}.jpg"),
        os.path.join(app_dir, "states", f"{screen_id}.jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return ""


def collect_screen_ids_from_graph(graph_path):
    if not os.path.exists(graph_path):
        return []
    data = load_json_file(graph_path)
    n_key, _ = detect_json_schema(data)
    if not n_key:
        return []

    screen_ids = []
    for node in data.get(n_key, []):
        nid = str(node.get("id", "")).strip()
        if not nid:
            continue
        sid = normalize_screen_id_from_node_id(nid)
        if sid:
            screen_ids.append(sid)
    return list(dict.fromkeys(screen_ids))


def collect_graph_image_paths_for_page(app_classification, app_name, page_name, type):
    """
    收集该聚类页面对应 PSG/PTG 子图中所有节点对应截图路径。
    """
    Graph_path = None
    if type == "PSG":
        Graph_path = os.path.join(
            picture_folder, app_classification, app_name, "page_subgraphs", "state", page_name + ".json"
        )
    if type == "PTG":
        Graph_path = os.path.join(
            picture_folder, app_classification, app_name, "page_subgraphs", "transition", page_name + "_transition.json"
        )

    screen_ids = []
    screen_ids.extend(collect_screen_ids_from_graph(Graph_path))
    screen_ids = list(dict.fromkeys(screen_ids))

    image_paths = []
    for sid in screen_ids:
        p = resolve_image_path_for_screen(app_classification, app_name, sid)
        if p:
            image_paths.append(p)
    return list(dict.fromkeys(image_paths))


def collect_graph_base64_images_for_page(app_classification, app_name, page_name, type):
    image_paths = collect_graph_image_paths_for_page(app_classification, app_name, page_name, type)
    out = []
    for p in image_paths:
        try:
            out.append(encode_image(p))
        except Exception:
            continue
    return out


def get_utg_first_screen_id(app_classification, app_name):
    utg_path = os.path.join(picture_folder, app_classification, app_name, "utg.json")
    if not os.path.exists(utg_path):
        return ""
    data = load_json_file(utg_path)
    n_key, _ = detect_json_schema(data)
    if not n_key:
        return ""
    nodes = data.get(n_key, [])
    if not nodes:
        return ""
    first_id = str(nodes[0].get("id", "")).strip()
    return normalize_screen_id_from_node_id(first_id)


def get_entry_page_from_utg_component(app_classification, app_name):
    """
    Entry page is defined by the first node's `page` attribute in utg_component.json.
    """
    utg_component_path = os.path.join(picture_folder, app_classification, app_name, "utg_component.json")
    data = load_json_file(utg_component_path)
    n_key, _ = detect_json_schema(data)
    nodes = data.get(n_key, [])
    first_page = str(nodes[0].get("page", "")).strip()
    return first_page


def find_cluster_page_by_screen_id(app_classification, app_name, cluster_pages, screen_id):
    if not screen_id:
        return ""
    target = str(screen_id).strip()
    for page_name in cluster_pages:
        state_path = os.path.join(
            picture_folder, app_classification, app_name, "page_subgraphs", "state", page_name + ".json"
        )
        screen_ids = collect_screen_ids_from_graph(state_path)
        if target in screen_ids:
            return page_name
    return ""


def get_page_list(app_classification, app_name):
    """
    获取聚类页面列表，并为 ArkTS 生成 main_pages.json。
    """
    raw_pages = get_cluster_page_list(app_classification, app_name)
    if UItype == "ArkTS":
        page_list = [f"pages/{p}Page" for p in raw_pages]
        main_pages = {"src": page_list}
        os.makedirs(os.path.join(output_folder, app_classification, app_name), exist_ok=True)
        with open(os.path.join(output_folder, app_classification, app_name, "main_pages.json"), "w", encoding="utf-8") as f:
            json.dump(main_pages, f, ensure_ascii=False, indent=4)
        return page_list
    return raw_pages


def get_navigation_list(app_classification, app_name, page_name):
    transition_path = os.path.join(
        picture_folder, app_classification, app_name, "page_subgraphs", "transition", page_name + "_transition.json"
    )
    if not os.path.exists(transition_path):
        return []

    trans = load_json_file(transition_path)
    n_key, e_key = detect_json_schema(trans)
    if not n_key or not e_key:
        return []

    node_page = {}
    for node in trans.get(n_key, []):
        nid = str(node.get("id", "")).strip()
        p = str(node.get("page", "")).strip()
        if nid:
            node_page[nid] = p

    navigation_list = []
    for e in trans.get(e_key, []):
        src = str(e.get("from", "")).strip()
        dst = str(e.get("to", "")).strip()
        if not src or not dst:
            continue
        src_page = node_page.get(src, "")
        dst_page = node_page.get(dst, "")
        if src_page != page_name:
            continue
        if not dst_page or dst_page == page_name:
            continue
        if UItype == "ArkTS":
            navigation_list.append(dst_page + "Page")
        else:
            navigation_list.append(dst_page)

    return list(dict.fromkeys(navigation_list))


def get_navigation_requirements_with_mode(app_classification, app_name, page_name):
    """
    导航需求按“边”提取，包含目标节点 mode 信息：
    - from_node_id / from_mode
    - to_node_id / to_mode
    - nav_id（用于代码中的路由目标）
    """
    transition_path = os.path.join(
        picture_folder, app_classification, app_name, "page_subgraphs", "transition", page_name + "_transition.json"
    )
    if not os.path.exists(transition_path):
        return []

    trans = load_json_file(transition_path)
    n_key, e_key = detect_json_schema(trans)
    if not n_key or not e_key:
        return []

    node_info = {}
    for node in trans.get(n_key, []):
        nid = str(node.get("id", "")).strip()
        if not nid:
            continue
        node_info[nid] = {
            "page": str(node.get("page", "")).strip(),
            "mode": str(node.get("mode", "")).strip(),
        }

    reqs = []
    seen = set()
    for e in trans.get(e_key, []):
        src = str(e.get("from", "")).strip()
        dst = str(e.get("to", "")).strip()
        nav_id = str(e.get("id", "")).strip()
        if not src or not dst:
            continue
        src_info = node_info.get(src, {})
        dst_info = node_info.get(dst, {})
        src_page = src_info.get("page", "")
        dst_page = dst_info.get("page", "")
        
        if src_page != page_name:
            continue
        if not dst_page or dst_page == page_name:
            continue
        item = {
            "from_node_id": src,
            "from_page": src_page,
            "from_mode": src_info.get("mode", ""),
            "to_node_id": dst,
            "to_mode": dst_info.get("mode", ""),
            "to_page": dst_page,
            "nav_id": nav_id,
        }
        key = (
            item["from_node_id"],
            item["from_mode"],
            item["to_node_id"],
            item["to_mode"],
            item["nav_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        reqs.append(item)

    return reqs


def format_navigation_requirement(item):
    return (
        f"from_page={item.get('from_page', '')}, "
        f"from_node={item.get('from_node_id', '')}, "
        f"from_mode={item.get('from_mode', '')}, "
        f"to_page={item.get('to_page', '')}, "
        f"to_node={item.get('to_node_id', '')}, "
        f"to_mode={item.get('to_mode', '')}, "
        f"nav_id={item.get('nav_id', '')}"
    )


def get_ptg_data(app_classification, app_name):
    ptg_path = os.path.join(picture_folder, app_classification, app_name, "PTG.json")
    if not os.path.exists(ptg_path):
        raise FileNotFoundError(f"PTG not found: {ptg_path}")
    return load_json_file(ptg_path)


def KG_retrival(reference_path):
    refer_folder = "./component_info/ArkTS/references"
    # KG-retrival 
    with open(reference_path, 'r', encoding='utf-8') as f:
        refer_list = json.load(f)
    refer_info = []
    for refer_id in refer_list:
        path = os.path.join(refer_folder, refer_id+'.json')
        # If the component refer for this id has been queried before, read it directly
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                refer_info.append(json.load(f))
        # If it has not been queried, query and save it
        else:
            refer_res = search_neo4j_by_id(int(refer_id))
            refer_info.append(refer_res)
    return refer_info


def prompt_construct(
    app_classification,
    app_name,
    page_name,
    homePage=None,
    type="ScreenPSG",
    implemented_ids=None,
    not_implemented_ids=None,
    error_msg="",
    error_filenames="",
):
    page_id = page_name
    home_page_name = str(homePage).strip() if homePage is not None else ""
    if not home_page_name:
        home_page_name = str(page_name).strip()
    implemented_ids = implemented_ids or []
    not_implemented_ids = not_implemented_ids or []

    type_path = os.path.join(component_info_folder, app_classification, app_name, home_page_name + "_type.txt")
    reference_path = os.path.join(component_info_folder, app_classification, app_name, home_page_name + "_refer.json")

    transition_path = os.path.join(
        picture_folder, app_classification, app_name, "page_subgraphs", "transition", page_name + "_transition.json"
    )
    state_path = os.path.join(
        picture_folder, app_classification, app_name, "page_subgraphs", "state", page_name + ".json"
    )

    baseline_prompt_path = os.path.join(prompt_templates_root, "baseline", "baselinePrompt.txt")
    baselineNavigation_prompt_path = os.path.join(prompt_templates_root, "baseline", "NavigationPrompt.txt")
    PTG_prompt_path = os.path.join(prompt_folder, "PTGPrompt.txt")
    PSG_prompt_path = os.path.join(prompt_folder, "PSGPrompt.txt")
    navigation_prompt_path = os.path.join(prompt_folder, "PageNavigationPrompt.txt")
    refine_prompt_path = os.path.join(prompt_folder, "RefineCodePrompt.txt")
    compile_refine_prompt_path = os.path.join(prompt_templates_root, "RefineCompileErrorPrompt.txt")
    if UItype == "ArkTS":
        refer_refine_prompt_path = os.path.join(prompt_folder, "ReferRefinePrompt.txt")

    page_list = get_page_list(app_classification, app_name)
    navigation_reqs = get_navigation_requirements_with_mode(app_classification, app_name, page_name)
    if navigation_reqs:
        navigation_list = [
            (
                f"target_page={req.get('to_page', '')}, "
                f"target_mode={req.get('to_mode', '')}, "
                f"target_node_id={req.get('to_node_id', '')}, "
                f"from_mode={req.get('from_mode', '')}"
            )
            for req in navigation_reqs
        ]
    else:
        navigation_list = get_navigation_list(app_classification, app_name, page_name)
    type_detail = get_uitype_detail()
    extend_name = type_detail["extend_name"]
    save_type = "." + type_detail["save_type"]

    component_type = ""
    if os.path.exists(type_path):
        with open(type_path, "r", encoding="utf-8") as file:
            component_type = file.read()

    PSG = {}
    if os.path.exists(state_path):
        PSG = load_json_file(state_path)

    PTG = {}
    if os.path.exists(transition_path):
        PTG = load_json_file(transition_path)

    if type == "baseline":
        with open(baseline_prompt_path, "r", encoding="utf-8") as file:
            baseline_prompt = file.read()
        return baseline_prompt.format(page_id=page_id, page_list=page_list)

    if type == "baselineNavigation":
        with open(baselineNavigation_prompt_path, "r", encoding="utf-8") as file:
            baselineNavigation = file.read()
        return baselineNavigation.format(page_list=page_list)

    if type == "ScreenPSG":
        with open(PSG_prompt_path, "r", encoding="utf-8") as file:
            psg_prompt = file.read()
        return psg_prompt.format(
            page_id=page_id,
            Component_Type=component_type,
            PSG=PSG,
        )

    
    if type == "PTGRefine":
        with open(PTG_prompt_path, "r", encoding="utf-8") as file:
            ptg_prompt = file.read()
        refine_page_path = os.path.join(output_folder, app_classification, app_name, page_name + extend_name + save_type)
        page_code = ""
        if os.path.exists(refine_page_path):
            with open(refine_page_path, "r", encoding="utf-8") as file:
                page_code = file.read()
        base_prompt = ptg_prompt.format(
            page_id=page_id,
            page_code=page_code,
            PTG=PTG,
        )
        return base_prompt

    if type == "Navigation":
        with open(navigation_prompt_path, "r", encoding="utf-8") as file:
            navigation_prompt = file.read()
        print(page_list)
        return navigation_prompt.format(page_list=page_list, nav_home = homePage)


    if type == "Refine":
        with open(refine_prompt_path, "r", encoding="utf-8") as file:
            refine_prompt = file.read()
        refine_page_path = os.path.join(output_folder, app_classification, app_name, page_name + extend_name + save_type)
        with open(refine_page_path, "r", encoding="utf-8") as file:
            page_code = file.read()
        return refine_prompt.format(
            page_id=page_id,
            implemented_ids=implemented_ids,
            not_implemented_ids=not_implemented_ids,
            page_code=page_code,
        )

    if type == "Refer_refine":
        with open(refer_refine_prompt_path, "r", encoding="utf-8") as file:
            refer_refine_prompt = file.read()
        refer_info = KG_retrival(reference_path)
        return refer_refine_prompt.format(
            UItype=UItype,
            refer_info=refer_info,
            page_id=page_id,
            page_list=page_list,
            navigation_list=navigation_list,
        )

    if type == "Compile_refine":
        with open(compile_refine_prompt_path, "r", encoding="utf-8") as file:
            compile_refine_prompt = file.read()
        save_type_raw = type_detail["save_type"]

        selected_error_filename = error_filenames if isinstance(error_filenames, str) else ""
        if not selected_error_filename:
            parsed_names = extract_error_filenames(error_msg)
            if parsed_names:
                selected_error_filename = parsed_names[0]
        if not selected_error_filename:
            selected_error_filename = f"{page_name}{extend_name}.{save_type_raw}"

        error_file_path = os.path.join(output_folder, app_classification, app_name, selected_error_filename)
        if not os.path.exists(error_file_path):
            short_name = os.path.splitext(selected_error_filename)[0]
            mapped_filename = f"{short_name}{extend_name}.{save_type_raw}"
            mapped_path = os.path.join(output_folder, app_classification, app_name, mapped_filename)
            if os.path.exists(mapped_path):
                selected_error_filename = mapped_filename
                error_file_path = mapped_path

        if os.path.exists(error_file_path):
            with open(error_file_path, "r", encoding="utf-8") as file:
                error_code = file.read()
        else:
            error_code = f"Source file not found: {error_file_path}"

        return compile_refine_prompt.format(
            compile_error_log=error_msg,
            error_code=error_code,
            error_filename=selected_error_filename,
        )

    raise ValueError(f"Unsupported prompt type: {type}")


def save_response(output_folder, app_classification, app_name, pic_name, response, save_type="js"):
    save_type = get_uitype_detail()['save_type']
    # Create output folder if it does not exist
    output_folder = os.path.join(output_folder, app_classification, app_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, f'{pic_name}'+f'.{save_type}') 
 
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(response)       
        

def save_compile_refined_code_to_files(output_folder, app_classification, app_name, data):
    pattern = r'@@@(.+?)@@@([\s\S]*?)(?=@@@|$)'
    matches = re.findall(pattern, data)
    print(matches)

    page_ids = []

    for raw_page_id, code in matches:
        code = code.strip()
        raw_page_id = raw_page_id.strip().replace("\\", "/")
        raw_page_id = raw_page_id.split("/")[-1]
        page_id, ext = os.path.splitext(raw_page_id)
        ext = ext.lower()

        if ext in [".js", ".jsx", ".ts", ".tsx", ".dart", ".ets"]:
            file_extension = ext[1:]
        elif UItype == 'Flutter':
            file_extension = 'dart'
        elif UItype == 'ArkTS':
            file_extension = 'ets'
        elif UItype == 'React Native':
            file_extension = 'js'
        else:
            file_extension = 'txt'

        if page_id == 'AppScreen':
            page_id = 'App'
        page_ids.append(page_id)
        if '/' in page_id:
            page_id = page_id.replace('/', '_')
        with open(os.path.join(output_folder,app_classification,app_name,f'{page_id}.{file_extension}'), 'w', encoding='utf-8') as file:
            file.write(code)
    return page_ids


def extract_package_name(code):
    if UItype == 'Flutter':
        package_list = re.findall(r"import 'package:(\w+)/", code)
        return package_list
    elif UItype == 'React Native':
        exist_package_list =["@react-native-community/masked-view","@react-navigation/bottom-tabs","@react-navigation/native","@react-navigation/stack","jsc-android","react","react-native","react-native-gesture-handler","react-native-safe-area-context","react-native-safe-area-view","react-native-screens","@react-native-community/slider"]
        add_package_list = []
        # Analyze the code line by line to find import statements
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import'):
                if not line.endswith(';'):
                  continue
                # Analyze the import lines to extract package names
                parts = line.split(' ')
                # Determine if it is a regular import or an import from a specific package
                if 'from' in line:
                    package_name = line.split('from ')[1].replace(';', '').strip().replace("'", "").replace('"', '')
                    if not package_name.startswith('@'):
                        package_name = package_name.split('/')[0]
                    if package_name not in exist_package_list:
                        add_package_list.append(package_name)
                else:
                    package_name = parts[1].replace(';', '').strip().replace("'", "").replace('"', '')
                    if not package_name.startswith('@'):
                        package_name = package_name.split('/')[0]
                    if package_name not in exist_package_list:
                        add_package_list.append(package_name)
        return add_package_list
    
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def remove_first_and_last_lines(input_str):
    if input_str.startswith('```'):
        lines = input_str.splitlines()
        if len(lines) > 2:
            return '\n'.join(lines[1:-1])
        else:
            return ''  
    return input_str


def extract_error_filenames(error_msg):
    """
    Parse all filenames from compiler error messages.
    Supports dict errors ({filename: message}) and raw log strings.
    """
    filenames = []
    if isinstance(error_msg, dict) and error_msg:
        for key in error_msg.keys():
            if isinstance(key, str) and key.strip():
                filenames.append(os.path.basename(key.strip()))
    elif isinstance(error_msg, str) and error_msg.strip():
        file_pattern = r'([A-Za-z0-9_\-./\\]+\.(?:js|jsx|ts|tsx|dart|ets))'
        matches = re.findall(file_pattern, error_msg)
        for item in matches:
            base_name = os.path.basename(item.strip())
            if base_name:
                filenames.append(base_name)

    # Keep order while removing duplicates.
    return list(dict.fromkeys(filenames))


def get_generated_code_filenames(app_classification, app_name):
    """
    Get generated source code filenames in the output folder for current UI type.
    """
    app_output_path = os.path.join(output_folder, app_classification, app_name)
    if not os.path.exists(app_output_path):
        return set()

    save_type = get_uitype_detail()["save_type"]
    valid_suffix = f".{save_type}"
    generated_names = set()
    for file_name in os.listdir(app_output_path):
        full_path = os.path.join(app_output_path, file_name)
        if os.path.isfile(full_path) and file_name.endswith(valid_suffix):
            generated_names.add(file_name)
    return generated_names


def find_generated_error_filename(error_msg, app_classification, app_name):
    """
    Find the first error filename that is one of generated source files.
    Return '' when compile errors do not point to generated code.
    """
    generated_names = get_generated_code_filenames(app_classification, app_name)
    if not generated_names:
        return ''

    for file_name in extract_error_filenames(error_msg):
        if file_name in generated_names:
            return file_name
    return ''


def get_error_detail_by_filename(error_msg, selected_filename):
    """
    Get error detail for a selected filename (basename match for dict keys).
    """
    if isinstance(error_msg, dict) and selected_filename:
        for key, value in error_msg.items():
            if isinstance(key, str) and os.path.basename(key.strip()) == selected_filename:
                return value
    return error_msg




def extract_rn_navigate_page_mode_pairs(js_file_content):
    """
    从 RN 代码中提取最终 navigation.navigate(...) 里的 (page, mode) 对。
    例如：
    navigation.navigate('Home', { mode: 'BASE' })
    """
    if not isinstance(js_file_content, str):
        return set()

    pairs = set()
    pattern = re.compile(
        r"navigation\.navigate\(\s*['\"](?P<page>[^'\"]+)['\"]\s*,\s*\{\s*mode\s*:\s*['\"](?P<mode>[^'\"]+)['\"]\s*\}\s*\)",
        re.S,
    )
    for m in pattern.finditer(js_file_content):
        pairs.add((m.group("page").strip(), m.group("mode").strip()))
    return pairs


def inspect_RNnavigation(app_classification,app_name,page_name):
    navigation_requirements = get_navigation_requirements_with_mode(app_classification, app_name, page_name)
    type_detail = get_uitype_detail()
    extend_name = type_detail["extend_name"]
    save_type = type_detail["save_type"]

    code_path = os.path.join(output_folder,app_classification,app_name,page_name+extend_name+'.'+save_type)
    with open(code_path, 'r', encoding='utf-8') as file:
        js_file_content = file.read()

    implemented_ids = []
    not_implemented_ids = []

    # 按最终代码里的实际跳转对进行判断（targetNode.* 已被替换）
    navigate_pairs = extract_rn_navigate_page_mode_pairs(js_file_content)
    all_pairs = set(navigate_pairs)

    for req in navigation_requirements:
        target_page = str(req.get("to_page", "")).strip()
        target_mode = str(req.get("to_mode", "")).strip()
        req_label = format_navigation_requirement(req)

        pair_hit = (target_page, target_mode) in all_pairs
        implemented = pair_hit

        if implemented:  
            if req_label not in implemented_ids:
                implemented_ids.append(req_label)
        else:
            if req_label not in not_implemented_ids:
                not_implemented_ids.append(req_label)


    logging.info("Implemented IDs:%s", implemented_ids)
    logging.info("Not Implemented IDs:%s", not_implemented_ids)
    return implemented_ids, not_implemented_ids


def is_fully_navigated(res,implemented_ids,not_implemented_ids, app_classification, app_name, page_name, extend_name,client):
    navi_iter_limit = 3
    iter_cnt = 0
    while not_implemented_ids and iter_cnt < navi_iter_limit:
        logging.info("%s %s Not implemented ids: %s", app_name, page_name, not_implemented_ids)
        RefinePrompt = prompt_construct(app_classification,app_name,page_name,type="Refine",implemented_ids=implemented_ids,not_implemented_ids=not_implemented_ids)
        res = send_single_prompt_and_get_response(client,RefinePrompt)
        save_response(output_folder, app_classification, app_name, page_name+ extend_name, res)
        # recheck navigation
        implemented_ids,not_implemented_ids=inspect_RNnavigation(app_classification,app_name,page_name)
        iter_cnt += 1
    return res


def extract_stream_content(delta_content):
    if isinstance(delta_content, str):
        return delta_content
    if isinstance(delta_content, list):
        parts = []
        for item in delta_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def send_single_prompt_and_get_response(client,prompt,base64_image=''):
    image_list = []
    if isinstance(base64_image, str):
        if base64_image:
            image_list = [base64_image]
    elif isinstance(base64_image, list):
        image_list = [x for x in base64_image if isinstance(x, str) and x]

    if image_list:
        content = [{"type": "text", "text": prompt}]
        for b64 in image_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                }
            )
        messages = [
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are an engineer proficient in React Native."},
            {"role": "user", "content": prompt}
        ]

    stream = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        stream=True
    )

    stream_parts = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if not delta:
            continue

        text_piece = extract_stream_content(getattr(delta, "content", None))
        if text_piece:
            stream_parts.append(text_piece)
            print(text_piece, end="", flush=True)

    if stream_parts:
        print()

    res = "".join(stream_parts)
    res = remove_first_and_last_lines(res)
    return res


def clear_folder(folder_path):
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        # Remove all contents within the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        
        # logging.info("Successfully cleared all contents of %s.",folder_path)

    except Exception as e:
        print(f"Error occurred: {e}")
  

def copy_file2project(src_folder, dst_folder):
    if os.path.exists(dst_folder) and not os.path.isdir(dst_folder):
        raise NotADirectoryError(f"Destination exists but is not a directory: {dst_folder}")
    os.makedirs(dst_folder, exist_ok=True)
    components_folder = os.path.join(dst_folder,"components")
    os.makedirs(components_folder, exist_ok=True)
    for item in os.listdir(src_folder):
        extension = os.path.splitext(item)[1]
        if extension == '.js' or extension == '.jsx':
            if item == "App.js":
                shutil.copy(os.path.join(src_folder, item), dst_folder)
            else:
                src_path = os.path.join(src_folder, item)
                shutil.copy(src_path, components_folder)
                print(f"{item} copied!") 


def send_mutiple_prompts_and_get_response(app_classification, app_name):
    client = OpenAI(
        base_url="http://47.76.75.25:9000/v1",
        api_key="sk-IuB1RyVtAz6G5xHq0fB8Fb7834F64eB4846b84C96a0a4bCf", 
        http_client=httpx.Client(
            base_url="http://47.76.75.25:9000/v1",
            follow_redirects=True,
            timeout=300.0,
            trust_env=False,
        ),
    )
    entry_func_name = get_uitype_detail()["entry_func_name"]
    extend_name = get_uitype_detail()["extend_name"]

    # 关键改动：按“聚类页面”生成，不再按每张截图生成
    cluster_pages = get_cluster_page_list(app_classification, app_name)
    if not cluster_pages:
        raise RuntimeError(f"No clustered pages found for {app_classification}/{app_name}.")

    home_page_map = {}
    for p in cluster_pages:
        home_page_map[p] = extract_home_screen_from_psg(app_classification, app_name, p)

    # 入口页：使用 APP 的 utg_component.json 第一个节点的 page 属性
    entry_page_name = get_entry_page_from_utg_component(app_classification, app_name)
    
    add_package_list = []
    current_page_name = entry_page_name

    # for page_name in cluster_pages:
    #     current_page_name = page_name
    #     home_page = home_page_map.get(page_name, page_name)

    #     # Step 1: 基于分类 + PSG 生成页面代码
    #     psg_prompt = prompt_construct(
    #         app_classification,
    #         app_name,
    #         page_name,
    #         home_page,
    #         type="ScreenPSG",
    #     )
    #     graph_base64_images = collect_graph_base64_images_for_page(app_classification, app_name, page_name, type="PSG")
    #     res = send_single_prompt_and_get_response(client, psg_prompt, graph_base64_images)
    #     print(f"----------{page_name} PSG code:----------\n{res}")
    #     save_response(output_folder, app_classification, app_name, page_name + extend_name, res)
    #     logging.info(f"----------{page_name} PSG code:----------\n{res}")

    #     # Step 2: 基于 PTG prompt + PTG 图，在已有页面代码上补充页面跳转
    #     ptg_prompt = prompt_construct(
    #         app_classification,
    #         app_name,
    #         page_name,
    #         home_page,
    #         type="PTGRefine",
    #     )
    #     graph_base64_images = collect_graph_base64_images_for_page(app_classification, app_name, page_name, type="PTG")
    #     res = send_single_prompt_and_get_response(client, ptg_prompt, graph_base64_images)
    #     print(f"----------{page_name} PTG refined code:----------\n{res}")
    #     save_response(output_folder, app_classification, app_name, page_name + extend_name, res)
    #     logging.info(f"----------{page_name} PTG refined code:----------\n{res}")

    #     # 导航检查与补全
    #     implemented_ids, not_implemented_ids = inspect_RNnavigation(app_classification, app_name, page_name)
    #     logging.info("%s Implemented ids: %s", page_name, implemented_ids)
    #     res = is_fully_navigated(
    #         res,
    #         implemented_ids,
    #         not_implemented_ids,
    #         app_classification,
    #         app_name,
    #         page_name,
    #         extend_name,
    #         client=client,
    #     )

    #     if UItype == "React Native":
    #         add_package_set = set(add_package_list)
    #         add_package_set = add_package_set.union(set(extract_package_name(res)))
    #         add_package_list = list(add_package_set)

    #     time.sleep(2)
    #     print("Succesfully saved the response")

    # Step 3: 用 navigation prompt 生成页面注册/入口文件
    print("Generating application entry.")
    nav_home = entry_page_name
    print(nav_home)
    NavigationPrompt = prompt_construct(
        app_classification,
        app_name,
        entry_page_name,
        nav_home,
        type="Navigation",
    )
    res = send_single_prompt_and_get_response(client, NavigationPrompt)
    logging.info(f'=========App.js code:=========\n{res}\n')
    save_response(output_folder, app_classification, app_name, entry_func_name, res)
    compile_state = False
    loop_count = 0
    error_msg = ''
    # Clean the project files in preparation for compilation
    clear_folder(os.path.join(project_template_folder,"components"))
    # Copy the generated code to the project folder
    copy_file2project(os.path.join(output_folder,app_classification,app_name),os.path.join(project_template_folder))
    print(f'Add Package List: {add_package_list}')
    compile_state,error_msg = run_gradlew_assemble(project_template_folder,app_classification,app_name,add_package_list)
    print(f"error_msg: {error_msg}")
     # compile iteration
    while not compile_state and loop_count < 3:
        # fix one file every time
        print(f"---------------{loop_count+1}fix---------------")
        selected_error_filename = find_generated_error_filename(error_msg, app_classification, app_name)
        if not selected_error_filename:
            print("No generated code filename found in compile errors. This is likely an internal project compile error.")
            logging.warning("Internal project compile error (not generated code): %s", error_msg)
            break

        selected_error_msg = get_error_detail_by_filename(error_msg, selected_error_filename)
        CompileRefinePrompt = prompt_construct(
            app_classification,
            app_name,
            entry_page_name,
            nav_home,
            type="Compile_refine",
            error_msg=selected_error_msg,
            error_filenames=selected_error_filename
        )
        res = send_single_prompt_and_get_response(client,CompileRefinePrompt)
        logging.info(f'{selected_error_msg} code:\n{res}\n')
        print(selected_error_msg)
        print(res)
        refined_list = save_compile_refined_code_to_files(output_folder, app_classification, app_name, res)
        print(f'refined_list:{refined_list}')
        copy_file2project(os.path.join(output_folder,app_classification,app_name),os.path.join(project_template_folder))
        compile_state,error_msg = run_gradlew_assemble(project_template_folder,app_classification,app_name,[])
        loop_count += 1
    return compile_state,loop_count,error_msg
        
        
def generate_by_classification(app_classification):
      
    complete_img_dir = os.path.join(picture_folder, app_classification)
        
    for app_name in os.listdir(complete_img_dir):
        start_time = time.time()
        print("Processing %s...",app_name)
        print(f"Processing {app_name}...")
        compile_state,loop_cnt,error_msg = send_mutiple_prompts_and_get_response(app_classification,app_name)
        end_time = time.time()
        print("time cost:",end_time-start_time)
        print("time cost: %s",end_time-start_time)

        print("%s processed.",app_name)


def generate_by_app(app_classification,app_name):
    print("Start processing %s...",app_name)
    start_time = time.time()
    compile_state,loop_cnt,error_msg = send_mutiple_prompts_and_get_response(app_classification,app_name)
    end_time = time.time()
    print("%s processed.",app_name)
    
    
if __name__ == "__main__":
    # app_classification_list = ["nordicusability","Hashmusic","Nor","Social Networking","Entertainment","Productivity","Education","Wellness","Finance","Shopping&E-commerce","Travel&Navigation","News&Information","Utility&PracticalTools"]
    app_classification_list = ["com"]
    app_classification = app_classification_list[0]
    app_name = 'jiffy'
    generate_by_app(app_classification, app_name)

    # generate_by_classification(app_classification_list[0])
   
