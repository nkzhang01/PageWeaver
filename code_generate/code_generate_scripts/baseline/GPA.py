import json
from openai import OpenAI
import shutil
import httpx
import base64
import re
import os
import sys
import time
import logging
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
CODE_GENERATE_DIR = os.path.join(PROJECT_ROOT, "code_generate")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from code_generate.utils.compiler.RN_compiler import run_gradlew_assemble

lang_index = 0
UItype_list = ["React Native"]
UItype = UItype_list[lang_index]
project_template_folder_list = [r""]
project_template_folder = project_template_folder_list[lang_index]
if project_template_folder and not os.path.isabs(project_template_folder):
    project_template_folder = os.path.join(SCRIPT_DIR, project_template_folder)
picture_folder = os.path.join(PROJECT_ROOT, "dataset")
prompt_templates_root = os.path.join(CODE_GENERATE_DIR, "utils", "prompt_templates")
prompt_folder = os.path.join(prompt_templates_root, "Baselines-GPA")

output_folder = os.path.join(PROJECT_ROOT, "results", "code_results", "baselines", "GPA")
STATE_PATH = os.path.join(output_folder, "state.json")


API_BASE = ""
MODEL_NAME = ""
API_KEY = ""


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
   
    save_type = "js"
    other_type = "txt"
    entry_func_name = "App"
    extend_name = "Screen"
    nav_code_template = ["navigation.navigate('{nav_id}'","'{nav_id}'"]
    
    type_detail["save_type"] = save_type
    type_detail["other_type"] = other_type
    type_detail["entry_func_name"] = entry_func_name
    type_detail["extend_name"] = extend_name
    type_detail['nav_code_template'] = nav_code_template

    return type_detail

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state):
    os.makedirs(output_folder, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def get_app_state(state, app_classification, app_name):
    return state.get(app_classification, {}).get(app_name, 0)

def set_app_state(state, app_classification, app_name, value):
    if app_classification not in state:
        state[app_classification] = {}
    state[app_classification][app_name] = value
    save_state(state)

def create_openai_client():
    api_key = API_KEY.strip()
    if not api_key:
        raise RuntimeError("API_KEY is empty. Please fill your model API key in API_KEY.")

    return OpenAI(
        base_url=API_BASE,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=API_BASE,
            follow_redirects=True,
            timeout=300.0,
            trust_env=False,
        ),
    )

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
    Read the clustered page list, rather than individual screenshots, from page_subgraphs/state/*.json.
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
            out.append(
                {
                    "name": os.path.basename(p),
                    "b64": encode_image(p),
                }
            )
        except Exception:
            continue
    return out

def get_app_ptg_path(app_classification, app_name):
    path = os.path.join(
        picture_folder,
        app_classification,
        app_name,
        "page_subgraphs",
        "app_transition_graph.json",
    )
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"App PTG not found: {path}")

def load_app_ptg(app_classification, app_name):
    return load_json_file(get_app_ptg_path(app_classification, app_name))

def collect_app_ptg_base64_images(app_classification, app_name):
    app_ptg = load_app_ptg(app_classification, app_name)
    n_key, _ = detect_json_schema(app_ptg)
    if not n_key:
        return []

    image_items = []
    seen_names = set()

    for screen_id in collect_screen_ids_from_graph(get_app_ptg_path(app_classification, app_name)):
        image_path = resolve_image_path_for_screen(app_classification, app_name, screen_id)
        if not image_path:
            continue
        image_name = os.path.basename(image_path)
        if image_name in seen_names:
            continue
        seen_names.add(image_name)
        try:
            image_items.append(
                {
                    "name": image_name,
                    "b64": encode_image(image_path),
                }
            )
        except Exception:
            continue

    return image_items

def build_all_page_code(app_classification, app_name, page_names):
    type_detail = get_uitype_detail()
    extend_name = type_detail["extend_name"]
    save_type = "." + type_detail["save_type"]
    page_code_blocks = []

    for page_name in page_names:
        file_name = page_name + extend_name + save_type
        candidates = [
            os.path.join(output_folder, app_classification, app_name, "static", file_name),
            os.path.join(output_folder, app_classification, app_name, file_name),
        ]
        file_path = next((path for path in candidates if os.path.exists(path)), "")
        if not file_path:
            logging.warning("Page code not found for PTG input: %s", file_name)
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            page_code = file.read()
        page_code_blocks.append(f"@@@{file_name}@@@\n{page_code}")

    return "\n".join(page_code_blocks)

def get_entry_page_from_utg_component(app_classification, app_name):
    """
    Entry page is defined directly by the first UTG node's `id`.
    """
    utg_component_path = os.path.join(picture_folder, app_classification, app_name, "UTG.json")
    data = load_json_file(utg_component_path)
    n_key, _ = detect_json_schema(data)
    nodes = data.get(n_key, []) if n_key else []
    if not nodes:
        return ""

    return str(nodes[0].get("id", "")).strip()

def get_page_list(app_classification, app_name):
    raw_pages = get_cluster_page_list(app_classification, app_name)

    return raw_pages

def safe_prompt_format(template, **kwargs):
    prompt = template
    for key, value in kwargs.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt

def prompt_construct(
    app_classification,
    app_name,
    page_name,
    type="ScreenPSG",
    error_msg="",
    error_filenames="",
):
    page_id = page_name
    state_path = os.path.join(
        picture_folder, app_classification, app_name, "page_subgraphs", "state", page_name + ".json"
    )
    PTG_prompt_path = os.path.join(prompt_folder, "PTGPrompt.txt")
    PSG_prompt_path = os.path.join(prompt_folder, "PSGPrompt.txt")
    compile_refine_prompt_path = os.path.join(prompt_templates_root, "RefineCompileErrorPrompt.txt")

    page_list = get_page_list(app_classification, app_name)
    type_detail = get_uitype_detail()
    extend_name = type_detail["extend_name"]

    PSG = {}
    if os.path.exists(state_path):
        PSG = load_json_file(state_path)

    if type == "ScreenPSG":
        with open(PSG_prompt_path, "r", encoding="utf-8") as file:
            psg_prompt = file.read()
        return psg_prompt.format(
            page_id=page_id,
            PSG=json.dumps(PSG, ensure_ascii=False, indent=2),
        )

    if type == "ScreenPTG":
        with open(PTG_prompt_path, "r", encoding="utf-8") as file:
            ptg_prompt = file.read()
        all_page_code = build_all_page_code(app_classification, app_name, page_list)
        app_ptg = load_app_ptg(app_classification, app_name)
        base_prompt = safe_prompt_format(
            ptg_prompt,
            page_code=all_page_code,
            PTG=json.dumps(app_ptg, ensure_ascii=False, indent=2),
        )
        return base_prompt

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
    output_folder = os.path.join(output_folder, app_classification, app_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    pic_name = str(pic_name)
    expected_suffix = f".{save_type}"
    if os.path.splitext(pic_name)[1]:
        output_filename = pic_name
    else:
        output_filename = f"{pic_name}{expected_suffix}"

    output_path = os.path.join(output_folder, output_filename) 
 
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(response)       
        

def save_generated_code_to_files(output_folder, app_classification, app_name, data):
    pattern = r'@@@(.+?)@@@([\s\S]*?)(?=@@@|$)'
    matches = re.findall(pattern, data)
    print(matches)
    if not matches:
        raise ValueError("No code files found in PTG output. Expected blocks like @@@App.js@@@.")

    saved_filenames = []
    output_dir = os.path.join(output_folder, app_classification, app_name)
    os.makedirs(output_dir, exist_ok=True)

    for raw_page_id, code in matches:
        code = code.strip()
        raw_page_id = raw_page_id.strip().replace("\\", "/")
        raw_page_id = raw_page_id.split("/")[-1]
        page_id, ext = os.path.splitext(raw_page_id)
        ext = ext.lower()

        if ext in [".js", ".jsx", ".ts", ".tsx", ".dart", ".ets"]:
            file_extension = ext[1:]
        elif UItype == 'React Native':
            file_extension = 'js'
        else:
            file_extension = 'txt'

        if page_id == 'AppScreen':
            page_id = 'App'
        if '/' in page_id:
            page_id = page_id.replace('/', '_')
        file_name = f'{page_id}.{file_extension}'
        saved_filenames.append(file_name)
        with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as file:
            file.write(code)
    return saved_filenames

def save_compile_refined_code_to_files(output_folder, app_classification, app_name, data):
    return save_generated_code_to_files(output_folder, app_classification, app_name, data)

def extract_package_name(code):
    exist_package_list =["@react-navigation/native","@react-navigation/stack","@react-navigation/native-stack","react","react-native","react-native-gesture-handler","react-native-safe-area-context","react-native-screens","@react-native-community/slider"]
    add_package_list = []

    def normalize_import_package(package_name):
        package_name = package_name.strip().replace("'", "").replace('"', "")
        if not package_name:
            return None
        if package_name.startswith((".", "/", "\\")):
            return None
        if package_name in {".", ".."}:
            return None

        if package_name.startswith('@'):
            parts = package_name.split('/')
            if len(parts) < 2:
                return None
            package_name = '/'.join(parts[:2])
        else:
            package_name = package_name.split('/')[0]

        if not re.fullmatch(r"(?:@[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+|[a-zA-Z0-9._-]+)", package_name):
            return None
        return package_name

    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('import'):
            if not line.endswith(';'):
                continue
            parts = line.split(' ')
            if 'from' in line:
                package_name = normalize_import_package(line.split('from ')[1].replace(';', '').strip())
                if package_name not in exist_package_list:
                    if package_name:
                        add_package_list.append(package_name)
            else:
                package_name = normalize_import_package(parts[1].replace(';', '').strip())
                if package_name not in exist_package_list:
                    if package_name:
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

def basename_from_log_path(path_text):
    return os.path.basename(str(path_text).strip().replace("\\", "/"))

def extract_error_filenames(error_msg):
    filenames = []
    if isinstance(error_msg, dict) and error_msg:
        for key in error_msg.keys():
            if isinstance(key, str) and key.strip():
                filenames.append(basename_from_log_path(key))
    elif isinstance(error_msg, str) and error_msg.strip():
        syntax_location_pattern = r'(?:SyntaxError|error|Error):\s+([A-Za-z]:[^\n:]+?\.(?:js|jsx|ts|tsx|dart|ets))\s*:'
        for item in re.findall(syntax_location_pattern, error_msg):
            base_name = basename_from_log_path(item)
            if base_name:
                filenames.append(base_name)

        file_pattern = r'([A-Za-z0-9_\-./\\]+\.(?:js|jsx|ts|tsx|dart|ets))'
        matches = re.findall(file_pattern, error_msg)
        for item in matches:
            base_name = basename_from_log_path(item)
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

def build_generated_project_res_text(app_classification, app_name, file_names=None):
    app_output_path = os.path.join(output_folder, app_classification, app_name)
    if file_names is None:
        file_names = sorted(get_generated_code_filenames(app_classification, app_name))
    else:
        file_names = sorted(file_names, key=lambda name: (name != "App.js", name))

    blocks = []
    for file_name in file_names:
        file_path = os.path.join(app_output_path, file_name)
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as file:
            code = file.read()
        blocks.append(f"@@@{file_name}@@@\n{code}")
    return "\n".join(blocks)

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
            if isinstance(key, str) and basename_from_log_path(key) == selected_filename:
                return value
    return error_msg

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
    if isinstance(base64_image, list):
        for item in base64_image:
            if not isinstance(item, dict):
                raise ValueError("Image input must be a list of dicts with 'name' and 'b64'.")
            name = str(item.get("name", "")).strip()
            b64 = item.get("b64")
            if not name or not isinstance(b64, str) or not b64:
                raise ValueError("Each image item must include non-empty 'name' and 'b64'.")
            image_list.append({"name": name, "b64": b64})
    elif base64_image:
        raise ValueError("Image input must be a list of dicts with 'name' and 'b64'.")

    if image_list:
        content = [{"type": "text", "text": prompt}]
        for image_item in image_list:
            content.append(
                {
                    "type": "text",
                    "text": f"screenshot name: {image_item['name']}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_item['b64']}"
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

    stream_parts = []
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True
    )

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
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

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
  

def copy_file2project(src_folder, dst_folder, allowed_filenames=None):
    if os.path.exists(dst_folder) and not os.path.isdir(dst_folder):
        raise NotADirectoryError(f"Destination exists but is not a directory: {dst_folder}")
    os.makedirs(dst_folder, exist_ok=True)
    components_folder = os.path.join(dst_folder,"components")
    os.makedirs(components_folder, exist_ok=True)
    allowed_set = set(allowed_filenames or [])
    for item in os.listdir(src_folder):
        if allowed_set and item not in allowed_set:
            continue
        extension = os.path.splitext(item)[1]
        if extension == '.js' or extension == '.jsx':
            if item == "App.js":
                shutil.copy(os.path.join(src_folder, item), dst_folder)
            else:
                src_path = os.path.join(src_folder, item)
                shutil.copy(src_path, components_folder)
                print(f"{item} copied!") 

def send_mutiple_prompts_and_get_response(app_classification, app_name):
    client = create_openai_client()
    extend_name = get_uitype_detail()["extend_name"]
    clear_folder(os.path.join(output_folder, app_classification, app_name))
    
    cluster_pages = get_cluster_page_list(app_classification, app_name)
    if not cluster_pages:
        raise RuntimeError(f"No clustered pages found for {app_classification}/{app_name}.")

    entry_page_name = get_entry_page_from_utg_component(app_classification, app_name)
    
    add_package_list = []

    # Step 1: PSG 
    for page_name in cluster_pages:
        psg_prompt = prompt_construct(
            app_classification,
            app_name,
            page_name,
            type="ScreenPSG",
        )
        graph_base64_images = collect_graph_base64_images_for_page(app_classification, app_name, page_name, type="PSG")
        res = send_single_prompt_and_get_response(client, psg_prompt, graph_base64_images)
        print(f"----------{page_name} PSG code:----------\n{res}")
        save_response(output_folder, app_classification, app_name, page_name + extend_name, res)
        save_response(output_folder, app_classification, app_name + "/static", page_name + extend_name, res)
        logging.info(f"----------{page_name} PSG code:----------\n{res}")
        add_package_set = set(add_package_list)
        add_package_set = add_package_set.union(set(extract_package_name(res)))
        add_package_list = list(add_package_set)
        time.sleep(2)
        print("Succesfully saved the PSG response")

    # Step 2: PTG
    ptg_prompt = prompt_construct(
        app_classification,
        app_name,
        entry_page_name,
        type="ScreenPTG",
    )
    graph_base64_images = collect_app_ptg_base64_images(app_classification, app_name)
    ptg_res = send_single_prompt_and_get_response(client, ptg_prompt, graph_base64_images)
    logging.info(f"----------PTG project code:----------\n{ptg_res}")
    generated_filenames = save_generated_code_to_files(output_folder, app_classification, app_name, ptg_res)

    add_package_set = set(add_package_list)
    add_package_set = add_package_set.union(set(extract_package_name(ptg_res)))
    add_package_list = list(add_package_set)

    final_generated_filenames = sorted(get_generated_code_filenames(app_classification, app_name))
    res_before_text = build_generated_project_res_text(app_classification, app_name, final_generated_filenames)
    save_response(
        output_folder,
        app_classification,
        os.path.join(app_name, "ptg_response"),
        "res_before",
        res_before_text,
        save_type="txt",
    )
    print(f"Succesfully saved PTG project files: {final_generated_filenames}")

    compile_state = False
    loop_count = 0
    error_msg = ''
    # Clean the project files in preparation for compilation
    clear_folder(os.path.join(project_template_folder,"components"))
    # Copy the generated code to the project folder
    copy_file2project(
        os.path.join(output_folder, app_classification, app_name),
        os.path.join(project_template_folder),
        final_generated_filenames,
    )
    print(f'Add Package List: {add_package_list}')
    compile_state,error_msg = run_gradlew_assemble(project_template_folder,app_classification,app_name,add_package_list)
    print(f"error_msg: {error_msg}")

    while not compile_state and loop_count < 3:
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
            type="Compile_refine",
            error_msg=selected_error_msg,
            error_filenames=selected_error_filename
        )
        res = send_single_prompt_and_get_response(client,CompileRefinePrompt)
        save_response(output_folder, app_classification, app_name + "/compile", f"{loop_count}_{selected_error_filename}", res)
        logging.info(f'{selected_error_msg} code:\n{res}\n')
        print(selected_error_msg)
        print(res)
        refined_list = save_compile_refined_code_to_files(output_folder, app_classification, app_name, res)
        print(f'refined_list:{refined_list}')
        generated_filenames = list(dict.fromkeys(generated_filenames + refined_list))
        copy_file2project(
            os.path.join(output_folder, app_classification, app_name),
            os.path.join(project_template_folder),
            generated_filenames,
        )
        compile_state,error_msg = run_gradlew_assemble(project_template_folder,app_classification,app_name,add_package_list)
        loop_count += 1

    final_generated_filenames = sorted(get_generated_code_filenames(app_classification, app_name))
    final_res_text = build_generated_project_res_text(app_classification, app_name, final_generated_filenames)
    save_response(output_folder, app_classification, app_name, "res", final_res_text, save_type="txt")
    return compile_state,loop_count,error_msg
        
        
def generate_by_classification(app_classification):
      
    complete_img_dir = os.path.join(picture_folder, app_classification)
    state = load_state()
        
    for app_name in os.listdir(complete_img_dir):

        if get_app_state(state, app_classification, app_name) == 1:
            print(f"[SKIP] {app_classification}/{app_name}: already completed")
            continue

        if app_classification not in state or app_name not in state[app_classification]:
            set_app_state(state, app_classification, app_name, 0)

        start_time = time.time()
        print(f"Processing {app_name}...")
        try:
            compile_state,loop_cnt,error_msg = send_mutiple_prompts_and_get_response(app_classification,app_name)
            set_app_state(state, app_classification, app_name, 1)
            end_time = time.time()
            print("time cost:",end_time-start_time)
            print(f"{app_name} processed.")
        except Exception as error:
            end_time = time.time()
            print(f"[FAIL] {app_classification}/{app_name}: {error}")
            print(traceback.format_exc())
            logging.exception("[FAIL] %s/%s", app_classification, app_name)
            print("time cost:",end_time-start_time)
            continue

if __name__ == "__main__":
    app_classification_lists = [
        "AI",
        "AppStore",
        "Browser",
        "Email",
        "Life",
        "Music",
        "Shopping",
        "Social",
        "Study",
        "Travel",
        "Vedio",
        "Work"
    ]
    for app_classification in app_classification_lists:
        generate_by_classification(app_classification)
   
