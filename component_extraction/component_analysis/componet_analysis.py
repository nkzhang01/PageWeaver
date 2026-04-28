import json
import math

from openai import OpenAI
import subprocess
import shutil
import httpx
import base64
import re
import os
import openai
from PIL import Image

# from similarity_search import similarity_search
os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
os.environ["OPENAI_API_KEY"] = "sk-SevA9Kr9w4rY7cxQB0E64eBc7aF442E098D1C6Bc31F09373"  #4

openai.api_key = os.getenv("OPENAI_API_KEY")



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def split2type(
    image_dir,
    split_img_base_dir,
    category,
    app_name,
    split_node_dir_name,
    complete_image_name,
    content,
    position_txt_content,
):
    IMG_LIMIT = 10
    image_files = []

    # complete_img_path = os.path.join(image_dir, "complete_imgs", category, app_name, image_name)
    complete_img_path = os.path.join(image_dir, category, app_name, complete_image_name)
    image_files.append(encode_image(complete_img_path))
    # split_img_dir = os.path.join(image_dir, "split_imgs",category, app_name, image_name.split(".")[0])
    split_img_dir = os.path.join(split_img_base_dir, category, app_name, split_node_dir_name)

    files = os.listdir(split_img_dir)

    png_files = [f for f in files if f.endswith('.png')]

    sorted_files = sorted(png_files, key=lambda x: int(os.path.splitext(x)[0]))
    print(sorted_files)
    for file in sorted_files:
        image_files.append(encode_image(os.path.join(split_img_dir, file)))

    client = OpenAI(
        base_url="https://api.xty.app/v1",
        api_key="sk-Xs7tLhUmFpTQ2bRYB1F6F93e81F848899e6eBbC17cF7C48c",
        http_client=httpx.Client(
            base_url="https://api.xty.app/v1",
            follow_redirects=True,
        ),
    )

    # complete_img
    complete_img_dic = img_content_template = {
            "type":"image_url",
            "image_url":{
            "url": f"data:image/png;base64,{image_files[0]}"
            }
        }
    # add complete screenshot
    img_dic_list =[]
    # add split component
    for base64_img_before in image_files[1:]:
        pic_type = "png"
        img_content_template = {
            "type":"image_url",
            "image_url":{
            "url": f"data:image/{pic_type};base64,{base64_img_before}"
            }
        }
        img_dic_list.append(img_content_template)
    img_cnt = len(img_dic_list)
    batch_size = IMG_LIMIT - 1
    # 使用向上取整计算 batch 次数
    query_times = math.ceil(img_cnt / batch_size)

    all_positions_list = [line.strip() for line in position_txt_content.strip().split('\n') if line.strip()]

    left = 0

    response = ""
    for i in range(int(query_times)):
        right = min(left + batch_size, img_cnt)  # 保证最后一个 batch 不越界

        current_batch_positions_list = all_positions_list[left:right]
        current_batch_positions_str = "\n".join(current_batch_positions_list)
        print(current_batch_positions_list)
        current_prompt = content.replace("{position_txt}", current_batch_positions_str)
        text_dic = {
            "type": "text",
            "text": current_prompt
        }

        content_list = []
        content_list.append(text_dic)
        content_list.append(complete_img_dic)
        content_list = content_list + img_dic_list[left:right]
        completion = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "user",
                 "content": content_list
                 },
            ],
        )
        print(completion.choices[0].message.content)
        response = response + completion.choices[0].message.content + '\n'
        # 更新 left
        left = right  # 下一个 batch 从 right 开始
    print(response)
    return response


def extract_component_type(s):
    pattern = r'@@@(.*?)@@@'
    matches = re.findall(pattern, s)
    unique_matches = list(set(matches))  
    return unique_matches


def image_name_sort_key(name):
    stem = os.path.splitext(name)[0]
    match = re.search(r'(\d+)$', stem)
    if match:
        return 0, int(match.group(1)), stem
    match = re.search(r'(\d+)', stem)
    if match:
        return 1, int(match.group(1)), stem
    return 2, stem


def load_page_type_first_nodes(app_img_dir):
    intra_page_dir = os.path.join(app_img_dir, "intra_page")
    if not os.path.exists(intra_page_dir) or not os.path.isdir(intra_page_dir):
        return {}

    page_type_to_first_node = {}
    for file_name in os.listdir(intra_page_dir):
        if not file_name.endswith(".json"):
            continue
        if file_name.endswith("_transition.json"):
            continue

        page_type_id = os.path.splitext(file_name)[0]
        json_path = os.path.join(intra_page_dir, file_name)

        try:
            with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            continue

        nodes = []
        if isinstance(data.get("N"), list):
            nodes = data.get("N", [])
        elif isinstance(data.get("nodes"), list):
            nodes = data.get("nodes", [])

        if not nodes:
            continue

        first_node = nodes[0] if isinstance(nodes[0], dict) else {}
        first_node_id = str(first_node.get("id", "")).strip()
        if first_node_id:
            page_type_to_first_node[page_type_id] = first_node_id

    return page_type_to_first_node


def resolve_complete_image_name(app_img_dir, node_id):
    node_id = str(node_id or "").strip()
    if not node_id:
        return ""

    candidates = []
    candidates.append(node_id)
    base_name = os.path.basename(node_id)
    if base_name and base_name not in candidates:
        candidates.append(base_name)

    stem, ext = os.path.splitext(base_name)
    if stem:
        if not ext:
            candidates.extend([f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"])
        else:
            candidates.extend([f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"])

    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(app_img_dir, name)
        if os.path.exists(p) and os.path.isfile(p):
            return name
    return ""


import os
import json


def normalize_position_coordinates(position_file, output_file, image_width, image_height):
    """
    将绝对坐标位置文件转换为归一化坐标

    参数:
        position_file: 输入的绝对坐标文件路径
        output_file: 输出的归一化坐标文件路径
        image_width: 图像宽度
        image_height: 图像高度
    """
    try:
        # 读取绝对坐标
        with open(position_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 解析坐标并转换
        normalized_data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                # 解析坐标，支持多种格式
                parts = line.split()
                if len(parts) >= 4:
                    # 提取坐标值
                    coords = []
                    for part in parts[:4]:
                        try:
                            # 尝试转换为浮点数
                            coord = float(part)
                            coords.append(coord)
                        except ValueError:
                            # 如果无法转换，跳过这一行
                            print(f"警告: 第{i + 1}行包含非数字坐标: {line}")
                            coords = None
                            break

                    if coords and len(coords) == 4:
                        x1, y1, x2, y2 = coords

                        # 转换为归一化坐标
                        x1_norm = x1 / image_width
                        y1_norm = y1 / image_height
                        x2_norm = x2 / image_width
                        y2_norm = y2 / image_height

                        # 确保坐标在合理范围内
                        x1_norm = max(0.0, min(1.0, x1_norm))
                        y1_norm = max(0.0, min(1.0, y1_norm))
                        x2_norm = max(0.0, min(1.0, x2_norm))
                        y2_norm = max(0.0, min(1.0, y2_norm))

                        # 保留其他信息（如标签）
                        other_info = ' '.join(parts[4:]) if len(parts) > 4 else ''

                        # 格式化为6位小数
                        normalized_line = f"{x1_norm:.6f},{y1_norm:.6f},{x2_norm:.6f},{y2_norm:.6f}"
                        if other_info:
                            normalized_line += f" {other_info}"

                        normalized_data.append(normalized_line)
                    else:
                        print(f"警告: 第{i + 1}行坐标格式错误: {line}")
                else:
                    print(f"警告: 第{i + 1}行坐标数量不足: {line}")

            except Exception as e:
                print(f"处理第{i + 1}行时出错: {line} - {e}")
                continue

        # 写入归一化坐标文件
        if normalized_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in normalized_data:
                    f.write(line + '\n')

            print(f"归一化坐标已保存到: {output_file}")
            return True
        else:
            print(f"警告: 未找到有效的坐标数据")
            return False

    except Exception as e:
        print(f"处理文件时出错: {position_file} - {e}")
        return False

def copy_file(src, dst):
    try:
        if not os.path.exists(src):
            print(f"Source file '{src}' does not exist.")
            return

        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            print(f"Created directory '{dst_dir}'.")

        shutil.copy2(src, dst)  # Windows & Linux 通用
        print(f"Successfully copied '{src}' to '{dst}'.")
    except Exception as e:
        print(f"Error occurred while copying file: {e}")


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
        
        print(f"Successfully cleared all contents of '{folder_path}'.")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lang_segment_anything_dir = os.path.join(project_root, "component_extraction", "lang-segment-anything")

    component_info_folder = os.path.join(script_dir, "results-test", "extract_results")
    framework = 'RN'
    language_list = ["RN","Flutter", "ArkTS"]
    lang = language_list[0]
    category_list = ["com"]
    # App 处理范围：
    # - 仅处理 jiffy：TARGET_APPS = {"jiffy"}
    # - 批处理当前 category 下全部 App：TARGET_APPS = None
    TARGET_APPS = ["jiffy"]
    image_dir = os.path.join(project_root, "data")
    initial_component_info_dir = os.path.join(component_info_folder, lang)
    initial_split_img_dir = os.path.join(lang_segment_anything_dir, "split_img")
    initial_complete_img_dir = image_dir
    prompt_dir = os.path.join(script_dir, "prompt_repo")

    # 读取 prompt
    if lang == "RN":
        prompt_path = os.path.join(prompt_dir, "RN_Split2TypeInfoPrompt.txt")
    elif lang == "ArkTS":
        prompt_path = os.path.join(prompt_dir, "ArkTS_Split2TypePrompt.txt")
    elif lang == "Flutter":
        prompt_path = os.path.join(prompt_dir, "Flutter_Split2TypePrompt.txt")
    with open(prompt_path, "r", encoding="utf-8", errors="ignore") as f:
        split2type_prompt_template = f.read()

    # 读取或初始化 state_file
    if os.path.exists('state.json'):
        with open('state.json', 'r', encoding='utf-8') as f:
            state_file = json.load(f)
    else:
        state_file = {}

    for category in category_list:
        print(f"\nProcessing category: {category}")

        complete_img_dir = os.path.join(initial_complete_img_dir, category)
        split_img_dir = os.path.join(initial_split_img_dir, category)
        component_info_dir = os.path.join(initial_component_info_dir, category)

        # 遍历每个 app
        for app in os.listdir(complete_img_dir):
            if TARGET_APPS is not None and app not in TARGET_APPS:
                continue

            app_img_dir = os.path.join(complete_img_dir, app)
            app_component_info_dir = os.path.join(component_info_dir, app)
            app_split_img_dir = os.path.join(split_img_dir, app)

            if not os.path.exists(app_component_info_dir):
                os.makedirs(app_component_info_dir)

            # 初始化 state_file[app]，保证 KeyError 不会出现
            if app not in state_file:
                state_file[app] = {}
            page_type_to_first_node = load_page_type_first_nodes(app_img_dir)
            if not page_type_to_first_node:
                print(f"Skip {app}: intra_page/*.json missing or empty.")
                continue

            page_type_ids = sorted(
                page_type_to_first_node.keys(),
                key=lambda page_type_id: image_name_sort_key(page_type_to_first_node[page_type_id])
            )

            # 初始化每个分类（page_type）的处理状态
            for page_type_id in page_type_ids:
                if page_type_id not in state_file[app]:
                    state_file[app][page_type_id] = 0

            # 每个分类只处理一次；完整图使用该分类第一个节点
            for page_type_id in page_type_ids:
                if state_file[app][page_type_id] == 1:
                    continue  # 已处理

                first_node_id = page_type_to_first_node[page_type_id]
                complete_image_name = resolve_complete_image_name(app_img_dir, first_node_id)
                print(f"Processing {app} {page_type_id} (first node id: {first_node_id})")

                position_before = os.path.join(app_split_img_dir, page_type_id, "position.txt")
                position_after = os.path.join(app_component_info_dir, page_type_id + "_position.txt")

                if not complete_image_name:
                    print(f"Skip {app} {page_type_id}: complete image not found from first node id - {first_node_id}")
                    continue

                image_path = os.path.join(app_img_dir, complete_image_name)
                if not os.path.exists(image_path):
                    print(f"Skip {app} {page_type_id}: complete image not found - {image_path}")
                    continue

                with Image.open(image_path) as img:
                    image_width, image_height = img.size

                normalized_ok = normalize_position_coordinates(
                    position_file=position_before,
                    output_file=position_after,
                    image_width=image_width,
                    image_height=image_height
                )
                if not normalized_ok or not os.path.exists(position_after):
                    print(f"Skip {app} {page_type_id}: position normalization failed.")
                    continue

                with open(position_after, 'r', encoding='utf-8') as f:
                    position_txt_content = f.read()

                component_type = split2type(
                    image_dir,
                    initial_split_img_dir,
                    category,
                    app,
                    page_type_id,
                    complete_image_name,
                    split2type_prompt_template,
                    position_txt_content,
                )

                component_type_path = os.path.join(
                    app_component_info_dir, page_type_id + "_type.txt"
                )
                with open(component_type_path, 'w', encoding='utf-8') as file:
                    file.write(component_type)

                # 更新 state_file 并写入 json
                state_file[app][page_type_id] = 1
                with open('state.json', 'w', encoding='utf-8') as f:
                    json.dump(state_file, f, ensure_ascii=False, indent=4)

                print(f"Finished {app} {page_type_id}")

            print(f"Finished {app}")

        print(f"Finished category: {category}")
