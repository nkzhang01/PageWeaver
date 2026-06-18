import os
import json
import base64
import math
import networkx as nx
import numpy as np
from pathlib import Path
from PIL import Image
import io


# "glitchvideoeditor" / "glitchvideoeffect
# 定义根目录
BASE_DIR = r"D:\ui2code\InteractionSequence\data4"


import networkx as nx

import networkx as nx
import numpy as np
def calculate_clustered_layout(nodes, edges, page_types):

    """
    自适应两阶段布局：
    box_spacing 会根据 page_types 的数量自动计算，不再需要手动传入。
    """
    num_clusters = len(page_types)
    
    # 【动态计算策略】：
    # 基础空间 600px，每多一个页面盒子，额外增加 150px 的空间
    # 这能保证：页面少时紧凑，页面多时自动撑开，防止重叠
    dynamic_spacing = 600 + (num_clusters * 150)
    
    # 还可以根据图片节点总数进行微调 (如果图片总数巨大，整体放大)
    num_total_nodes = len(nodes)
    if num_total_nodes > 100:
        dynamic_spacing *= 1.2 # 如果整图节点超过100，再额外放大 20%
        
    print(f"  [布局优化] 页面类型: {num_clusters}, 节点总数: {num_total_nodes}, 自动设置 box_spacing: {int(dynamic_spacing)}")
    
    box_spacing = dynamic_spacing
    # 1. 建立节点到 PageType 的映射表
    node_to_pt = {n['data']['id']: n['data']['parent'] for n in nodes if 'parent' in n['data']}

    # 2. 构建 Page Type 之间的关系图
    cluster_graph = nx.Graph()
    for pt in page_types: cluster_graph.add_node(pt)
    
    # 【核心优化】：统计 Page Type 之间的跳转权重
    for e in edges:
        # 只处理跨页面的跳转 (inter_edge)
        if e.get('classes') == 'inter_edge':
            s_pt = node_to_pt.get(e['data']['source'])
            t_pt = node_to_pt.get(e['data']['target'])
            
            if s_pt and t_pt and s_pt != t_pt:
                # 增加边的权重，权重越高，布局时越靠拢
                if cluster_graph.has_edge(s_pt, t_pt):
                    cluster_graph[s_pt][t_pt]['weight'] += 1
                else:
                    cluster_graph.add_edge(s_pt, t_pt, weight=1)

    # 3. 计算盒子中心坐标 (使用 kamada_kawai_layout)
    # 这个算法会通过计算图的能量最小化，使得交叉最少
    cluster_pos = nx.kamada_kawai_layout(cluster_graph, weight='weight', scale=box_spacing)
    
    # 4. 为每个节点分配坐标
    final_positions = {}
    grouped_nodes = {pt: [] for pt in page_types}
    for n in nodes:
        if 'parent' in n['data']:
            grouped_nodes[n['data']['parent']].append(n['data']['id'])
        
    # 5. 在盒子内部环形布局
    for pt, node_ids in grouped_nodes.items():
        center = cluster_pos[pt]
        num_nodes = len(node_ids)
        for i, nid in enumerate(node_ids):
            # 这里的 radius 可以根据节点数量动态增大，防止内部重叠
            # radius = 100 + (num_nodes * 10) 
            radius = 150 + (num_nodes * 25)
            angle = 2 * math.pi * i / max(num_nodes, 1)
            final_positions[nid] = {
                'x': int(center[0] + math.cos(angle) * radius), 
                'y': int(center[1] + math.sin(angle) * radius)
            }
    
    # 把盒子节点位置也记录进去
    for pt in page_types:
        final_positions[pt] = {'x': int(cluster_pos[pt][0]), 'y': int(cluster_pos[pt][1])}
        
    return final_positions

def get_base64_image(image_path):
    """转换图片为 Base64，保持较高分辨率"""
    try:
        if not os.path.exists(image_path): return ""
        with Image.open(image_path) as img:
            # 调整缩略图尺寸为 600x900，保证放大后清晰
            img.thumbnail((600, 900)) 
            buffered = io.BytesIO()
            # 使用 JPEG 可以显著减小大图的 Base64 长度
            img.save(buffered, format="JPEG", quality=85) 
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception: return ""


def resolve_utg_node_image(app_dir, node_id):
    node_id = str(node_id or "").strip()
    if not node_id:
        return ""

    direct_path = app_dir / node_id
    if direct_path.exists():
        return str(direct_path)

    stem = Path(node_id).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = app_dir / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate)

    return ""


def read_utg_json(utg_path):
    text = Path(utg_path).read_text(encoding="utf-8", errors="ignore")
    marker = "var utg ="
    if marker in text:
        text = text[text.find(marker) + len(marker):].strip()
    return json.loads(text)


def build_plain_utg_elements(utg_data, app_dir):
    nodes = utg_data.get("N", utg_data.get("nodes", []))
    edges = utg_data.get("E", utg_data.get("edges", []))

    elements = []
    known_node_ids = set()
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        known_node_ids.add(node_id)
        image_path = resolve_utg_node_image(app_dir, node_id)
        elements.append({
            "data": {
                "id": node_id,
                "label": node_id,
                "img": get_base64_image(image_path),
            }
        })

    for idx, edge in enumerate(edges):
        src = str(edge.get("from", edge.get("source", ""))).strip()
        dst = str(edge.get("to", edge.get("target", ""))).strip()
        if not src or not dst:
            continue
        if src not in known_node_ids or dst not in known_node_ids:
            continue

        conditions = edge.get("conditions", edge.get("condition", ""))
        if isinstance(conditions, list):
            edge_label = ", ".join(str(x) for x in conditions)
        else:
            edge_label = str(conditions or "")
        edge_id = str(edge.get("id", f"edge::{idx}::{src}->{dst}"))
        elements.append({
            "data": {
                "id": edge_id,
                "source": src,
                "target": dst,
                "label": edge_label,
            }
        })

    return elements


def generate_plain_utg_html(elements, output_path, app_name):
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{app_name} UTG</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: sans-serif; }}
        #cy {{ width: 100vw; height: 100vh; background: #f5f6f8; }}
        #tooltip {{
            display: none;
            position: fixed;
            z-index: 9999;
            width: 360px;
            height: 640px;
            border: 2px solid #333;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            background-color: #fff;
            pointer-events: none;
        }}
        #title {{
            position: fixed;
            left: 12px;
            top: 10px;
            z-index: 10;
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #d0d7de;
            border-radius: 6px;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div id="title">{app_name} UTG</div>
    <div id="cy"></div>
    <div id="tooltip"></div>
    <script>
        const elements = {json.dumps(elements, ensure_ascii=False)};
        const tooltip = document.getElementById("tooltip");
        const cy = cytoscape({{
            container: document.getElementById("cy"),
            elements: elements,
            style: [
                {{
                    selector: "node",
                    style: {{
                        "shape": "rectangle",
                        "width": 90,
                        "height": 160,
                        "background-image": "data(img)",
                        "background-fit": "cover",
                        "background-color": "#e5e7eb",
                        "border-width": 2,
                        "border-color": "#4b5563",
                        "label": "data(label)",
                        "font-size": 10,
                        "text-valign": "bottom",
                        "text-margin-y": 10,
                        "text-wrap": "wrap",
                        "text-max-width": 150,
                        "color": "#111827"
                    }}
                }},
                {{
                    selector: "edge",
                    style: {{
                        "curve-style": "bezier",
                        "width": 1.4,
                        "line-color": "#6b7280",
                        "target-arrow-color": "#6b7280",
                        "target-arrow-shape": "triangle",
                        "arrow-scale": 0.8,
                        "label": "data(label)",
                        "font-size": 6,
                        "text-rotation": "autorotate",
                        "color": "#374151",
                        "text-background-opacity": 0.8,
                        "text-background-color": "#fff",
                        "text-background-padding": 2
                    }}
                }},
                {{
                    selector: "node:selected",
                    style: {{
                        "border-width": 5,
                        "border-color": "#f59e0b"
                    }}
                }}
            ],
            layout: {{
                name: "cose",
                nodeRepulsion: 5000,
                idealEdgeLength: 140,
                edgeElasticity: 90,
                gravity: 0.35,
                componentSpacing: 120,
                padding: 70,
                animate: false
            }}
        }});

        cy.on("mouseover", "node", function(evt) {{
            const img = evt.target.data("img");
            if (!img) return;
            tooltip.style.display = "block";
            tooltip.style.backgroundImage = "url(" + img + ")";
        }});
        cy.on("mousemove", "node", function(evt) {{
            tooltip.style.left = (evt.originalEvent.clientX + 20) + "px";
            tooltip.style.top = Math.max(10, evt.originalEvent.clientY - 260) + "px";
        }});
        cy.on("mouseout", "node", function() {{
            tooltip.style.display = "none";
        }});
    </script>
</body>
</html>"""
    Path(output_path).write_text(html_content, encoding="utf-8")


def process_plain_utg_app(app_dir):
    app_dir = Path(app_dir)
    utg_path = app_dir / "utg.json"
    if not utg_path.exists():
        return False
    utg_data = read_utg_json(utg_path)
    elements = build_plain_utg_elements(utg_data, app_dir)
    output_path = app_dir / "utg_show.html"
    generate_plain_utg_html(elements, output_path, app_dir.name)
    print(f"[OK] {output_path}")
    return True


def run_all_plain_utg_apps(base_dir=BASE_DIR):
    base = Path(base_dir)
    total = 0
    for utg_path in sorted(base.rglob("utg.json"), key=lambda p: str(p)):
        app_dir = utg_path.parent
        try:
            if process_plain_utg_app(app_dir):
                total += 1
        except Exception as exc:
            print(f"[FAIL] {app_dir}: {exc}")
    print(f"generated_utg_show_count={total}")

def generate_full_utg_html(master_nodes, master_edges, output_path, app_name="App Master UTG"):
    print(f"🎉 正在生成包含 Hover 交互的 Master HTML: {output_path.name}...")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{app_name}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        #cy {{ width: 100vw; height: 100vh; background: #1a1a1a; }}
        body {{ margin: 0; overflow: hidden; font-family: sans-serif; }}
        /* 增加一个辅助提示框 */
        #info-panel {{
            position: absolute; bottom: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 5px 10px; border-radius: 4px; font-size: 12px;
            z-index: 100; pointer-events: none;
        }}
    </style>
</head>
<body>
    <div id="info-panel">鼠标悬停节点可放大查看细节</div>
    <div id="cy"></div>
    <script>
        var elements = {json.dumps(master_nodes + master_edges)};
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'shape': 'rectangle',
                        // 1. 基础尺寸调大，确保缩小时可见
                        'width': 180,  
                        'height': 270,
                        'background-image': 'data(b64)',
                        'background-fit': 'cover',
                        // 2. 增加显眼的边框，在远视距下突出节点
                        'border-width': 4,
                        'border-color': '#444',
                        // 3. 优化文字显示
                        'label': 'data(label)',
                        'font-size': '10px',
                        'text-valign': 'bottom',
                        'text-margin-y': 8,
                        'color': '#eee',
                        'text-background-opacity': 0.5,
                        'text-background-color': '#000',
                        'text-background-padding': '2px',
                        // 4. 开启高清渲染
                        'background-image-smoothing': 'yes',
                        // 5. 动画过渡效果，让放大更平滑
                        'transition-property': 'width, height, border-color, border-width',
                        'transition-duration': '0.2s'
                    }}
                }},
                // 鼠标悬停时的动态样式 (Hover Style)
                {{
                    selector: 'node:selected, node:active, .hover', 
                    style: {{
                        // 6. 悬停时剧烈放大
                        'width': 600, 
                        'height': 900,
                        'border-color': '#ffc107', // 变为黄色边框
                        'border-width': 8,
                        'font-size': '20px', // 文字也变大
                        'z-index': 9999      // 确保在最上层显示
                    }}
                }},
                {{
                    selector: '.page_group',
                    style: {{
                        'background-opacity': 0.03,
                        'background-color': '#007bff',
                        'label': 'data(label)',
                        'text-valign': 'top',
                        'font-size': '24px',
                        'font-weight': 'bold',
                        'border-width': 2,
                        'border-color': '#007bff',
                        'border-style': 'dashed',
                        'color': '#007bff',
                        'padding': '50px'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                       'curve-style': 'bezier',        // 强制使用贝塞尔曲线
                        'width': 1.5,
                        'line-color': '#888',
                        'target-arrow-shape': 'triangle',
                        'target-arrow-size': 6,
                        
                        // 关键点1：控制曲线的弯曲度，防止线段完全重叠
                        'control-point-step-size': 40,  
                        
                        // 关键点2：如果存在多条线连接同一对节点，使用 edge-distances
                        // 这会自动在多条线之间产生间隔，防止它们“贴”在一起
                        'edge-distances': 'node-position', 
                        
                        'font-size': '6px',
                        'color': '#888',
                        'opacity': 0.6
                    }}
                }},
                {{
                    selector: '.inter_edge',
                    style: {{
                        'width': 2.5,
                        'line-color': '#ffc107',
                        'target-arrow-color': '#ffc107',
                        'curve-style': 'bezier',
                        
                        // 关键点3：跨页线弯曲度更大，避开中间的拥挤区域
                        'control-point-step-size': 100, 
                        
                        // 关键点4：为多线连接设置偏移，让线束像“发散”开一样
                        'edge-distances': 'node-position',
                        
                        'opacity': 0.9
                    }}
                }}
            ],
            layout: {{ name: 'preset', animate: false }},
            // 开启像素比优化
            pixelRatio: 'auto' 
        }});

        // 【核心 JS 交互逻辑】：鼠标悬停放大
        cy.on('mouseover', 'node', function(e){{
            var node = e.target;
            // 只有截图节点才放大，Page Group 盒子不放大
            if (!node.hasClass('page_group')) {{
                node.addClass('hover');
            }}
        }});

        cy.on('mouseout', 'node', function(e){{
            e.target.removeClass('hover');
        }});

    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
def process_app_package(pkg_path, pkg_name):
    """处理单个 App 包目录，合并数据"""
    print(f"Processing package: {pkg_name} ...")
    pkg_root = Path(pkg_path)
    
    # 定义核心文件路径
    inter_json_path = pkg_root / "inter_page_transition_graph_2.json"
    intra_json_path = pkg_root / "intra_page_type_subgraphs_2.json"
    
    if not inter_json_path.exists() or not intra_json_path.exists():
        print(f"  [Error] 缺少核心 JSON 文件，跳过: {pkg_name}")
        return

    # 初始化主 elements
    master_nodes = []
    master_edges = []
    
    # 步骤一：处理 Intra-page (页面内状态)
    print("  -> 处理 Intra 数据（Base64图片编码，请稍候）...")
    with open(intra_json_path, 'r', encoding='utf-8') as f:
        intra_data = json.load(f)
    
    for item in intra_data.get("subgraphs", []):
        pt_id = item.get("page_type_id")
        pt_name = item.get("page_type_name")
        utg_data = item.get("utg_subgraph", {})
        
        # 1. 创建页面大盒子节点 (修复这里：改为单括号)
        master_nodes.append({
            'data': { 'id': pt_id, 'label': f"Page: {pt_name} ({pt_id})" },
            'classes': 'page_group'
        })
        
        # 2. 提取并编码盒子内节点 (修复这里：改为单括号)
        for node in utg_data.get("N", []):
            nid = node.get('id', '')
            full_img_path = pkg_root / nid
            master_nodes.append({
                'data': {
                    'id': nid,
                    'parent': pt_id,
                    # 'label': node.get('name', nid),
                    'label': nid,
                    'b64': get_base64_image(str(full_img_path))
                }
            })
        
        # 3. 提取盒内的同页交互边 (修复这里：改为单括号)
        for edge in utg_data.get("E", []):
            master_edges.append({
                'data': {
                    'source': edge.get('from', ''),
                    'target': edge.get('to', ''),
                    'label': ", ".join(edge.get('conditions', []))
                }
            })

    # 步骤二：处理 Inter-page (跨页面跳转)
   # 替代方案：直接连接具体的节点，而不是连接页面盒子
    print("  -> 处理 Inter 跳转数据 (直接连接节点)...")
    with open(inter_json_path, 'r', encoding='utf-8') as f:
        inter_data = json.load(f)
    
    utg = inter_data.get("utg_graph", inter_data)
    for edge in utg.get("E", []):
        fnid = edge.get('from', '')
        tnid = edge.get('to', '')
        
        # 直接添加边，连接两个具体的图片节点
        master_edges.append({
            'data': {
                'source': fnid,
                'target': tnid,
                'label': "Jump"
            },
            'classes': 'inter_edge'
        })
        print(f"  [成功添加跳转线] {fnid} -> {tnid}")
    
    # 步骤三：输出全量 HTML
    output_master_html = pkg_root / f"{pkg_name}_Full_UTG_2.html"

    page_types = list(set([n['data']['parent'] for n in master_nodes if 'parent' in n['data']]))
    pos_map = calculate_clustered_layout(master_nodes, master_edges, page_types)
    for n in master_nodes: n['position'] = pos_map.get(n['data']['id'])
    generate_full_utg_html(master_nodes, master_edges, output_master_html, app_name=pkg_name)

    # 插入布局计算
    # print("  -> 正在进行布局预计算...")
    # master_nodes = compute_preset_layout(master_nodes, master_edges)

    # # 然后生成 HTML
    # generate_full_utg_html(master_nodes, master_edges, output_master_html, app_name=pkg_name)
    # generate_full_utg_html(master_nodes, master_edges, output_master_html, app_name=pkg_name)

def run_all_apps():
    """遍历目录处理每个 App"""


    app_classification_lists = [
        "com", "a2z", "app", "br", "co", "de", "dogcat", "drzio", "english",
        "fr", "freemusic", "glitchvideoeditor", "hk", "in", "ir", "is", "it",
        "media", "order", "org", "ott", "sk", "textmessages", "zehitech",
    ]   


    com_root = Path(BASE_DIR)

    for category in app_classification_lists:
        category_dir = com_root / str(category)
        if not category_dir.exists() or not category_dir.is_dir():
            print(f"[SKIP] {category}: category dir not found")
            continue
        for item in category_dir.iterdir():
            if item.is_dir():
                process_app_package(item, item.name)



def run_single_app(app_path):
    """处理单个 App 目录"""
    app_dir = Path(app_path)
    if not app_dir.exists():
        print(f"Error: 目录不存在 {app_path}")
        return

    if not app_dir.is_dir():
        print(f"Error: 不是目录 {app_path}")
        return

    process_app_package(app_dir, app_dir.name)


if __name__ == "__main__":
    run_all_plain_utg_apps()
    # # run_all_apps()
    # path = r"D:\ui2code\InteractionSequence\data\com\aestheticmedicine"
    # run_single_app(path)
