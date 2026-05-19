import os
import json
import base64
import io
from PIL import Image


def get_base64_image(image_path):
    try:
        if not os.path.exists(image_path):
            return ""
        with Image.open(image_path) as img:
            img.thumbnail((600, 900))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""


def get_html_template(elements_json):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>UTG Interaction Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; }}
        #cy {{ width: 100vw; height: 100vh; background: #f4f4f4; }}
        #tooltip {{
            display: none;
            position: fixed;
            z-index: 9999;
            width: 600px;
            height: 900px;
            background-color: transparent;
            border: none;
            box-shadow: none;
            pointer-events: none;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
    </style>
</head>
<body>
    <div id="cy"></div>
    <div id="tooltip"></div>

    <script>
        const elements = {elements_json};
        const tooltip = document.getElementById('tooltip');

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node[type="parent"]',
                    style: {{
                        'shape': 'round-rectangle',
                        'background-opacity': 0.06,
                        'background-color': '#9aa5b1',
                        'border-width': 1.2,
                        'border-color': '#8a8f98',
                        'label': 'data(label)',
                        'text-valign': 'top',
                        'text-halign': 'center',
                        'font-size': '12px',
                        'color': '#333'
                    }}
                }},
                {{
                    selector: 'node[img]',
                    style: {{
                        'background-image': 'data(img)',
                        'background-fit': 'cover',
                        'width': 60,
                        'height': 90,
                        'shape': 'rectangle',
                        'label': 'data(id)',
                        'text-valign': 'bottom',
                        'font-size': '12px',
                        'text-wrap': 'wrap',
                        'text-max-width': 120,
                        'border-width': 1,
                        'border-color': '#666'
                    }}
                }},
                {{
                    selector: 'node.edge-active-node',
                    style: {{
                        'width': 120,
                        'height': 180,
                        'border-width': 3,
                        'border-color': '#ff8c00'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 1.2,
                        'line-color': 'data(edge_color)',
                        'target-arrow-color': 'data(edge_color)',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 0.7,
                        'curve-style': 'bezier'
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                nodeRepulsion: 1800,
                idealEdgeLength: 45,
                edgeElasticity: 90,
                gravity: 1.2,
                nestingFactor: 0.7,
                componentSpacing: 35,
                padding: 30,
                animate: false
            }}
        }});

        // 只放大预览图，不改变节点/父盒子尺寸
        cy.on('mouseover', 'node[img]', function(evt) {{
            const node = evt.target;
            tooltip.style.display = 'block';
            tooltip.style.backgroundImage = 'url(' + node.data('img') + ')';
        }});
        cy.on('mousemove', 'node[img]', function(evt) {{
            tooltip.style.left = (evt.originalEvent.clientX + 20) + 'px';
            tooltip.style.top = (evt.originalEvent.clientY - 280) + 'px';
        }});
        cy.on('mouseout', 'node[img]', function(evt) {{
            tooltip.style.display = 'none';
        }});

        // 点击边：同时放大该边两端节点
        function clearEdgeActiveNodes() {{
            cy.nodes().removeClass('edge-active-node');
        }}

        cy.on('tap', 'edge', function(evt) {{
            clearEdgeActiveNodes();
            const e = evt.target;
            e.source().addClass('edge-active-node');
            e.target().addClass('edge-active-node');
        }});

        // 点击空白区域时取消放大
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                clearEdgeActiveNodes();
            }}
        }});
    </script>
</body>
</html>
    """


def process_utg_data(json_path, root_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_key = "N" if isinstance(data.get("N"), list) else "nodes"
    e_key = "E" if isinstance(data.get("E"), list) else "edges"
    nodes = data.get(n_key, [])
    edges = data.get(e_key, [])

    elements = []
    page_to_parent_id = {}
    for n in nodes:
        page = str(n.get("page", n.get("page_type", ""))).strip()
        if not page:
            page = "UNKNOWN_PAGE"
        if page not in page_to_parent_id:
            page_to_parent_id[page] = "page::" + page

    # 先放父盒子
    for page, parent_id in page_to_parent_id.items():
        elements.append(
            {
                "data": {
                    "id": parent_id,
                    "label": page,
                    "type": "parent",
                }
            }
        )

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        page = str(node.get("page", node.get("page_type", ""))).strip()
        if not page:
            page = "UNKNOWN_PAGE"
        img_path = os.path.join(root_dir, node_id)
        elements.append(
            {
                "data": {
                    "id": node_id,
                    "parent": page_to_parent_id[page],
                    "img": get_base64_image(img_path),
                }
            }
        )

    for edge in edges:
        edge_id = str(edge.get("id", "")).strip()
        src = str(edge.get("from", "")).strip()
        dst = str(edge.get("to", "")).strip()
        same_page = bool(edge.get("same_page", False))
        edge_color = "#FFD700" if same_page is False else "#000000"
        if not src or not dst:
            continue
        elements.append(
            {
                "data": {
                    "id": edge_id if edge_id else f"{src}->{dst}",
                    "source": src,
                    "target": dst,
                    "same_page": same_page,
                    "edge_color": edge_color,
                }
            }
        )

    return json.dumps(elements)


def batch_process(base_path):
    for root, dirs, files in os.walk(base_path):
        if "utg_label.json" in files:
            json_path = os.path.join(root, "utg_label.json")
            output_path = os.path.join(root, "visualization.html")
            elements_json = process_utg_data(json_path, root)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(get_html_template(elements_json))
            print(f"已生成: {output_path}")

if __name__ == "__main__":
    batch_process(r'D:\ui2code\InteractionSequence\data')
    
    # root = r"D:\ui2code\InteractionSequence\data\com\healthbuddy"
    # json_path = os.path.join(root, "utg_label.json")
    # output_path = os.path.join(root, "visualization.html")
    # elements_json = process_utg_data(json_path, root)
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write(get_html_template(elements_json))
    # print(f"已生成: {output_path}")
