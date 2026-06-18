import posixpath
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_ROOT = REPO_ROOT / "results" / "code_results" / "baseline" / "baseline_UTG+Mockups"
TARGET_ROOT = DEFAULT_TARGET_ROOT
WRITE_CHANGES = True
LIMIT_FILES = 1
OUTPUT_FILENAME = "res_label.txt"
SECTION_MARKER_RE = re.compile(r"^@@@(.+?)@@@$", re.MULTILINE)
LOCAL_IMPORT_RE = re.compile(
    r"import\s+(?P<names>[^;]+?)\s+from\s+['\"](?P<path>\.[^'\"]+)['\"]\s*;"
)
APP_IMPORT_RE = re.compile(
    r"import\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s+from\s+['\"](?P<path>\.[^'\"]+)['\"]\s*;"
)
DECLARED_COMPONENT_RE = re.compile(
    r"(?:export\s+default\s+function|function|const)\s+(?P<name>[A-Z][A-Za-z0-9_]*)\b"
)
ROUTED_COMPONENT_RE = re.compile(
    r"(?:case\s+['\"][^'\"]+['\"]\s*:\s*return|default\s*:\s*return)\s*<(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
FUNCTION_COMPONENT_RE = re.compile(
    r"export\s+default\s+function\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s*\((?P<params>[\s\S]*?)\)"
)
ARROW_COMPONENT_RE = re.compile(
    r"const\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s*=\s*\((?P<params>[\s\S]*?)\)\s*=>"
)
ONPRESS_PROP_NAME_RE = re.compile(r"(?<![\w.])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=")
RETURN_BLOCK_RE = re.compile(r"\breturn\s*\(")


@dataclass
class Section:
    name: str
    code: str


def split_res_sections(res_text: str) -> list[Section]:
    matches = list(SECTION_MARKER_RE.finditer(res_text))
    sections: list[Section] = []
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(res_text)
        body = res_text[body_start:body_end].lstrip("\r\n")
        sections.append(Section(name=name, code=body))
    return sections


def join_res_sections(sections: list[Section]) -> str:
    parts: list[str] = []
    for section in sections:
        code = section.code.rstrip() + "\n"
        parts.append(f"@@@{section.name}@@@\n{code}")
    return "\n".join(parts).rstrip() + "\n"


def normalize_relative_js_path(import_path: str, base_section_name: str | None = None) -> str:
    normalized = import_path.replace("\\", "/").strip()
    if not normalized.endswith(".js"):
        normalized = f"{normalized}.js"

    if base_section_name:
        base_dir = Path(base_section_name.replace("\\", "/")).parent.as_posix()
        return posixpath.normpath(posixpath.join(base_dir, normalized))

    normalized = re.sub(r"^\./", "", normalized)
    normalized = re.sub(r"^\.\./", "", normalized)
    return normalized


def extract_names_from_import_clause(import_clause: str) -> list[str]:
    names: list[str] = []
    normalized = import_clause.strip()
    if not normalized:
        return names

    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            inner = part[1:-1]
            for item in inner.split(","):
                item = item.strip()
                if not item:
                    continue
                names.append(item.split(" as ")[-1].strip())
        else:
            names.append(part)
    return [name for name in names if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)]


def extract_local_component_imports(code: str, section_name: str) -> dict[str, str]:
    imports: dict[str, str] = {}
    for match in LOCAL_IMPORT_RE.finditer(code):
        import_path = normalize_relative_js_path(match.group("path"), section_name)
        for import_name in extract_names_from_import_clause(match.group("names")):
            if import_name[:1].isupper():
                imports[import_name] = import_path
    return imports


def extract_page_section_names(app_code: str) -> set[str]:
    app_imports = {
        match.group("name"): normalize_relative_js_path(match.group("path"))
        for match in APP_IMPORT_RE.finditer(app_code)
    }
    routed_component_names = {
        match.group("name")
        for match in ROUTED_COMPONENT_RE.finditer(app_code)
    }
    if not routed_component_names:
        return set(app_imports.values())
    return {
        import_path
        for component_name, import_path in app_imports.items()
        if component_name in routed_component_names
    }


def extract_declared_component_names(code: str) -> set[str]:
    return {
        match.group("name")
        for match in DECLARED_COMPONENT_RE.finditer(code)
    }


def extract_used_component_names(code: str) -> set[str]:
    used_component_names: set[str] = set()
    for _, _, tag_text in iter_labelable_jsx_start_tags(code):
        name_match = re.match(r"<\s*([A-Z][A-Za-z0-9_]*)", tag_text)
        if name_match:
            used_component_names.add(name_match.group(1))
    return used_component_names


def detect_runtime_page_name_expr(code: str) -> str | None:
    for pattern in (r"\bcurrent\.name\b", r"\broute\.name\b", r"\bscreenName\b"):
        match = re.search(pattern, code)
        if match:
            return match.group(0)
    return None


def is_meaningful_onpress_expression(expression: str | None) -> bool:
    if expression is None:
        return False
    text = expression.strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if compact in {"null", "undefined", "false", "()=>{}", "()=>null", "(()=>{})", "(()=>null)"}:
        return False
    if re.fullmatch(r"\(?[\s\S]*?\)?=>\{\s*\}", text):
        return False
    if re.fullmatch(r"\(?[\s\S]*?\)?=>null", text):
        return False
    return True


def parse_jsx_attribute_value(tag_text: str, attribute_name: str) -> tuple[str, int, int] | None:
    match = re.search(rf"(?<![\w.]){re.escape(attribute_name)}\s*=", tag_text)
    if not match:
        return None

    index = match.end()
    while index < len(tag_text) and tag_text[index].isspace():
        index += 1
    if index >= len(tag_text):
        return None

    start = index
    start_char = tag_text[index]
    if start_char in {"'", '"'}:
        index += 1
        while index < len(tag_text):
            if tag_text[index] == start_char and tag_text[index - 1] != "\\":
                return tag_text[start + 1:index], start, index + 1
            index += 1
        return None

    if start_char == "{":
        depth = 1
        index += 1
        in_string: str | None = None
        while index < len(tag_text):
            char = tag_text[index]
            prev_char = tag_text[index - 1] if index > start + 1 else ""
            if in_string:
                if char == in_string and prev_char != "\\":
                    in_string = None
            else:
                if char in {"'", '"', "`"}:
                    in_string = char
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return tag_text[start + 1:index], start, index + 1
            index += 1
        return None

    while index < len(tag_text) and not tag_text[index].isspace() and tag_text[index] not in {">", "/"}:
        index += 1
    return tag_text[start:index], start, index


def set_or_replace_attribute(tag_text: str, attribute_name: str, attribute_value: str) -> str:
    pattern = re.compile(rf"(\s){re.escape(attribute_name)}\s*=\s*(\{{[\s\S]*?\}}|\"[^\"]*\"|'[^']*')")
    replacement = rf"\1{attribute_name}={attribute_value}"
    if pattern.search(tag_text):
        return pattern.sub(replacement, tag_text, count=1)

    insert_at = len(tag_text) - 2 if tag_text.endswith("/>") else len(tag_text) - 1
    prefix = "" if tag_text[max(0, insert_at - 1):insert_at].isspace() else " "
    return tag_text[:insert_at] + f"{prefix}{attribute_name}={attribute_value}" + tag_text[insert_at:]


def set_attribute_if_missing(tag_text: str, attribute_name: str, attribute_value: str) -> str:
    if parse_jsx_attribute_value(tag_text, attribute_name):
        return tag_text
    return set_or_replace_attribute(tag_text, attribute_name, attribute_value)


def iter_jsx_start_tags(source: str) -> list[tuple[int, int, str]]:
    tags: list[tuple[int, int, str]] = []
    index = 0
    while index < len(source):
        start = source.find("<", index)
        if start == -1:
            break
        if start + 1 >= len(source) or source[start + 1] in {"/", "!", "?", "%"}:
            index = start + 1
            continue
        if source[start + 1].isspace():
            index = start + 1
            continue

        depth = 0
        in_string: str | None = None
        end = start + 1
        while end < len(source):
            char = source[end]
            prev_char = source[end - 1] if end > start + 1 else ""
            if in_string:
                if char == in_string and prev_char != "\\":
                    in_string = None
            else:
                if char in {"'", '"', "`"}:
                    in_string = char
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth = max(0, depth - 1)
                elif char == ">" and depth == 0:
                    tags.append((start, end + 1, source[start:end + 1]))
                    break
            end += 1
        index = end + 1
    return tags


def find_matching_parenthesis(source: str, open_index: int) -> int | None:
    if open_index < 0 or open_index >= len(source) or source[open_index] != "(":
        return None

    depth = 1
    index = open_index + 1
    in_string: str | None = None
    while index < len(source):
        char = source[index]
        prev_char = source[index - 1] if index > open_index + 1 else ""
        if in_string:
            if char == in_string and prev_char != "\\":
                in_string = None
        else:
            if char in {"'", '"', "`"}:
                in_string = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        index += 1
    return None


def iter_labelable_ranges(code: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in RETURN_BLOCK_RE.finditer(code):
        open_index = match.end() - 1
        close_index = find_matching_parenthesis(code, open_index)
        if close_index is None:
            continue
        ranges.append((open_index + 1, close_index))

    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def iter_labelable_jsx_start_tags(code: str) -> list[tuple[int, int, str]]:
    tags: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for range_start, range_end in iter_labelable_ranges(code):
        for start, end, tag_text in iter_jsx_start_tags(code[range_start:range_end]):
            absolute_start = range_start + start
            absolute_end = range_start + end
            key = (absolute_start, absolute_end)
            if key in seen:
                continue
            seen.add(key)
            tags.append((absolute_start, absolute_end, tag_text))
    return tags


def has_meaningful_press_prop(tag_text: str) -> bool:
    press_prop_names = {
        match.group("name")
        for match in ONPRESS_PROP_NAME_RE.finditer(tag_text)
        if "onpress" in match.group("name").lower()
    }
    for prop_name in press_prop_names:
        prop_value = parse_jsx_attribute_value(tag_text, prop_name)
        if prop_value and is_meaningful_onpress_expression(prop_value[0]):
            return True
    return False


def add_root_container_label_if_needed(
    code: str,
    page_name: str,
    skipped_component_names: set[str] | None = None,
) -> str:
    return_match = re.search(r"return\s*\(", code)
    search_start = return_match.end() if return_match else 0
    skipped_component_names = skipped_component_names or set()

    for absolute_start, absolute_end, tag_text in iter_labelable_jsx_start_tags(code):
        if absolute_start < search_start:
            continue
        name_match = re.match(r"<\s*([A-Za-z_][A-Za-z0-9_.-]*)", tag_text)
        if not name_match:
            continue
        tag_name = name_match.group(1)
        if tag_name in skipped_component_names:
            continue
        if tag_name[0].isupper() is False and tag_name not in {"View", "SafeAreaView", "ScrollView"}:
            continue

        label_value = build_static_label(page_name, tag_name, 1)
        updated_tag = set_or_replace_attribute(tag_text, "testID", label_value)
        updated_tag = set_or_replace_attribute(updated_tag, "accessibilityLabel", label_value)
        return code[:absolute_start] + updated_tag + code[absolute_end:]
    return code


def add_labels_to_clickable_components(
    code: str,
    prefix_builder,
    skipped_component_names: set[str] | None = None,
) -> tuple[str, int]:
    replacements: list[tuple[int, int, str]] = []
    counters: dict[str, int] = {}
    skipped_component_names = skipped_component_names or set()

    for start, end, tag_text in iter_labelable_jsx_start_tags(code):
        name_match = re.match(r"<\s*([A-Za-z_][A-Za-z0-9_.-]*)", tag_text)
        if not name_match:
            continue
        tag_name = name_match.group(1)
        if tag_name in skipped_component_names:
            continue

        if not has_meaningful_press_prop(tag_text):
            continue

        counters[tag_name] = counters.get(tag_name, 0) + 1
        sequence = counters[tag_name]
        label_value = prefix_builder(tag_name, sequence)
        updated_tag = set_or_replace_attribute(tag_text, "testID", label_value)
        updated_tag = set_or_replace_attribute(updated_tag, "accessibilityLabel", label_value)
        replacements.append((start, end, updated_tag))

    if not replacements:
        return code, 0

    output: list[str] = []
    cursor = 0
    for start, end, updated_tag in replacements:
        output.append(code[cursor:start])
        output.append(updated_tag)
        cursor = end
    output.append(code[cursor:])
    return "".join(output), len(replacements)


def inject_page_name_prop_into_usages(code: str, imported_components: set[str], page_name_expr: str) -> str:
    replacements: list[tuple[int, int, str]] = []
    for start, end, tag_text in iter_labelable_jsx_start_tags(code):
        name_match = re.match(r"<\s*([A-Z][A-Za-z0-9_]*)", tag_text)
        if not name_match:
            continue
        tag_name = name_match.group(1)
        if tag_name not in imported_components:
            continue
        if not has_meaningful_press_prop(tag_text):
            continue
        if parse_jsx_attribute_value(tag_text, "pageName"):
            continue
        updated_tag = set_or_replace_attribute(tag_text, "pageName", page_name_expr)
        replacements.append((start, end, updated_tag))

    if not replacements:
        return code

    output: list[str] = []
    cursor = 0
    for start, end, updated_tag in replacements:
        output.append(code[cursor:start])
        output.append(updated_tag)
        cursor = end
    output.append(code[cursor:])
    return "".join(output)


def ensure_component_signature_props(code: str, prop_names: list[str]) -> str:
    replacements: list[tuple[int, int, str]] = []
    for regex in (FUNCTION_COMPONENT_RE, ARROW_COMPONENT_RE):
        for match in regex.finditer(code):
            params = match.group("params").strip()
            replacement: str | None = None
            if params.startswith("{") and params.endswith("}"):
                inner = params[1:-1].strip()
                existing_names = {
                    item.split(":")[0].split("=")[0].strip()
                    for item in inner.split(",")
                    if item.strip()
                }
                missing = [prop_name for prop_name in prop_names if prop_name not in existing_names]
                if missing:
                    additions = ", ".join(missing)
                    inner = f"{inner}, {additions}" if inner else additions
                    replacement = "{ " + inner + " }"
            elif params == "":
                replacement = "{ " + ", ".join(prop_names) + " }"
            if replacement is not None:
                replacements.append((match.start("params"), match.end("params"), replacement))

    if not replacements:
        return code

    output: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(replacements):
        output.append(code[cursor:start])
        output.append(replacement)
        cursor = end
    output.append(code[cursor:])
    return "".join(output)


def inject_forwarded_label_props_into_component_roots(code: str) -> str:
    replacements: list[tuple[int, int, str]] = []

    for match in FUNCTION_COMPONENT_RE.finditer(code):
        return_match = RETURN_BLOCK_RE.search(code, match.end())
        if not return_match:
            continue
        open_index = return_match.end() - 1
        close_index = find_matching_parenthesis(code, open_index)
        if close_index is None:
            continue
        tags = iter_jsx_start_tags(code[open_index + 1:close_index])
        if not tags:
            continue
        start, end, tag_text = tags[0]
        updated_tag = set_attribute_if_missing(tag_text, "testID", "{testID}")
        updated_tag = set_attribute_if_missing(updated_tag, "accessibilityLabel", "{accessibilityLabel}")
        replacements.append((open_index + 1 + start, open_index + 1 + end, updated_tag))

    for match in ARROW_COMPONENT_RE.finditer(code):
        index = match.end()
        while index < len(code) and code[index].isspace():
            index += 1
        if index >= len(code) or code[index] != "(":
            continue
        close_index = find_matching_parenthesis(code, index)
        if close_index is None:
            continue
        tags = iter_jsx_start_tags(code[index + 1:close_index])
        if not tags:
            continue
        start, end, tag_text = tags[0]
        updated_tag = set_attribute_if_missing(tag_text, "testID", "{testID}")
        updated_tag = set_attribute_if_missing(updated_tag, "accessibilityLabel", "{accessibilityLabel}")
        replacements.append((index + 1 + start, index + 1 + end, updated_tag))

    if not replacements:
        return code

    output: list[str] = []
    cursor = 0
    for start, end, updated_tag in sorted(replacements):
        output.append(code[cursor:start])
        output.append(updated_tag)
        cursor = end
    output.append(code[cursor:])
    return "".join(output)


def build_static_label(page_name: str, component_type: str, index: int) -> str:
    return f'"{page_name}.{component_type}.{index}"'


def build_component_label(component_name: str, component_type: str, index: int) -> str:
    return '{' + f"pageName ? `$"+"{pageName}."+f"{component_name}.{component_type}.{index}` : undefined" + '}'


def process_page_code(code: str, page_name: str, imported_components: set[str]) -> str:
    updated, label_count = add_labels_to_clickable_components(
        code,
        lambda component_type, index: build_static_label(page_name, component_type, index),
    )
    if imported_components:
        updated = inject_page_name_prop_into_usages(updated, imported_components, f'"{page_name}"')
    if label_count == 0:
        updated = add_root_container_label_if_needed(updated, page_name)
    return updated


def process_component_code(code: str, component_name: str, imported_components: set[str]) -> str:
    updated = ensure_component_signature_props(code, ["pageName", "testID", "accessibilityLabel"])
    updated = inject_forwarded_label_props_into_component_roots(updated)
    updated, _ = add_labels_to_clickable_components(
        updated,
        lambda component_type, index: build_component_label(component_name, component_type, index),
    )
    if imported_components:
        updated = inject_page_name_prop_into_usages(updated, imported_components, "{pageName}")
    return updated


def process_app_code(code: str, imported_components: set[str]) -> str:
    page_name_expr = detect_runtime_page_name_expr(code)
    if not page_name_expr or not imported_components:
        return code
    return inject_page_name_prop_into_usages(code, imported_components, "{" + page_name_expr + "}")


def process_res_sections(sections: list[Section]) -> list[Section]:
    section_by_name = {section.name: section for section in sections}
    app_section = section_by_name.get("App.js")
    page_section_names = extract_page_section_names(app_section.code) if app_section else set()

    local_imports_by_section = {
        section.name: extract_local_component_imports(section.code, section.name)
        for section in sections
    }
    used_component_names_by_section = {
        section.name: extract_used_component_names(section.code)
        for section in sections
    }
    common_component_sections: set[str] = set()
    changed = True
    while changed:
        changed = False
        caller_section_names = set(page_section_names) | {"App.js"} | common_component_sections
        for caller_section_name in caller_section_names:
            imports = local_imports_by_section.get(caller_section_name, {})
            used_component_names = used_component_names_by_section.get(caller_section_name, set())
            for component_name, import_path in imports.items():
                if component_name not in used_component_names:
                    continue
                if import_path in page_section_names or import_path == "App.js" or import_path in common_component_sections:
                    continue
                common_component_sections.add(import_path)
                changed = True

    updated_sections: list[Section] = []
    for section in sections:
        file_name = Path(section.name).name
        section_stem = Path(file_name).stem
        imported_components = {
            component_name
            for component_name, import_path in local_imports_by_section.get(section.name, {}).items()
            if import_path in common_component_sections
        }

        if section.name == "App.js":
            updated_code = process_app_code(section.code, imported_components)
        elif section.name in page_section_names:
            updated_code = process_page_code(section.code, section_stem, imported_components)
        else:
            updated_code = process_component_code(section.code, section_stem, imported_components)
        updated_sections.append(Section(name=section.name, code=updated_code))
    return updated_sections


def get_output_res_path(res_path: Path) -> Path:
    return res_path.with_name(OUTPUT_FILENAME)


def process_res_file(res_path: Path, write: bool) -> tuple[bool, Path]:
    original_text = res_path.read_text(encoding="utf-8", errors="ignore")
    sections = split_res_sections(original_text)
    if not sections:
        return False, get_output_res_path(res_path)

    updated_text = join_res_sections(process_res_sections(sections))
    changed = updated_text != original_text
    output_path = get_output_res_path(res_path)
    if write:
        output_path.write_text(updated_text, encoding="utf-8")
    return changed, output_path


def iter_res_files(root: Path) -> list[Path]:
    return sorted(root.rglob("res.txt"))


def main() -> None:
    root = TARGET_ROOT.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    res_files = iter_res_files(root)
    if LIMIT_FILES > 0:
        res_files = res_files[:LIMIT_FILES]

    changed_count = 0
    for res_file in res_files:
        changed, output_path = process_res_file(res_file, write=WRITE_CHANGES)
        status = "changed" if changed else "unchanged"
        mode = "write" if WRITE_CHANGES else "preview"
        print(f"[{mode}] {status}: {res_file} -> {output_path}")
        if changed:
            changed_count += 1

    print(f"[summary] res.txt files: {len(res_files)}, changed: {changed_count}, write: {WRITE_CHANGES}")


if __name__ == "__main__":
    main()
