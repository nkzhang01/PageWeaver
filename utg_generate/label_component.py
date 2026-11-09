import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_ROOT = REPO_ROOT / "results" / "code_results" /"baselines"/ "E2E-prompted"
# DEFAULT_TARGET_ROOT = REPO_ROOT / "results" / "code_results"/ "PageWeaver_gemini2.5"
TARGET_ROOT = DEFAULT_TARGET_ROOT
WRITE_CHANGES = True
LIMIT_FILES = None
OUTPUT_FILENAME = "res_label.txt"
SECTION_MARKER_RE = re.compile(r"^@@@(.+?)@@@$", re.MULTILINE)

LOCAL_IMPORT_RE = re.compile(
    r"import\s+(?P<names>[^;]+?)\s+from\s+['\"](?P<path>\.[^'\"]+)['\"]\s*;"
)
APP_IMPORT_RE = re.compile(
    r"import\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s+from\s+['\"](?P<path>\.[^'\"]+)['\"]\s*;"
)
ROUTED_COMPONENT_RE = re.compile(
    r"(?:case\s+['\"][^'\"]+['\"]\s*:\s*return|default\s*:\s*return)\s*(?:\(\s*)?<(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
RETURNED_COMPONENT_RE = re.compile(
    r"\breturn\s*(?:\(\s*)?<(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
ASSIGNED_ROUTE_COMPONENT_RE = re.compile(
    r"\b(?:content|screen|screenContent|currentScreen|renderedScreen|Screen|ScreenComponent)\s*=\s*(?:\(\s*)?<(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
ASSIGNED_ROUTE_COMPONENT_REFERENCE_RE = re.compile(
    r"\b(?:Screen|screen|ScreenComp|screenComp|ScreenComponent|screenComponent|"
    r"CurrentScreen|currentScreen|RenderedScreen|renderedScreen)\s*=\s*"
    r"(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
ASSIGNED_ROUTE_RENDER_FUNCTION_RE = re.compile(
    r"\b(?:Screen|screen|ScreenComp|screenComp|ScreenComponent|screenComponent|"
    r"CurrentScreen|currentScreen|RenderedScreen|renderedScreen)\s*=\s*"
    r"(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>\s*(?:\(\s*)?"
    r"<(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
CONDITIONAL_ROUTE_COMPONENT_RE = re.compile(
    r"\{[^{}]*(?:===|==|!==|!=)[^{}]*&&\s*(?:\(\s*)?<(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
STACK_SCREEN_COMPONENT_RE = re.compile(
    r"<[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\b[^>]*\bcomponent\s*=\s*\{\s*(?P<name>[A-Z][A-Za-z0-9_]*)\s*\}",
    re.MULTILINE,
)
ROUTE_MAP_OBJECT_RE = re.compile(
    r"\b(?:const|let|var)\s+(?:screenRegistry|screens|Screens|SCREENS|routes|Routes|ROUTES|screenMap|routeMap)\s*=\s*\{(?P<body>[\s\S]*?)\}\s*;?",
    re.MULTILINE,
)
ROUTE_MAP_ENTRY_RE = re.compile(
    r"(?:^|[,{\n])\s*(?:['\"][^'\"]+['\"]|[A-Za-z0-9_]+)\s*:\s*(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
SCREEN_COMPONENT_FALLBACK_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_]*Component\s*=\s*[^;]*\|\|\s*(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
JSX_COMPONENT_RE = re.compile(r"<(?P<name>[A-Z][A-Za-z0-9_]*)\b")
FUNCTION_COMPONENT_RE = re.compile(
    r"(?:export\s+default\s+)?function\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s*\((?P<params>[\s\S]*?)\)"
)
ARROW_COMPONENT_RE = re.compile(
    r"const\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s*=\s*\((?P<params>[^;]*?)\)\s*=>"
)
ONPRESS_PROP_NAME_RE = re.compile(r"(?<![\w.])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=")
RETURN_BLOCK_RE = re.compile(r"\breturn\s*\(")
RETURN_JSX_RE = re.compile(r"\breturn\s*<")
KEY_INDEX_EXPR_RE = re.compile(r"(i|idx|index)")
KEY_VALUE_EXPR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
CALLABLE_WRAPPERS = (
    "React.memo",
    "memo",
    "React.forwardRef",
    "forwardRef",
)

HOST_CLICKABLE_COMPONENT_NAMES = {
    "Button",
    "BorderlessButton",
    "Pressable",
    "RectButton",
    "Switch",
    "TouchableHighlight",
    "TouchableNativeFeedback",
    "TouchableOpacity",
    "TouchableWithoutFeedback",
    "Wrapper",
    "Container",
    "Text",
    "View"
}
NATIVE_FALLBACK_COMPONENT_NAMES = {
    "ActivityIndicator",
    "Button",
    "FlatList",
    "Image",
    "ImageBackground",
    "KeyboardAvoidingView",
    "Modal",
    "Pressable",
    "RefreshControl",
    "SafeAreaView",
    "ScrollView",
    "SectionList",
    "StatusBar",
    "Switch",
    "Text",
    "TextInput",
    "TouchableHighlight",
    "TouchableNativeFeedback",
    "TouchableOpacity",
    "TouchableWithoutFeedback",
    "View",
    "VirtualizedList",
}
LIST_RENDER_ITEM_COMPONENT_NAMES = {"FlatList", "SectionList", "VirtualizedList"}

@dataclass
class Section:
    name: str
    code: str

@dataclass
class RenderUnit:
    name: str
    params_start: int
    params_end: int
    params_text: str
    params_has_parens: bool
    body_start: int
    body_end: int
    definition_start: int
    definition_end: int
    host_slots: list[str]
    call_style: str
    key: str = ""
    parent_key: str | None = None
    requires_press_prop: bool = False

@dataclass
class BodyTarget:
    kind: str
    start: int
    end: int
    text: str
    name: str
    host_slots: list[str]
    args_start: int | None = None
    args_end: int | None = None

def apply_replacements(
    text: str,
    replacements: list[tuple[int, int, str]],
    *,
    sort: bool = False,
) -> str:
    if not replacements:
        return text

    output: list[str] = []
    cursor = 0
    ordered_replacements = sorted(replacements, key=lambda item: item[0]) if sort else replacements
    for start, end, replacement in ordered_replacements:
        output.append(text[cursor:start])
        output.append(replacement)
        cursor = end
    output.append(text[cursor:])
    return "".join(output)

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

def deduplicate_sections(sections: list[Section]) -> list[Section]:
    """Keep the last body for duplicate section paths while preserving file order."""
    last_section_by_name = {section.name: section for section in sections}
    unique_sections: list[Section] = []
    seen_names: set[str] = set()
    for section in sections:
        if section.name in seen_names:
            continue
        seen_names.add(section.name)
        unique_sections.append(last_section_by_name[section.name])
    return unique_sections

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

    for named_group in re.finditer(r"\{(?P<inner>[^{}]*)\}", normalized):
        for item in named_group.group("inner").split(","):
            item = item.strip()
            if not item:
                continue
            names.append(re.split(r"\s+as\s+", item)[-1].strip())

    without_named_groups = re.sub(r"\{[^{}]*\}", "", normalized)
    for part in without_named_groups.split(","):
        part = part.strip()
        if part:
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

def resolve_section_name(section_name: str, section_by_name: dict[str, Section]) -> str:
    if section_name in section_by_name:
        return section_name

    basename = posixpath.basename(section_name.replace("\\", "/"))
    if basename in section_by_name:
        return basename

    matches = [
        candidate_name
        for candidate_name in section_by_name
        if posixpath.basename(candidate_name.replace("\\", "/")) == basename
    ]
    if len(matches) == 1:
        return matches[0]

    return section_name

def resolve_import_paths_to_sections(
    local_imports_by_section: dict[str, dict[str, str]],
    section_by_name: dict[str, Section],
) -> dict[str, dict[str, str]]:
    return {
        section_name: {
            component_name: resolve_section_name(import_path, section_by_name)
            for component_name, import_path in imports.items()
        }
        for section_name, imports in local_imports_by_section.items()
    }

def resolve_section_name_set(
    section_names: set[str],
    section_by_name: dict[str, Section],
) -> set[str]:
    return {
        resolve_section_name(section_name, section_by_name)
        for section_name in section_names
    }

def extract_page_section_names(app_code: str) -> set[str]:
    app_imports = {
        match.group("name"): normalize_relative_js_path(match.group("path"))
        for match in APP_IMPORT_RE.finditer(app_code)
    }
    route_map_component_names: set[str] = set()
    for map_match in ROUTE_MAP_OBJECT_RE.finditer(app_code):
        route_map_component_names.update(
            entry_match.group("name")
            for entry_match in ROUTE_MAP_ENTRY_RE.finditer(map_match.group("body"))
        )

    routed_component_names = {
        match.group("name")
        for match in ROUTED_COMPONENT_RE.finditer(app_code)
    }
    routed_component_names.update(
        match.group("name")
        for match in RETURNED_COMPONENT_RE.finditer(app_code)
    )
    routed_component_names.update(
        match.group("name")
        for match in ASSIGNED_ROUTE_COMPONENT_RE.finditer(app_code)
    )
    routed_component_names.update(
        match.group("name")
        for match in ASSIGNED_ROUTE_COMPONENT_REFERENCE_RE.finditer(app_code)
    )
    routed_component_names.update(
        match.group("name")
        for match in ASSIGNED_ROUTE_RENDER_FUNCTION_RE.finditer(app_code)
    )
    routed_component_names.update(
        match.group("name")
        for match in CONDITIONAL_ROUTE_COMPONENT_RE.finditer(app_code)
    )
    routed_component_names.update(
        match.group("name")
        for match in STACK_SCREEN_COMPONENT_RE.finditer(app_code)
    )
    routed_component_names.update(route_map_component_names)
    routed_component_names.update(
        match.group("name")
        for match in SCREEN_COMPONENT_FALLBACK_RE.finditer(app_code)
    )
    routed_component_names &= set(app_imports)
    if not routed_component_names:
        used_imported_component_names = {
            match.group("name")
            for match in JSX_COMPONENT_RE.finditer(app_code)
            if match.group("name") in app_imports
        }
        if used_imported_component_names:
            return {
                import_path
                for component_name, import_path in app_imports.items()
                if component_name in used_imported_component_names
            }
        return set(app_imports.values())
    return {
        import_path
        for component_name, import_path in app_imports.items()
        if component_name in routed_component_names
    }

def extract_navigation_component_names(code: str) -> set[str]:
    component_names: set[str] = set()
    for _, _, tag_text in iter_jsx_start_tags(code):
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or not (
            tag_name.endswith(".Screen")
            or tag_name in {"Screen", "Stack.Screen", "Tab.Screen", "Drawer.Screen"}
        ):
            continue
        component_attr = parse_jsx_attribute_value(tag_text, "component")
        if not component_attr:
            continue
        component_name = component_attr[0].strip()
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", component_name):
            component_names.add(component_name)
    return component_names

def expand_page_section_names(
    initial_page_section_names: set[str],
    local_imports_by_section: dict[str, dict[str, str]],
    section_by_name: dict[str, Section],
) -> set[str]:
    page_section_names = set(initial_page_section_names)
    changed = True
    while changed:
        changed = False
        for section_name in list(page_section_names) + ["App.js"]:
            section = section_by_name.get(section_name)
            if not section:
                continue
            imports = local_imports_by_section.get(section_name, {})
            for component_name in extract_navigation_component_names(section.code):
                import_path = imports.get(component_name)
                if not import_path or import_path in page_section_names:
                    continue
                page_section_names.add(import_path)
                changed = True
    return page_section_names

def extract_used_component_names(code: str) -> set[str]:
    used_component_names: set[str] = set()
    for _, _, tag_text in iter_labelable_jsx_start_tags(code):
        tag_name = get_jsx_tag_name(tag_text)
        if tag_name and re.fullmatch(r"[A-Z][A-Za-z0-9_]*", tag_name):
            used_component_names.add(tag_name)
    return used_component_names

def iter_declared_component_matches(code: str) -> list[re.Match[str]]:
    matches = list(FUNCTION_COMPONENT_RE.finditer(code)) + list(ARROW_COMPONENT_RE.finditer(code))
    return sorted(matches, key=lambda match: match.start())

def skip_ws(source: str, index: int, limit: int | None = None) -> int:
    end = len(source) if limit is None else min(limit, len(source))
    while index < end and source[index].isspace():
        index += 1
    return index

def starts_identifier_at(source: str, index: int, text: str) -> bool:
    if not source.startswith(text, index):
        return False
    before = source[index - 1] if index > 0 else ""
    after_index = index + len(text)
    after = source[after_index] if after_index < len(source) else ""
    if before and re.match(r"[A-Za-z0-9_.$]", before):
        return False
    if after and re.match(r"[A-Za-z0-9_.$]", after):
        return False
    return True

def find_matching_brace(source: str, open_index: int) -> int | None:
    if open_index < 0 or open_index >= len(source) or source[open_index] != "{":
        return None

    depth = 1
    index = open_index + 1
    in_string: str | None = None
    while index < len(source):
        char = source[index]
        prev_char = source[index - 1] if index > open_index + 1 else ""
        if in_string:
            if in_string in {"'", '"'} and char in {"\n", "\r"}:
                in_string = None
                index += 1
                continue
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
                    return index
        index += 1
    return None

def include_trailing_semicolon(code: str, index: int) -> int:
    cursor = index + 1
    while cursor < len(code) and code[cursor].isspace():
        cursor += 1
    if cursor < len(code) and code[cursor] == ";":
        return cursor + 1
    return index + 1

def find_statement_end(
    source: str,
    start_index: int,
    limit: int | None = None,
    *,
    stop_at_newline: bool = True,
) -> int:
    max_index = len(source) if limit is None else min(limit, len(source))
    index = start_index
    depth_parenthesis = 0
    depth_brace = 0
    depth_bracket = 0
    in_string: str | None = None
    while index < max_index:
        char = source[index]
        prev_char = source[index - 1] if index > start_index else ""
        if in_string:
            if in_string in {"'", '"'} and char in {"\n", "\r"}:
                in_string = None
                return index
            if char == in_string and prev_char != "\\":
                in_string = None
        else:
            if char in {"'", '"', "`"}:
                in_string = char
            elif char == "(":
                depth_parenthesis += 1
            elif char == ")" and depth_parenthesis > 0:
                depth_parenthesis -= 1
            elif char == "{":
                depth_brace += 1
            elif char == "}" and depth_brace > 0:
                depth_brace -= 1
            elif char == "[":
                depth_bracket += 1
            elif char == "]" and depth_bracket > 0:
                depth_bracket -= 1
            elif char == ";" and depth_parenthesis == 0 and depth_brace == 0 and depth_bracket == 0:
                return index
            elif (
                stop_at_newline
                and char in {"\n", "\r"}
                and depth_parenthesis == 0
                and depth_brace == 0
                and depth_bracket == 0
            ):
                return index
        index += 1
    return max_index

def find_declared_component_end(code: str, match: re.Match[str]) -> int:
    cursor = match.end()
    while cursor < len(code) and code[cursor].isspace():
        cursor += 1

    if cursor < len(code) and code[cursor] == "{":
        close_index = find_matching_brace(code, cursor)
        if close_index is not None:
            return close_index + 1

    if cursor < len(code) and code[cursor] == "(":
        close_index = find_matching_parenthesis(code, cursor)
        if close_index is not None:
            return include_trailing_semicolon(code, close_index)

    line_end = code.find("\n", cursor)
    semicolon = code.find(";", cursor)
    if semicolon != -1 and (line_end == -1 or semicolon < line_end):
        return semicolon + 1
    return line_end if line_end != -1 else len(code)

def extract_declared_component_ranges(code: str) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for match in iter_declared_component_matches(code):
        ranges[match.group("name")] = (match.start(), find_declared_component_end(code, match))
    return ranges

def get_jsx_tag_name(tag_text: str) -> str | None:
    name_match = re.match(r"<\s*([A-Za-z_][A-Za-z0-9_.-]*)", tag_text)
    return name_match.group(1) if name_match else None

def is_native_clickable_host_tag(tag_text: str) -> bool:
    tag_name = get_jsx_tag_name(tag_text)
    return bool(
        tag_name
        and tag_name in HOST_CLICKABLE_COMPONENT_NAMES
        and has_meaningful_press_prop(tag_text)
    )

def is_clickable_host_tag(tag_text: str) -> bool:
    return is_native_clickable_host_tag(tag_text)

def extract_clickable_host_alias_names(source: str) -> set[str]:
    aliases: set[str] = set()
    host_names_re = "|".join(re.escape(name) for name in sorted(HOST_CLICKABLE_COMPONENT_NAMES))
    assignment_re = re.compile(
        rf"\b(?:const|let|var)\s+(?P<name>[A-Z][A-Za-z0-9_]*)\s*=\s*(?P<expr>[^;\n]*\b(?:{host_names_re})\b[^;\n]*)"
    )
    for match in assignment_re.finditer(source):
        aliases.add(match.group("name"))
    return aliases

def is_custom_component_tag_name(tag_name: str) -> bool:
    """Return whether a JSX tag can be a user-defined component or local alias."""
    root_name = tag_name.split(".", 1)[0]
    return bool(root_name[:1].isupper() and tag_name not in NATIVE_FALLBACK_COMPONENT_NAMES)

def extract_effective_component_interaction_host_types(code: str) -> dict[str, list[str]]:
    return extract_effective_component_interaction_slots(code)

def unwrap_callable_wrappers(source: str, index: int, limit: int) -> int:
    changed = True
    while changed:
        changed = False
        index = skip_ws(source, index, limit)
        for wrapper in CALLABLE_WRAPPERS:
            if starts_identifier_at(source, index, wrapper):
                cursor = skip_ws(source, index + len(wrapper), limit)
                if cursor < limit and source[cursor] == "(":
                    index = skip_ws(source, cursor + 1, limit)
                    changed = True
                    break
    return index

def find_render_expression_range(source: str, expression_start: int, expression_limit: int) -> tuple[int, int] | None:
    cursor = skip_ws(source, expression_start, expression_limit)
    if cursor >= expression_limit:
        return None

    if cursor < expression_limit and source[cursor] == "(":
        close_index = find_matching_parenthesis(source, cursor)
        if close_index is not None and close_index <= expression_limit:
            return cursor + 1, close_index

    if cursor < expression_limit and source[cursor] == "<":
        body_end = find_statement_end(source, cursor, expression_limit, stop_at_newline=False)
        return cursor, body_end

    if cursor < expression_limit and source[cursor] == "{":
        close_index = find_matching_brace(source, cursor)
        if close_index is None or close_index > expression_limit:
            return None
        block_start = cursor + 1
        block_end = close_index
        for return_start in find_top_level_return_expression_starts(source, block_start, block_end):
            jsx_start = skip_ws(source, return_start, block_end)
            if source.startswith(("null", "undefined", "false"), jsx_start):
                continue
            if jsx_start >= block_end or source[jsx_start] not in {"(", "<"}:
                continue
            jsx_end = find_statement_end(source, jsx_start, block_end, stop_at_newline=False)
            if jsx_start < jsx_end:
                return jsx_start, jsx_end

    return None

def parse_arrow_body_range(source: str, index: int, limit: int) -> tuple[int, int] | None:
    index = skip_ws(source, index, limit)
    if not source.startswith("=>", index):
        return None
    return find_render_expression_range(source, index + 2, limit)

def find_top_level_return_expression_starts(source: str, block_start: int, block_end: int) -> list[int]:
    starts: list[int] = []
    index = block_start
    depth_parenthesis = 0
    depth_brace = 0
    depth_bracket = 0
    in_string: str | None = None
    while index < block_end:
        char = source[index]
        prev_char = source[index - 1] if index > block_start else ""
        if in_string:
            if in_string in {"'", '"'} and char in {"\n", "\r"}:
                in_string = None
            elif char == in_string and prev_char != "\\":
                in_string = None
        else:
            if (
                depth_parenthesis == 0
                and depth_brace == 0
                and depth_bracket == 0
                and starts_identifier_at(source, index, "return")
            ):
                starts.append(index + len("return"))
                index += len("return")
                continue
            if char in {"'", '"', "`"}:
                in_string = char
            elif char == "(":
                depth_parenthesis += 1
            elif char == ")" and depth_parenthesis > 0:
                depth_parenthesis -= 1
            elif char == "{":
                depth_brace += 1
            elif char == "}" and depth_brace > 0:
                depth_brace -= 1
            elif char == "[":
                depth_bracket += 1
            elif char == "]" and depth_bracket > 0:
                depth_bracket -= 1
        index += 1
    return starts

def collect_render_unit_candidates(code: str) -> list[RenderUnit]:
    units: list[RenderUnit] = []
    function_decl_re = re.compile(
        r"\b(?:export\s+default\s+|export\s+)?function\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    var_assign_re = re.compile(
        r"\b(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    )
    identifier_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    for match in function_decl_re.finditer(code):
        name = match.group("name")
        open_index = match.end() - 1
        close_index = find_matching_parenthesis(code, open_index)
        if close_index is None:
            continue
        cursor = skip_ws(code, close_index + 1)
        if cursor >= len(code) or code[cursor] != "{":
            continue
        block_close = find_matching_brace(code, cursor)
        if block_close is None:
            continue
        render_range = find_render_expression_range(code, cursor, block_close + 1)
        if render_range is None:
            continue
        body_start, body_end = render_range
        units.append(RenderUnit(
            name=name,
            params_start=open_index + 1,
            params_end=close_index,
            params_text=code[open_index + 1:close_index],
            params_has_parens=True,
            body_start=body_start,
            body_end=body_end,
            definition_start=match.start(),
            definition_end=block_close + 1,
            host_slots=[],
            call_style="jsx" if name[:1].isupper() else "function",
        ))

    for match in var_assign_re.finditer(code):
        name = match.group("name")
        definition_start = match.start()
        definition_end = find_statement_end(code, definition_start)
        cursor = unwrap_callable_wrappers(code, match.end(), definition_end)
        cursor = skip_ws(code, cursor, definition_end)

        if starts_identifier_at(code, cursor, "async"):
            cursor = skip_ws(code, cursor + len("async"), definition_end)

        if starts_identifier_at(code, cursor, "function"):
            cursor = skip_ws(code, cursor + len("function"), definition_end)
            optional_name = identifier_re.match(code[cursor:])
            if optional_name:
                cursor = skip_ws(code, cursor + optional_name.end(), definition_end)
            if cursor >= definition_end or code[cursor] != "(":
                continue
            open_index = cursor
            close_index = find_matching_parenthesis(code, open_index)
            if close_index is None:
                continue
            block_cursor = skip_ws(code, close_index + 1, definition_end)
            if block_cursor >= definition_end or code[block_cursor] != "{":
                continue
            block_close = find_matching_brace(code, block_cursor)
            if block_close is None:
                continue
            render_range = find_render_expression_range(code, block_cursor, min(block_close + 1, definition_end))
            if render_range is None:
                continue
            body_start, body_end = render_range
            units.append(RenderUnit(
                name=name,
                params_start=open_index + 1,
                params_end=close_index,
                params_text=code[open_index + 1:close_index],
                params_has_parens=True,
                body_start=body_start,
                body_end=body_end,
                definition_start=definition_start,
                definition_end=definition_end,
                host_slots=[],
                call_style="jsx" if name[:1].isupper() else "function",
            ))
            continue

        if cursor < definition_end and code[cursor] == "(":
            open_index = cursor
            close_index = find_matching_parenthesis(code, open_index)
            if close_index is None:
                continue
            body_range = parse_arrow_body_range(code, close_index + 1, definition_end)
            if body_range is None:
                continue
            body_start, body_end = body_range
            units.append(RenderUnit(
                name=name,
                params_start=open_index + 1,
                params_end=close_index,
                params_text=code[open_index + 1:close_index],
                params_has_parens=True,
                body_start=body_start,
                body_end=body_end,
                definition_start=definition_start,
                definition_end=definition_end,
                host_slots=[],
                call_style="jsx" if name[:1].isupper() else "function",
            ))
            continue

        param_match = identifier_re.match(code[cursor:])
        if param_match:
            params_start = cursor
            params_end = cursor + param_match.end()
            body_range = parse_arrow_body_range(code, params_end, definition_end)
            if body_range is None:
                continue
            body_start, body_end = body_range
            units.append(RenderUnit(
                name=name,
                params_start=params_start,
                params_end=params_end,
                params_text=code[params_start:params_end],
                params_has_parens=False,
                body_start=body_start,
                body_end=body_end,
                definition_start=definition_start,
                definition_end=definition_end,
                host_slots=[],
                call_style="jsx" if name[:1].isupper() else "function",
            ))

    return sorted(units, key=lambda item: item.definition_start)

def assign_render_unit_scope(units: list[RenderUnit]) -> None:
    for unit in units:
        unit.key = f"{unit.name}@{unit.definition_start}"
        unit.parent_key = None

    for unit in units:
        containing = [
            candidate
            for candidate in units
            if candidate is not unit
            and candidate.definition_start < unit.definition_start
            and unit.definition_end <= candidate.definition_end
        ]
        if containing:
            parent = min(containing, key=lambda candidate: candidate.definition_end - candidate.definition_start)
            unit.parent_key = parent.key

def ancestor_render_unit_keys(unit: RenderUnit, units_by_key: dict[str, RenderUnit]) -> set[str]:
    ancestors: set[str] = set()
    parent_key = unit.parent_key
    while parent_key and parent_key not in ancestors:
        ancestors.add(parent_key)
        parent = units_by_key.get(parent_key)
        parent_key = parent.parent_key if parent else None
    return ancestors

def build_visible_render_unit_slots(
    units: list[RenderUnit],
    slots_by_key: dict[str, list[str]],
    current_unit: RenderUnit,
) -> dict[str, list[str]]:
    """Return local render units visible from current_unit's JSX body.

    Nested helper components may reuse short names like `Item` in different
    parent components. Visibility is lexical, so the closest same-name helper
    in the current scope should override unrelated helpers from another scope.
    """
    units_by_key = {unit.key: unit for unit in units}
    ancestors = ancestor_render_unit_keys(current_unit, units_by_key)
    visible: dict[str, list[str]] = {}

    def add_if_slots(unit: RenderUnit) -> None:
        if unit.key == current_unit.key:
            return
        slots = slots_by_key.get(unit.key, [])
        if slots:
            visible[unit.name] = slots

    # Top-level helpers are visible everywhere in the file.
    for unit in units:
        if unit.parent_key is None:
            add_if_slots(unit)

    # Helpers declared in ancestor scopes are visible to nested helpers.
    for unit in units:
        if unit.parent_key in ancestors:
            add_if_slots(unit)

    # Sibling helpers in the same lexical scope are visible.
    for unit in units:
        if unit.parent_key == current_unit.parent_key:
            add_if_slots(unit)

    # Helpers declared inside the current unit are visible from its return body.
    # Add this last so a local `Item` overrides any top-level `Item`.
    for unit in units:
        if unit.parent_key == current_unit.key:
            add_if_slots(unit)

    return visible

def collect_body_targets(
    source: str,
    component_host_slots: dict[str, list[str]],
    function_host_slots: dict[str, list[str]] | None = None,
    clickable_host_alias_names: set[str] | None = None,
    press_required_component_names: set[str] | None = None,
) -> list[BodyTarget]:
    function_host_slots = function_host_slots or {}
    clickable_host_alias_names = set(clickable_host_alias_names or set()) | extract_clickable_host_alias_names(source)
    press_required_component_names = press_required_component_names or set()
    targets: list[BodyTarget] = []
    jsx_ranges: list[tuple[int, int]] = []

    for start, end, tag_text in iter_jsx_start_tags(source):
        jsx_ranges.append((start, end))
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name:
            continue
        host_slots = component_host_slots.get(tag_name) or function_host_slots.get(tag_name)
        if host_slots:
            if tag_name in press_required_component_names and not has_meaningful_press_prop(tag_text):
                continue
            targets.append(BodyTarget("jsx-component", start, end, tag_text, tag_name, host_slots))
            continue
        if tag_name in clickable_host_alias_names and has_meaningful_press_prop(tag_text):
            targets.append(BodyTarget("jsx-host", start, end, tag_text, tag_name, [tag_name]))
            continue
        if is_clickable_host_tag(tag_text):
            targets.append(BodyTarget("jsx-host", start, end, tag_text, tag_name, [tag_name]))

    for function_name, host_slots in function_host_slots.items():
        if not host_slots:
            continue
        call_re = re.compile(rf"(?<![\w.]){re.escape(function_name)}\s*\(")
        for match in call_re.finditer(source):
            if position_in_ranges(match.start(), jsx_ranges):
                continue
            open_index = match.end() - 1
            close_index = find_matching_parenthesis(source, open_index)
            if close_index is None:
                continue
            targets.append(BodyTarget(
                "function-call",
                match.start(),
                close_index + 1,
                source[match.start():close_index + 1],
                function_name,
                host_slots,
                args_start=open_index + 1,
                args_end=close_index,
            ))

    return sorted(targets, key=lambda target: target.start)

def collect_render_units(
    code: str,
    component_host_slots: dict[str, list[str]],
    *,
    exclude_names: set[str] | None = None,
) -> list[RenderUnit]:
    exclude_names = exclude_names or set()
    candidates = [
        unit for unit in collect_render_unit_candidates(code)
        if unit.name not in exclude_names
    ]
    assign_render_unit_scope(candidates)
    for unit in candidates:
        unit.requires_press_prop = bool(re.search(r"\bon[A-Za-z0-9_]*Press\b|\bonPress\b", unit.params_text))
    slots_by_key: dict[str, list[str]] = {unit.key: [] for unit in candidates}

    changed = True
    iteration = 0
    max_iterations = max(4, len(candidates) * 4)
    while changed:
        iteration += 1
        if iteration > max_iterations:
            print(
                "[label] warning: collect_render_units did not converge; "
                f"using last stable approximation after {max_iterations} iterations"
            )
            break
        changed = False

        for unit in candidates:
            visible_local_slots = build_visible_render_unit_slots(candidates, slots_by_key, unit)
            active_component_slots = {**component_host_slots, **visible_local_slots}
            active_function_slots = visible_local_slots
            press_required_component_names = {
                visible_unit.name
                for visible_unit in candidates
                if visible_unit.requires_press_prop
            }
            body = code[unit.body_start:unit.body_end]
            unit_source = code[unit.definition_start:unit.definition_end]
            body_targets = collect_body_targets(
                body,
                active_component_slots,
                active_function_slots,
                extract_clickable_host_alias_names(unit_source),
                press_required_component_names,
            )
            slots: list[str] = []
            for target in body_targets:
                slots.extend(target.host_slots)
            if slots != slots_by_key.get(unit.key, []):
                slots_by_key[unit.key] = slots
                changed = True

    units: list[RenderUnit] = []
    for unit in candidates:
        slots = slots_by_key.get(unit.key, [])
        if not slots:
            continue
        unit.host_slots = slots
        units.append(unit)
    return sorted(units, key=lambda unit: unit.definition_start)

def extract_effective_component_interaction_slots(code: str) -> dict[str, list[str]]:
    units = collect_render_units(code, {}, exclude_names=set())
    return {
        unit.name: unit.host_slots
        for unit in units
        if unit.name[:1].isupper() and unit.parent_key is None
    }

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
            if start_char in {"'", '"'} and tag_text[index] in {"\n", "\r"}:
                return None
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
                if in_string in {"'", '"'} and char in {"\n", "\r"}:
                    in_string = None
                    index += 1
                    continue
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

LABEL_KEY_PROP_NAMES = ("label", "title", "text", "name")

def normalize_jsx_attribute_key_expr(
    tag_text: str,
    attribute_name: str,
    *,
    allow_static_literal: bool = False,
) -> str | None:
    attr = parse_jsx_attribute_value(tag_text, attribute_name)
    if not attr:
        return None

    value_text, value_start, _ = attr
    expr = value_text.strip()
    if not expr:
        return None

    value_prefix = tag_text[value_start]
    if value_prefix == "{":
        return expr
    if allow_static_literal and value_prefix in {"'", '"'}:
        return json.dumps(value_text, ensure_ascii=False)
    return None

def get_jsx_key_or_label_expr(tag_text: str) -> str | None:
    key_expr = normalize_jsx_attribute_key_expr(
        tag_text,
        "key",
        allow_static_literal=True,
    )
    if key_expr:
        return key_expr

    for prop_name in LABEL_KEY_PROP_NAMES:
        prop_expr = normalize_jsx_attribute_key_expr(tag_text, prop_name)
        if prop_expr and KEY_VALUE_EXPR_RE.fullmatch(prop_expr):
            return prop_expr

    return None

def find_enclosing_map_key_expr(code: str, position: int) -> str | None:
    best_map_start = -1
    best_open_index: int | None = None
    best_close_index: int | None = None

    for match in re.finditer(r"\.map\s*\(", code):
        open_index = code.find("(", match.start())
        if open_index < 0 or open_index > position:
            continue
        close_index = find_matching_parenthesis(code, open_index)
        if close_index is None or not (open_index < position < close_index):
            continue
        if match.start() > best_map_start:
            best_map_start = match.start()
            best_open_index = open_index
            best_close_index = close_index

    if best_open_index is None or best_close_index is None:
        return None

    key_expr: str | None = None
    callback_prefix = code[best_open_index + 1:position]
    for _, _, tag_text in iter_jsx_start_tags(callback_prefix):
        tag_key_expr = get_jsx_key_or_label_expr(tag_text)
        if tag_key_expr:
            key_expr = tag_key_expr
    return key_expr

def find_jsx_attribute_insert_position(tag_text: str) -> int:
    fallback = len(tag_text) - 2 if tag_text.endswith("/>") else len(tag_text) - 1
    depth = 0
    in_string: str | None = None
    index = 0
    while index < fallback:
        char = tag_text[index]
        next_char = tag_text[index + 1] if index + 1 < fallback else ""
        prev_char = tag_text[index - 1] if index > 0 else ""
        if in_string:
            if char == in_string and prev_char != "\\":
                in_string = None
        else:
            if char in {"'", '"', "`"}:
                in_string = char
            elif char == "{":
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
            elif char == "/" and next_char == "/" and depth == 0:
                while index > 0 and tag_text[index - 1] in " \t":
                    index -= 1
                return index
        index += 1
    return fallback

def set_or_replace_attribute(tag_text: str, attribute_name: str, attribute_value: str) -> str:
    existing_value = parse_jsx_attribute_value(tag_text, attribute_name)
    if existing_value:
        _, value_start, value_end = existing_value
        name_match = re.search(rf"(?<![\w.]){re.escape(attribute_name)}\s*=", tag_text)
        if name_match:
            return tag_text[:name_match.start()] + f"{attribute_name}={attribute_value}" + tag_text[value_end:]

    insert_at = find_jsx_attribute_insert_position(tag_text)
    prefix = "" if tag_text[max(0, insert_at - 1):insert_at].isspace() else " "
    return tag_text[:insert_at] + f"{prefix}{attribute_name}={attribute_value}" + tag_text[insert_at:]

def remove_jsx_attribute(tag_text: str, attribute_name: str) -> str:
    attr_value = parse_jsx_attribute_value(tag_text, attribute_name)
    if not attr_value:
        return tag_text
    _, value_start, value_end = attr_value
    name_match = re.search(rf"(?<![\w.]){re.escape(attribute_name)}\s*=", tag_text)
    if not name_match:
        return tag_text
    start = name_match.start()
    while start > 0 and tag_text[start - 1] in " \t":
        start -= 1
    return tag_text[:start] + tag_text[value_end:]

def set_generated_interaction_labels(tag_text: str, label_bag: str) -> str:
    updated = remove_jsx_attribute(tag_text, "testID")
    updated = remove_jsx_attribute(updated, "accessibilityLabel")
    return set_or_replace_attribute(updated, "interactionLabels", label_bag)

def set_native_interaction_label_attributes(
    tag_text: str,
    test_id_value: str,
    accessibility_value: str,
) -> str:
    updated_tag = set_or_replace_attribute(tag_text, "testID", test_id_value)
    return set_or_replace_attribute(updated_tag, "accessibilityLabel", accessibility_value)

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
        if source[start + 1].isspace() or not re.match(r"[A-Za-z]", source[start + 1]):
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

def add_labels_to_embedded_jsx_attribute_components(
    code: str,
    page_name: str,
    jsx_range: tuple[int, int] | None,
) -> tuple[str, int]:
    """Label interactive JSX nodes used inside another component's props."""
    if jsx_range is None:
        return code, 0

    range_start, range_end = jsx_range
    source = code[range_start:range_end]
    outer_tags = iter_jsx_start_tags(source)

    def without_embedded_jsx(tag_text: str) -> str:
        embedded_ranges = [
            (start + 1, end + 1, "")
            for start, end, _ in iter_jsx_start_tags(tag_text[1:])
        ]
        return apply_replacements(tag_text, embedded_ranges, sort=True)

    counters: dict[str, int] = {}
    for _, _, tag_text in outer_tags:
        if not is_clickable_host_tag(without_embedded_jsx(tag_text)):
            continue
        tag_name = get_jsx_tag_name(tag_text)
        if tag_name:
            counters[tag_name] = counters.get(tag_name, 0) + 1

    labeled_count = 0

    def label_embedded_tag(tag_text: str) -> str:
        nonlocal labeled_count
        updated_tag = tag_text
        tag_name = get_jsx_tag_name(updated_tag)
        if tag_name and is_clickable_host_tag(without_embedded_jsx(updated_tag)):
            key_expr = get_jsx_key_or_label_expr(updated_tag)
            if key_expr:
                label_value = build_static_key_label(page_name, tag_name, key_expr)
            else:
                counters[tag_name] = counters.get(tag_name, 0) + 1
                label_value = build_static_label(page_name, tag_name, counters[tag_name])
            updated_tag = set_native_interaction_label_attributes(
                updated_tag,
                label_value,
                label_value,
            )
            labeled_count += 1

        nested_replacements: list[tuple[int, int, str]] = []
        for nested_start, nested_end, nested_tag in iter_jsx_start_tags(updated_tag[1:]):
            transformed = label_embedded_tag(nested_tag)
            if transformed != nested_tag:
                nested_replacements.append((nested_start + 1, nested_end + 1, transformed))
        return apply_replacements(updated_tag, nested_replacements, sort=True)

    replacements: list[tuple[int, int, str]] = []
    for start, end, outer_tag in outer_tags:
        embedded_replacements: list[tuple[int, int, str]] = []
        for embedded_start, embedded_end, embedded_tag in iter_jsx_start_tags(outer_tag[1:]):
            transformed = label_embedded_tag(embedded_tag)
            if transformed != embedded_tag:
                embedded_replacements.append((embedded_start + 1, embedded_end + 1, transformed))
        if embedded_replacements:
            replacements.append(
                (start, end, apply_replacements(outer_tag, embedded_replacements, sort=True))
            )

    if not replacements:
        return code, 0
    updated_source = apply_replacements(source, replacements, sort=True)
    return code[:range_start] + updated_source + code[range_end:], labeled_count

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
            if in_string in {"'", '"'} and char in {"\n", "\r"}:
                in_string = None
                index += 1
                continue
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
    for match in RETURN_JSX_RE.finditer(code):
        body_start = match.end() - 1
        body_end = find_statement_end(code, body_start, stop_at_newline=False)
        if body_start < body_end:
            ranges.append((body_start, body_end))

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
        if is_press_callback_prop_name(match.group("name"))
    }
    for prop_name in press_prop_names:
        prop_value = parse_jsx_attribute_value(tag_text, prop_name)
        if prop_value and is_meaningful_onpress_expression(prop_value[0]):
            return True
    return False

def is_press_callback_prop_name(prop_name: str) -> bool:
    normalized = prop_name.lower()
    return (
        normalized == "onpress"
        or normalized == "onvaluechange"
        or normalized.endswith("onpress")
        or (normalized.startswith("on") and normalized.endswith("press"))
    )

def is_tab_screen_tag_name(tag_name: str) -> bool:
    return tag_name == "Tab.Screen"

def find_page_main_jsx_range(code: str, page_name: str) -> tuple[int, int] | None:
    page_unit: RenderUnit | None = None
    candidates = collect_render_unit_candidates(code)
    for unit in candidates:
        if unit.name == page_name:
            page_unit = unit
            break

    if page_unit is None:
        uppercase_units = [unit for unit in candidates if unit.name[:1].isupper()]
        if uppercase_units:
            page_unit = uppercase_units[-1]

    if page_unit is None:
        return None
    return page_unit.body_start, page_unit.body_end

def add_root_container_label_if_needed(
    code: str,
    page_name: str,
    skipped_component_names: set[str] | None = None,
) -> str:
    main_range = find_page_main_jsx_range(code, page_name)
    if main_range is None:
        return code
    range_start, range_end = main_range
    slice_text = code[range_start:range_end]
    skipped_component_names = skipped_component_names or set()

    for start, end, tag_text in iter_jsx_start_tags(slice_text):
        absolute_start = range_start + start
        absolute_end = range_start + end
        name_match = re.match(r"<\s*([A-Za-z_][A-Za-z0-9_.-]*)", tag_text)
        if not name_match:
            continue
        tag_name = name_match.group(1)
        if is_tab_screen_tag_name(tag_name):
            continue
        if tag_name in skipped_component_names:
            continue
        if tag_name not in NATIVE_FALLBACK_COMPONENT_NAMES:
            continue

        label_value = build_static_label(page_name, tag_name, 1)
        updated_tag = set_native_interaction_label_attributes(tag_text, label_value, label_value)
        return code[:absolute_start] + updated_tag + code[absolute_end:]
    return code

def add_labels_to_clickable_components(
    code: str,
    prefix_builder,
    key_label_builder=None,
    skipped_component_names: set[str] | None = None,
    skipped_ranges: list[tuple[int, int]] | None = None,
) -> tuple[str, int]:
    replacements: list[tuple[int, int, str]] = []
    counters: dict[str, int] = {}
    skipped_component_names = skipped_component_names or set()
    skipped_ranges = skipped_ranges or []

    for start, end, tag_text in iter_labelable_jsx_start_tags(code):
        if any(range_start <= start < range_end for range_start, range_end in skipped_ranges):
            continue
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or tag_name in skipped_component_names:
            continue
        if not is_clickable_host_tag(tag_text):
            continue

        existing_index = get_existing_static_label_index_from_builder(tag_text, prefix_builder, tag_name)
        if existing_index is not None:
            counters[tag_name] = max(counters.get(tag_name, 0), existing_index)
            continue
 
        key_expr = get_jsx_key_or_label_expr(tag_text) or find_enclosing_map_key_expr(code, start)
        if key_expr:
            if key_label_builder:
                label_value = key_label_builder(tag_name, key_expr)
            elif is_key_index_expr(key_expr):
                label_value = (
                    "{"
                    f"`$"+"{pageName}."
                    f"{tag_name}.{build_key_suffix(key_expr)}`"
                    "}"
                )
            else:
                label_value = (
                    "{"
                    f"`$"+"{pageName}."
                    f"{tag_name}.{build_key_suffix(key_expr)}`"
                    "}"
                )
        else:
            counters[tag_name] = counters.get(tag_name, 0) + 1
            sequence = counters[tag_name]

            label_value = prefix_builder(
                tag_name,
                sequence
            )
        updated_tag = set_native_interaction_label_attributes(tag_text, label_value, label_value)
        replacements.append((start, end, updated_tag))

    if not replacements:
        return code, 0

    return apply_replacements(code, replacements), len(replacements)

def build_interaction_labels_object(entries: list[str], *, jsx_attribute: bool) -> str:
    if jsx_attribute:
        return "{{ " + ", ".join(entries) + " }}"
    return "{ " + ", ".join(entries) + " }"

def build_static_label_base(page_name: str, component_name: str, index: int | str) -> str:
    return f"{page_name}.{component_name}.{index}"

def get_existing_static_label_index(tag_text: str, page_name: str, component_name: str) -> int | None:
    label_re = re.compile(
        rf"^{re.escape(page_name)}\.{re.escape(component_name)}\.(?P<index>\d+)(?:\.|$)"
    )
    for attr_name in ("testID", "accessibilityLabel"):
        attr_value = parse_jsx_attribute_value(tag_text, attr_name)
        if not attr_value:
            continue
        raw_value = attr_value[0].strip()
        match = label_re.match(raw_value)
        if match:
            return int(match.group("index"))
    return None

def get_existing_static_label_index_from_builder(tag_text: str, prefix_builder, component_name: str) -> int | None:
    def normalize_label_literal(value: str) -> str:
        text = str(value or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
            return text[1:-1]
        return text

    for attr_name in ("testID", "accessibilityLabel"):
        attr_value = parse_jsx_attribute_value(tag_text, attr_name)
        if not attr_value:
            continue
        raw_value = attr_value[0].strip()
        for index in range(1, 10000):
            try:
                expected_value = prefix_builder(component_name, index)
            except Exception:
                return None
            if normalize_label_literal(expected_value) == raw_value:
                return index
            if not isinstance(expected_value, str) or not expected_value.startswith('"'):
                return None
            if index > 1 and f".{index}" not in expected_value:
                return None
    return None

def build_label_bag_value(
    page_name: str,
    component_name: str,
    host_slots: list[str],
    call_index: int,
    key_expr: str | None = None,
    *,
    jsx_attribute: bool,
) -> str:
    entries: list[str] = []
    for slot_index, _ in enumerate(host_slots, start=1):
        index_value = str(call_index) if len(host_slots) == 1 else f"{call_index}.{slot_index}"
        base = build_static_label_base(page_name, component_name, index_value)
        if key_expr:
            entries.append(f"{slot_index}: `{base}.${{{key_expr.strip()}}}`")
        else:
            entries.append(f'{slot_index}: "{base}"')
    return build_interaction_labels_object(entries, jsx_attribute=jsx_attribute)

def build_label_bag_slice_value(
    start_slot: int,
    host_slots: list[str],
    source_expr: str,
    key_expr: str | None = None,
    *,
    jsx_attribute: bool,
) -> str:
    entries: list[str] = []
    for offset, _ in enumerate(host_slots):
        slot = start_slot + offset
        label_expr = f"{source_expr}?.[{slot}]"
        if key_expr:
            label_expr = build_forwarded_label_expr(label_expr, key_expr)
        entries.append(f"{offset + 1}: {label_expr}")
    return build_interaction_labels_object(entries, jsx_attribute=jsx_attribute)

def find_first_argument_end(args_text: str) -> int | None:
    index = 0
    depth_parenthesis = 0
    depth_brace = 0
    depth_bracket = 0
    in_string: str | None = None
    while index < len(args_text):
        char = args_text[index]
        prev_char = args_text[index - 1] if index > 0 else ""
        if in_string:
            if in_string in {"'", '"'} and char in {"\n", "\r"}:
                in_string = None
            elif char == in_string and prev_char != "\\":
                in_string = None
        else:
            if char in {"'", '"', "`"}:
                in_string = char
            elif char == "(":
                depth_parenthesis += 1
            elif char == ")" and depth_parenthesis > 0:
                depth_parenthesis -= 1
            elif char == "{":
                depth_brace += 1
            elif char == "}" and depth_brace > 0:
                depth_brace -= 1
            elif char == "[":
                depth_bracket += 1
            elif char == "]" and depth_bracket > 0:
                depth_bracket -= 1
            elif char == "," and depth_parenthesis == 0 and depth_brace == 0 and depth_bracket == 0:
                return index
        index += 1
    return len(args_text) if args_text.strip() else None

def looks_like_label_argument(argument_text: str) -> bool:
    text = argument_text.strip()
    if not text:
        return False
    if re.search(r"\binteractionLabels\b|\btestID\b|\baccessibilityLabel\b", text):
        return True
    if text.startswith("{") and re.search(r"(?:^|[{,])\s*\d+\s*:", text):
        return True
    if re.fullmatch(r"[`'\"][A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*\.[^`'\"]+[`'\"]", text):
        return True
    return False

def replace_or_insert_leading_label_bag(args_text: str, label_bag: str) -> str:
    stripped = args_text.strip()
    if not stripped:
        return label_bag

    leading_ws_len = len(args_text) - len(args_text.lstrip())
    trailing = args_text[leading_ws_len:]
    first_end = find_first_argument_end(trailing)
    if first_end is None:
        return label_bag

    first_arg = trailing[:first_end]
    rest = trailing[first_end:]
    if looks_like_label_argument(first_arg):
        if rest.startswith(","):
            second_text = rest[1:].lstrip()
            second_end = find_first_argument_end(second_text)
            if second_end is not None and looks_like_label_argument(second_text[:second_end]):
                rest_after_second = second_text[second_end:]
                rest = rest_after_second if rest_after_second.startswith(",") else (
                    ", " + rest_after_second.lstrip() if rest_after_second.strip() else ""
                )
        return label_bag + rest
    return label_bag + ", " + stripped

def replace_or_insert_leading_label_args(args_text: str, additions: list[str]) -> str:
    stripped = args_text.strip()
    prefix = ", ".join(additions)
    if not stripped:
        return prefix

    leading_ws_len = len(args_text) - len(args_text.lstrip())
    trailing = args_text[leading_ws_len:]
    first_end = find_first_argument_end(trailing)
    if first_end is None:
        return prefix

    first_arg = trailing[:first_end]
    rest = trailing[first_end:]
    if looks_like_label_argument(first_arg):
        if rest.startswith(","):
            second_text = rest[1:].lstrip()
            second_end = find_first_argument_end(second_text)
            if second_end is not None and looks_like_label_argument(second_text[:second_end]):
                rest_after_second = second_text[second_end:]
                rest = rest_after_second if rest_after_second.startswith(",") else (
                    ", " + rest_after_second.lstrip() if rest_after_second.strip() else ""
                )
        return prefix + rest
    return prefix + ", " + stripped
def build_component_reference_wrapper(
    component_name: str,
    host_types: list[str],
    label_value: str,
    *,
    accessibility_label_value: str | None = None,
    props_name: str = "props",
) -> str:
    if len(host_types) == 1:
        accessibility_label_value = accessibility_label_value or label_value
        return (
            "{("
            + props_name
            + ") => <"
            + component_name
            + " {..."
            + props_name
            + "} testID="
            + label_value
            + " accessibilityLabel="
            + accessibility_label_value
            + "/>}"
        )
    return (
        "{("
        + props_name
        + ") => <"
        + component_name
        + " {..."
        + props_name
        + "} interactionLabels="
        + label_value
        + "/>}"
    )

def inject_label_props_into_component_references(
    code: str,
    component_host_types: dict[str, list[str]],
    label_value_builder,
) -> tuple[str, int]:
    if not component_host_types:
        return code, 0

    replacements: list[tuple[int, int, str]] = []
    counters: dict[str, int] = {}
    for start, end, tag_text in iter_labelable_jsx_start_tags(code):
        for attr_match in re.finditer(r"(?<![\w.])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=", tag_text):
            attr_name = attr_match.group("name")
            if attr_name in {"testID", "accessibilityLabel", "interactionLabels"}:
                continue
            attr_value = parse_jsx_attribute_value(tag_text, attr_name)
            if not attr_value:
                continue
            value_text, value_start, value_end = attr_value
            component_name = value_text.strip()
            host_types = component_host_types.get(component_name)
            if not host_types:
                continue
            counters[component_name] = counters.get(component_name, 0) + 1
            label_value = label_value_builder(component_name, host_types, counters[component_name])
            accessibility_label_value = None
            if isinstance(label_value, tuple):
                label_value, accessibility_label_value = label_value
            replacement = build_component_reference_wrapper(
                component_name,
                host_types,
                label_value,
                accessibility_label_value=accessibility_label_value,
            )
            replacements.append((start + value_start, start + value_end, replacement))

    if not replacements:
        return code, 0

    return apply_replacements(code, replacements, sort=True), len(replacements)

def position_in_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)

def find_enclosing_component_name(
    position: int,
    component_ranges: dict[str, tuple[int, int]],
    fallback_name: str,
) -> str:
    containing = [
        (name, start, end)
        for name, (start, end) in component_ranges.items()
        if start <= position < end
    ]
    if not containing:
        return fallback_name
    name, _, _ = max(containing, key=lambda item: item[1])
    return name

def build_forwarded_label_attr_value(base_expr: str, key_expr: str | None = None) -> str:
    return "{" + build_forwarded_label_expr(base_expr, key_expr) + "}"

def build_forwarded_label_expr(base_expr: str, key_expr: str | None = None) -> str:
    if not key_expr:
        return base_expr
    return f"{base_expr} ? `$" + "{" + base_expr + "}." + "${" + key_expr.strip() + "}` : undefined"

def add_interaction_labels_to_params(unit: RenderUnit) -> str | None:
    params = unit.params_text.strip()
    prop_names = ["testID", "accessibilityLabel"] if len(unit.host_slots) == 1 else ["interactionLabels"]
    if unit.call_style == "jsx":
        if params.startswith("{") and params.endswith("}"):
            inner = params[1:-1].strip()
            existing = {
                item.split(":")[0].split("=")[0].strip()
                for item in inner.split(",")
                if item.strip()
            }
            missing = [prop_name for prop_name in prop_names if prop_name not in existing]
            if not missing:
                return None
            additions = ", ".join(missing)
            inner = f"{additions}, {inner}" if inner else additions
            return "{ " + inner + " }"
        if not params:
            return "{ " + ", ".join(prop_names) + " }"
        return None

    if all(re.search(rf"(?<![\w.]){re.escape(prop_name)}(?![\w.])", params) for prop_name in prop_names):
        return None

    additions = ", ".join(prop_names)
    new_params = f"{additions}, {params}" if params else additions
    if not unit.params_has_parens:
        return f"({new_params})"
    return new_params

def get_unit_label_source_exprs(unit: RenderUnit) -> tuple[str, str, str]:
    params = unit.params_text.strip()
    if (
        unit.call_style == "jsx"
        and params
        and not params.startswith("{")
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", params)
    ):
        return f"{params}.testID", f"{params}.accessibilityLabel", f"{params}.interactionLabels"
    return "testID", "accessibilityLabel", "interactionLabels"

def add_labels_to_body_targets(
    source: str,
    component_host_slots: dict[str, list[str]],
    function_host_slots: dict[str, list[str]],
    test_id_expr: str,
    accessibility_expr: str,
    labels_expr: str,
    clickable_host_alias_names: set[str] | None = None,
    press_required_component_names: set[str] | None = None,
) -> tuple[str, int]:
    targets = collect_body_targets(
        source,
        component_host_slots,
        function_host_slots,
        clickable_host_alias_names,
        press_required_component_names,
    )
    total_slots = sum(len(target.host_slots) for target in targets)
    if total_slots == 0:
        return source, 0

    replacements: list[tuple[int, int, str]] = []
    slot = 1
    for target in targets:
        key_expr = get_jsx_key_or_label_expr(target.text) or find_enclosing_map_key_expr(source, target.start)

        if target.kind == "jsx-host":
            if total_slots == 1:
                test_id_value = build_forwarded_label_attr_value(test_id_expr, key_expr)
                accessibility_value = build_forwarded_label_attr_value(accessibility_expr, key_expr)
            else:
                label_value = build_forwarded_label_attr_value(f"{labels_expr}?.[{slot}]", key_expr)
                test_id_value = label_value
                accessibility_value = label_value
            updated_tag = set_native_interaction_label_attributes(target.text, test_id_value, accessibility_value)
            replacements.append((target.start, target.end, updated_tag))
        elif target.kind == "jsx-component":
            if len(target.host_slots) == 1:
                if total_slots == 1:
                    test_id_value = build_forwarded_label_attr_value(test_id_expr, key_expr)
                    accessibility_value = build_forwarded_label_attr_value(accessibility_expr, key_expr)
                else:
                    label_value = build_forwarded_label_attr_value(f"{labels_expr}?.[{slot}]", key_expr)
                    test_id_value = label_value
                    accessibility_value = label_value
                updated_tag = remove_jsx_attribute(target.text, "interactionLabels")
                updated_tag = set_native_interaction_label_attributes(updated_tag, test_id_value, accessibility_value)
            else:
                label_bag = build_label_bag_slice_value(slot, target.host_slots, labels_expr, key_expr, jsx_attribute=True)
                updated_tag = set_generated_interaction_labels(target.text, label_bag)
            replacements.append((target.start, target.end, updated_tag))
        elif target.kind == "function-call" and target.args_start is not None and target.args_end is not None:
            args_text = source[target.args_start:target.args_end]
            if len(target.host_slots) == 1:
                if total_slots == 1:
                    additions = [test_id_expr, accessibility_expr]
                else:
                    label_expr = f"{labels_expr}?.[{slot}]"
                    additions = [label_expr, label_expr]
                replacement = replace_or_insert_leading_label_args(args_text, additions)
            else:
                label_bag = build_label_bag_slice_value(slot, target.host_slots, labels_expr, key_expr, jsx_attribute=False)
                replacement = replace_or_insert_leading_label_bag(args_text, label_bag)
            replacements.append((target.args_start, target.args_end, replacement))

        slot += len(target.host_slots)

    return apply_replacements(source, replacements, sort=True), total_slots

def wire_render_unit_definitions(
    code: str,
    units: list[RenderUnit],
    component_host_slots: dict[str, list[str]],
) -> tuple[str, int]:
    assign_render_unit_scope(units)
    slots_by_key = {unit.key: unit.host_slots for unit in units}
    replacements: list[tuple[int, int, str]] = []
    label_count = 0

    for unit in units:
        updated_params = add_interaction_labels_to_params(unit)
        if updated_params is not None and updated_params != unit.params_text:
            replacements.append((unit.params_start, unit.params_end, updated_params))

        body = code[unit.body_start:unit.body_end]
        unit_source = code[unit.definition_start:unit.definition_end]
        test_id_expr, accessibility_expr, labels_expr = get_unit_label_source_exprs(unit)
        visible_local_slots = build_visible_render_unit_slots(units, slots_by_key, unit)
        press_required_component_names = {
            visible_unit.name
            for visible_unit in units
            if visible_unit.requires_press_prop
        }
        updated_body, count = add_labels_to_body_targets(
            body,
            {**component_host_slots, **visible_local_slots},
            visible_local_slots,
            test_id_expr,
            accessibility_expr,
            labels_expr,
            extract_clickable_host_alias_names(unit_source),
            press_required_component_names,
        )
        label_count += count
        if updated_body != body:
            replacements.append((unit.body_start, unit.body_end, updated_body))

    if not replacements:
        return code, 0
    return apply_replacements(code, replacements, sort=True), label_count

def detect_unit_call_style(code: str, unit: RenderUnit, definition_ranges: list[tuple[int, int]]) -> str:
    for start, _, tag_text in iter_jsx_start_tags(code):
        if position_in_ranges(start, definition_ranges):
            continue
        if get_jsx_tag_name(tag_text) == unit.name:
            return "jsx"

    call_re = re.compile(rf"(?<![\w.]){re.escape(unit.name)}\s*\(")
    for match in call_re.finditer(code):
        if position_in_ranges(match.start(), definition_ranges):
            continue
        prev_char = code[match.start() - 1] if match.start() > 0 else ""
        if prev_char != "<":
            return "function"

    return "jsx" if unit.name[:1].isupper() else "function"

def inject_label_bags_into_render_unit_calls(
    code: str,
    units: list[RenderUnit],
    page_name: str,
) -> tuple[str, int]:
    if not units:
        return code, 0

    definition_ranges = [(unit.definition_start, unit.definition_end) for unit in units]
    units_by_name = {unit.name: unit for unit in units}
    counters: dict[str, int] = {}
    replacements: list[tuple[int, int, str]] = []

    for start, end, tag_text in iter_jsx_start_tags(code):
        if position_in_ranges(start, definition_ranges):
            continue
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or tag_name not in units_by_name:
            continue
        unit = units_by_name[tag_name]
        existing_index = get_existing_static_label_index(tag_text, page_name, tag_name)
        if existing_index is not None:
            counters[tag_name] = max(counters.get(tag_name, 0), existing_index)
            continue
        if unit.requires_press_prop and not has_meaningful_press_prop(tag_text):
            continue
        counters[tag_name] = counters.get(tag_name, 0) + 1
        key_expr = get_jsx_key_or_label_expr(tag_text) or find_enclosing_map_key_expr(code, start)
        if len(unit.host_slots) == 1:
            label_value = build_static_label(page_name, tag_name, counters[tag_name])
            if key_expr:
                label_value = build_static_index_key_label(page_name, tag_name, counters[tag_name], key_expr)
            updated_tag = remove_jsx_attribute(tag_text, "interactionLabels")
            updated_tag = set_native_interaction_label_attributes(updated_tag, label_value, label_value)
        else:
            label_bag = build_label_bag_value(page_name, tag_name, unit.host_slots, counters[tag_name], key_expr, jsx_attribute=True)
            updated_tag = set_generated_interaction_labels(tag_text, label_bag)
        replacements.append((start, end, updated_tag))

    for unit_name, unit in units_by_name.items():
        call_re = re.compile(rf"(?<![\w.]){re.escape(unit_name)}\s*\(")
        for match in call_re.finditer(code):
            if position_in_ranges(match.start(), definition_ranges):
                continue
            prev_char = code[match.start() - 1] if match.start() > 0 else ""
            if prev_char == "<":
                continue
            open_index = match.end() - 1
            close_index = find_matching_parenthesis(code, open_index)
            if close_index is None:
                continue
            args_text = code[open_index + 1:close_index]
            counters[unit_name] = counters.get(unit_name, 0) + 1
            if len(unit.host_slots) == 1:
                label_value = build_static_label(page_name, unit_name, counters[unit_name])
                replacement = replace_or_insert_leading_label_args(args_text, [label_value, label_value])
            else:
                label_bag = build_label_bag_value(page_name, unit_name, unit.host_slots, counters[unit_name], None, jsx_attribute=False)
                replacement = replace_or_insert_leading_label_bag(args_text, label_bag)
            replacements.append((open_index + 1, close_index, replacement))

    for start, end, tag_text in iter_jsx_start_tags(code):
        if position_in_ranges(start, definition_ranges):
            continue
        container_tag_name = get_jsx_tag_name(tag_text)
        tag_replacements: list[tuple[int, int, str]] = []
        for attr_match in re.finditer(r"(?<![\w.])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=", tag_text):
            attr_name = attr_match.group("name")
            if attr_name in {"testID", "accessibilityLabel", "interactionLabels"}:
                continue
            attr_value = parse_jsx_attribute_value(tag_text, attr_name)
            if not attr_value:
                continue
            value_text, value_start, value_end = attr_value
            if (
                container_tag_name in LIST_RENDER_ITEM_COMPONENT_NAMES
                and attr_name == "renderItem"
                and "<" in value_text
            ):
                updated_value, inline_count = inject_labels_into_inline_list_render_item(
                    value_text,
                    units_by_name,
                    page_name,
                    counters,
                )
                if inline_count:
                    tag_replacements.append((value_start, value_end, "{" + updated_value + "}"))
                continue

            referenced_name = value_text.strip()
            if referenced_name not in units_by_name:
                continue
            unit = units_by_name[referenced_name]
            counters[referenced_name] = counters.get(referenced_name, 0) + 1
            list_render_item_key_expr = (
                "args.item.key"
                if container_tag_name in LIST_RENDER_ITEM_COMPONENT_NAMES and attr_name == "renderItem"
                else None
            )
            if len(unit.host_slots) == 1:
                if list_render_item_key_expr:
                    label_value = build_static_key_label_expression(
                        page_name,
                        referenced_name,
                        counters[referenced_name],
                        list_render_item_key_expr,
                    )
                else:
                    label_value = build_static_label(page_name, referenced_name, counters[referenced_name])
                wrapper = "{(args) => " + referenced_name + "(" + label_value + ", " + label_value + ", args)}"
            else:
                label_bag = build_label_bag_value(
                    page_name,
                    referenced_name,
                    unit.host_slots,
                    counters[referenced_name],
                    list_render_item_key_expr,
                    jsx_attribute=False,
                )
                wrapper = "{(args) => " + referenced_name + "(" + label_bag + ", args)}"
            tag_replacements.append((value_start, value_end, wrapper))
        if tag_replacements:
            updated_tag = apply_replacements(tag_text, tag_replacements, sort=True)
            replacements.append((start, end, updated_tag))

    if not replacements:
        return code, 0
    return apply_replacements(code, replacements, sort=True), sum(counters.values())

def infer_list_render_item_key_expr(render_item_expr: str, tag_text: str) -> str | None:
    key_expr = get_jsx_key_or_label_expr(tag_text)
    if key_expr:
        return key_expr

    if re.search(r"\{\s*item\s*(?:,|\})", render_item_expr):
        return "item"

    match = re.search(r"(?P<arg>[A-Za-z_][A-Za-z0-9_]*)\s*=>", render_item_expr)
    if match:
        return f"{match.group('arg')}.item"

    match = re.search(r"\(\s*(?P<arg>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*=>", render_item_expr)
    if match:
        return f"{match.group('arg')}.item"

    return None

def build_static_index_key_label(page_name: str, component_type: str, index: int, key_expr: str) -> str:
    return "{" + build_static_key_label_expression(page_name, component_type, index, key_expr) + "}"

def inject_labels_into_inline_list_render_item(
    render_item_expr: str,
    units_by_name: dict[str, RenderUnit],
    page_name: str,
    counters: dict[str, int],
) -> tuple[str, int]:
    replacements: list[tuple[int, int, str]] = []
    label_count = 0

    for start, end, tag_text in iter_jsx_start_tags(render_item_expr):
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or tag_name not in units_by_name:
            continue

        unit = units_by_name[tag_name]
        existing_index = get_existing_static_label_index(tag_text, page_name, tag_name)
        if existing_index is not None:
            counters[tag_name] = max(counters.get(tag_name, 0), existing_index)
            continue

        counters[tag_name] = counters.get(tag_name, 0) + 1
        key_expr = infer_list_render_item_key_expr(render_item_expr, tag_text)
        if len(unit.host_slots) == 1:
            if key_expr:
                label_value = build_static_index_key_label(page_name, tag_name, counters[tag_name], key_expr)
            else:
                label_value = build_static_label(page_name, tag_name, counters[tag_name])
            updated_tag = remove_jsx_attribute(tag_text, "interactionLabels")
            updated_tag = set_native_interaction_label_attributes(updated_tag, label_value, label_value)
        else:
            label_bag = build_label_bag_value(
                page_name,
                tag_name,
                unit.host_slots,
                counters[tag_name],
                key_expr,
                jsx_attribute=True,
            )
            updated_tag = set_generated_interaction_labels(tag_text, label_bag)
        replacements.append((start, end, updated_tag))
        label_count += len(unit.host_slots)

    if not replacements:
        return render_item_expr, 0
    return apply_replacements(render_item_expr, replacements, sort=True), label_count

def inject_label_bags_into_imported_component_calls(
    code: str,
    component_host_slots: dict[str, list[str]],
    page_name: str,
    skipped_ranges: list[tuple[int, int]] | None = None,
) -> tuple[str, int]:
    if not component_host_slots:
        return code, 0
    skipped_ranges = skipped_ranges or []
    counters: dict[str, int] = {}
    replacements: list[tuple[int, int, str]] = []

    for start, end, tag_text in iter_jsx_start_tags(code):
        if position_in_ranges(start, skipped_ranges):
            continue
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or tag_name not in component_host_slots:
            continue
        host_slots = component_host_slots[tag_name]
        existing_index = get_existing_static_label_index(tag_text, page_name, tag_name)
        if existing_index is not None:
            counters[tag_name] = max(counters.get(tag_name, 0), existing_index)
            continue
        counters[tag_name] = counters.get(tag_name, 0) + 1
        key_expr = get_jsx_key_or_label_expr(tag_text) or find_enclosing_map_key_expr(code, start)
        if len(host_slots) == 1:
            label_value = build_static_label(page_name, tag_name, counters[tag_name])
            if key_expr:
                label_value = build_static_index_key_label(page_name, tag_name, counters[tag_name], key_expr)
            updated_tag = remove_jsx_attribute(tag_text, "interactionLabels")
            updated_tag = set_native_interaction_label_attributes(updated_tag, label_value, label_value)
        else:
            label_bag = build_label_bag_value(page_name, tag_name, host_slots, counters[tag_name], key_expr, jsx_attribute=True)
            updated_tag = set_generated_interaction_labels(tag_text, label_bag)
        replacements.append((start, end, updated_tag))

    if not replacements:
        return code, 0
    return apply_replacements(code, replacements, sort=True), sum(counters.values())

def inject_labels_into_clickable_custom_component_calls(
    code: str,
    page_name: str,
    skipped_ranges: list[tuple[int, int]] | None = None,
) -> tuple[str, int]:
    skipped_ranges = skipped_ranges or []
    counters: dict[str, int] = {}
    replacements: list[tuple[int, int, str]] = []

    for start, end, tag_text in iter_jsx_start_tags(code):
        if position_in_ranges(start, skipped_ranges):
            continue
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or not is_custom_component_tag_name(tag_name):
            continue
        if not has_meaningful_press_prop(tag_text):
            continue
        if parse_jsx_attribute_value(tag_text, "interactionLabels"):
            continue

        existing_index = get_existing_static_label_index(tag_text, page_name, tag_name)
        if existing_index is not None:
            counters[tag_name] = max(counters.get(tag_name, 0), existing_index)
            continue

        counters[tag_name] = counters.get(tag_name, 0) + 1
        key_expr = get_jsx_key_or_label_expr(tag_text) or find_enclosing_map_key_expr(code, start)
        label_value = (
            build_static_index_key_label(page_name, tag_name, counters[tag_name], key_expr)
            if key_expr
            else build_static_label(page_name, tag_name, counters[tag_name])
        )
        updated_tag = set_native_interaction_label_attributes(tag_text, label_value, label_value)
        replacements.append((start, end, updated_tag))

    if not replacements:
        return code, 0
    return apply_replacements(code, replacements, sort=True), len(replacements)

def build_imported_component_host_types(
    section_name: str,
    local_imports_by_section: dict[str, dict[str, str]],
    host_types_by_section: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    imported_component_host_types: dict[str, list[str]] = {}
    for component_name, import_path in local_imports_by_section.get(section_name, {}).items():
        imported_host_types = host_types_by_section.get(import_path, {})
        host_types = imported_host_types.get(component_name)
        if host_types is None:
            host_types = imported_host_types.get(Path(import_path).stem)
        if host_types is None and len(imported_host_types) == 1:
            host_types = next(iter(imported_host_types.values()))
        if host_types:
            imported_component_host_types[component_name] = host_types
    return imported_component_host_types

def update_host_types_by_section(
    sections: list[Section],
    local_imports_by_section: dict[str, dict[str, str]],
) -> dict[str, dict[str, list[str]]]:
    host_types_by_section = {
        section.name: extract_effective_component_interaction_host_types(section.code)
        for section in sections
    }

    changed = True
    iteration = 0
    max_iterations = max(4, len(sections) * 4)
    while changed:
        iteration += 1
        if iteration > max_iterations:
            print(
                "[label] warning: update_host_types_by_section did not converge; "
                f"using last approximation after {max_iterations} iterations"
            )
            break
        changed = False
        for section in sections:
            imported_component_host_types = build_imported_component_host_types(
                section.name,
                local_imports_by_section,
                host_types_by_section,
            )
            component_host_types = {
                **imported_component_host_types,
                **host_types_by_section.get(section.name, {}),
            }
            next_host_types = dict(host_types_by_section.get(section.name, {}))
            for unit in collect_render_units(section.code, component_host_types, exclude_names=set()):
                if (
                    unit.name[:1].isupper()
                    and unit.parent_key is None
                    and next_host_types.get(unit.name) != unit.host_slots
                ):
                    next_host_types[unit.name] = unit.host_slots
                    changed = True
            host_types_by_section[section.name] = next_host_types

    return host_types_by_section

def build_static_label(page_name: str, component_type: str, index: int) -> str:
    return f'"{page_name}.{component_type}.{index}"'

def is_key_index_expr(key_expr: str) -> bool:
    return bool(KEY_INDEX_EXPR_RE.fullmatch(key_expr))

def is_key_value_expr(key_expr: str) -> bool:
    return bool(KEY_VALUE_EXPR_RE.fullmatch(key_expr))

def build_key_suffix(key_expr: str) -> str:
    if is_key_index_expr(key_expr):
        return f"${{{key_expr}+100}}"
    if is_key_value_expr(key_expr):
        return f"${{{key_expr}}}"
    return f"${{{key_expr}}}"

def build_static_key_label(page_name: str, component_type: str, key_expr: str) -> str:
    return "{`" + f"{page_name}.{component_type}.{build_key_suffix(key_expr)}`" + "}"

def build_static_key_label_expression(page_name: str, component_type: str, index: int, key_expr: str) -> str:
    base = build_static_label_base(page_name, component_type, index)
    return "`" + f"{base}.${{{key_expr.strip()}}}`"

def build_tab_screen_label(owner_name: str, index: int) -> str:
    return f'"{owner_name}.Tab.Screen.{index}"'

def add_tab_screen_options_to_tag(tag_text: str, label_value: str) -> str:
    if "tabBarButtonTestID" in tag_text and "tabBarAccessibilityLabel" in tag_text:
        return tag_text

    option_props = (
        f"tabBarButtonTestID: {label_value}, "
        f"tabBarAccessibilityLabel: {label_value}"
    )
    options_attr = parse_jsx_attribute_value(tag_text, "options")
    if not options_attr:
        return set_or_replace_attribute(tag_text, "options", "{{ " + option_props + " }}")

    options_expr, value_start, value_end = options_attr
    stripped_options_expr = options_expr.strip()
    if stripped_options_expr.startswith("{") and stripped_options_expr.endswith("}"):
        inner = stripped_options_expr[1:-1].strip()
        updated_options_expr = "{ " + option_props + (", " + inner if inner else "") + " }"
    else:
        updated_options_expr = "{ ...(" + stripped_options_expr + "), " + option_props + " }"
    return tag_text[:value_start] + "{" + updated_options_expr + "}" + tag_text[value_end:]

def add_tab_screen_labels(code: str, fallback_component_name: str) -> str:
    component_ranges = extract_declared_component_ranges(code)
    replacements: list[tuple[int, int, str]] = []
    counters: dict[str, int] = {}

    for start, end, tag_text in iter_labelable_jsx_start_tags(code):
        tag_name = get_jsx_tag_name(tag_text)
        if not tag_name or not is_tab_screen_tag_name(tag_name):
            continue

        owner_name = find_enclosing_component_name(start, component_ranges, fallback_component_name)
        counters[owner_name] = counters.get(owner_name, 0) + 1
        label_value = build_tab_screen_label(owner_name, counters[owner_name])
        updated_tag = add_tab_screen_options_to_tag(tag_text, label_value)
        if updated_tag != tag_text:
            replacements.append((start, end, updated_tag))

    if not replacements:
        return code

    return apply_replacements(code, replacements)

def process_page_code(
    code: str,
    page_name: str,
    imported_component_host_types: dict[str, list[str]],
) -> str:
    updated = code
    local_units = collect_render_units(
        updated,
        imported_component_host_types,
        exclude_names={page_name},
    )
    definition_ranges = [(unit.definition_start, unit.definition_end) for unit in local_units]
    for unit in local_units:
        unit.call_style = detect_unit_call_style(updated, unit, definition_ranges)

    updated, _ = wire_render_unit_definitions(updated, local_units, imported_component_host_types)

    local_units = collect_render_units(
        updated,
        imported_component_host_types,
        exclude_names={page_name},
    )
    definition_ranges = [(unit.definition_start, unit.definition_end) for unit in local_units]
    updated, _ = inject_label_bags_into_render_unit_calls(updated, local_units, page_name)
    updated, _ = inject_label_bags_into_imported_component_calls(
        updated,
        imported_component_host_types,
        page_name,
        skipped_ranges=definition_ranges,
    )
    updated, _ = inject_labels_into_clickable_custom_component_calls(
        updated,
        page_name,
        skipped_ranges=definition_ranges,
    )
    updated, _ = add_labels_to_clickable_components(
        updated,
        lambda component_type, index: build_static_label(page_name, component_type, index),
        key_label_builder=lambda component_type, key_expr: build_static_key_label(page_name, component_type, key_expr),
        skipped_ranges=definition_ranges,
    )
    updated = add_root_container_label_if_needed(updated, page_name)
    updated, _ = add_labels_to_embedded_jsx_attribute_components(
        updated,
        page_name,
        find_page_main_jsx_range(updated, page_name),
    )
    return updated

def process_component_code(
    code: str,
    component_name: str,
    imported_component_host_types: dict[str, list[str]],
) -> str:
    updated = code
    local_units = collect_render_units(
        updated,
        imported_component_host_types,
        exclude_names=set(),
    )
    definition_ranges = [(unit.definition_start, unit.definition_end) for unit in local_units]
    for unit in local_units:
        unit.call_style = detect_unit_call_style(updated, unit, definition_ranges)
    updated, _ = wire_render_unit_definitions(updated, local_units, imported_component_host_types)
    return updated

def process_app_code(
    code: str,
    imported_components: set[str],
    imported_component_host_types: dict[str, list[str]],
) -> str:
    app_page_name = "App"
    updated = code
    call_component_host_types = {
        component_name: host_types
        for component_name, host_types in imported_component_host_types.items()
        if component_name in imported_components
    }
    if call_component_host_types:
        updated, _ = inject_label_bags_into_imported_component_calls(
            updated,
            call_component_host_types,
            app_page_name,
        )
        updated, _ = inject_label_props_into_component_references(
            updated,
            call_component_host_types,
            lambda child_component_name, host_types, index: (
                build_static_label(app_page_name, child_component_name, index)
                if len(host_types) == 1
                else build_label_bag_value(
                    app_page_name,
                    child_component_name,
                    host_types,
                    index,
                    None,
                    jsx_attribute=True,
                )
            ),
        )
    updated, _ = inject_labels_into_clickable_custom_component_calls(
        updated,
        app_page_name,
    )
    return updated

def process_res_sections(sections: list[Section]) -> list[Section]:
    sections = deduplicate_sections(sections)
    section_by_name = {section.name: section for section in sections}
    app_section = section_by_name.get("App.js")

    local_imports_by_section = {
        section.name: extract_local_component_imports(section.code, section.name)
        for section in sections
    }
    local_imports_by_section = resolve_import_paths_to_sections(
        local_imports_by_section,
        section_by_name,
    )
    initial_page_section_names = (
        resolve_section_name_set(extract_page_section_names(app_section.code), section_by_name)
        if app_section
        else set()
    )
    page_section_names = expand_page_section_names(
        initial_page_section_names,
        local_imports_by_section,
        section_by_name,
    )
    used_component_names_by_section = {
        section.name: extract_used_component_names(section.code)
        for section in sections
    }
    host_types_by_section = update_host_types_by_section(sections, local_imports_by_section)
    common_component_sections: set[str] = {
        import_path
        for page_section_name in page_section_names
        for import_path in local_imports_by_section.get(page_section_name, {}).values()
        if import_path not in page_section_names and import_path != "App.js"
    }
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
        imported_component_host_types = build_imported_component_host_types(
            section.name,
            local_imports_by_section,
            host_types_by_section,
        )

        if section.name == "App.js":
            updated_code = process_app_code(section.code, imported_components, imported_component_host_types)
        elif section.name in page_section_names:
            updated_code = process_page_code(
                section.code,
                section_stem,
                imported_component_host_types,
            )
        else:
            updated_code = process_component_code(
                section.code,
                section_stem,
                imported_component_host_types,
            )
        updated_code = add_tab_screen_labels(updated_code, section_stem)
        updated_sections.append(Section(name=section.name, code=updated_code))
    return updated_sections

def get_output_res_path(res_path: Path) -> Path:
    return res_path.with_name(OUTPUT_FILENAME)

def process_res_file(res_path: Path, write: bool) -> tuple[bool, Path]:
    original_text = res_path.read_text(encoding="utf-8")
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
    if LIMIT_FILES is not None and LIMIT_FILES > 0:
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
