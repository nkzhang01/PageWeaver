/**
 * shoot2.mjs —— baseline_UTG+Component+Mockups 专用（带"屏幕内多状态聚类"+截图↔js映射）。
 * 规则: 同一屏幕(代码文件)内点击触发的多个状态 = 同一 page_id 的多个 state; 跳转到另一屏幕 = 新 page_id。
 * 做法: 解析每屏幕的内部状态(useState 布尔弹窗 + 字符串tab) → 注入初始状态逐个独立渲染截图 → 按屏幕聚类。
 *       page_cluster.json 里每个 page_id 带 source_js(主屏幕文件 + 递归 import 的本地子组件)。
 * 复用 shoot.mjs 机制: am start -S --ei pageIndex i 重启独立渲染 + 离线 bundle 推送(push模式)。
 *
 * 用法: CATEGORY=Work APP=WeCom node tools/shoot2.mjs        (单个)
 *       TARGET="Work/WeCom,Life/ChinaTelecom,..." node tools/shoot2.mjs  (指定多个)
 *       CATEGORY=Work APP=WeCom BUNDLE_ONLY=1 node tools/shoot2.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';

const md5 = (b) => crypto.createHash('md5').update(b).digest('hex');

// ===== 配置 =====
const HOST_PROJECT = 'D:\\ui2code\\testAndroid\\AwesomeProject';
const BASE2_ROOT = 'D:\\ui2code\\InteractionSequence\\results\\code_results\\baseline\\baseline_UTG+Component';
const OUTPUT_ROOT = 'D:\\ui2code\\InteractionSequence\\results\\code_screenshots_state\\UTG+Com';
const RETRY_APPS_JSON = 'D:\\ui2code\\InteractionSequence\\results\\code_screenshots_state\\UTG+Com\\blank_and_failed_apps.json';
const PACKAGE = 'com.awesomeproject';
const ACTIVITY = 'com.awesomeproject/.MainActivity';
const DEVICE_BUNDLE_DIR = '/sdcard/Android/data/com.awesomeproject/files';
const DEVICE_BUNDLE_PATH = `${DEVICE_BUNDLE_DIR}/awesome.bundle`;

const FILTER_CATEGORY = process.env.CATEGORY || null;
const FILTER_APP = process.env.APP || null;
const TARGET = (process.env.TARGET || '').split(',').map((s) => s.trim()).filter(Boolean);
const BUNDLE_ONLY = process.env.BUNDLE_ONLY === '1';
const DEVICE = process.env.DEVICE || null;
const COLD_START_MS = Number(process.env.COLD_START_MS || 2800);
const MIN_RENDER_MS = Number(process.env.MIN_RENDER_MS || 3500);
const WAIT_MS = Number(process.env.WAIT_MS || 6000);
const CAP_TAB_VALUES = Number(process.env.CAP_TAB_VALUES || 6);

const APP_JS = path.join(HOST_PROJECT, 'App.js');
const COMPONENTS_DIR = path.join(HOST_PROJECT, 'components');
const SHOOT2_TOOLS_DIR = path.join(HOST_PROJECT, 'tools');
const SHOOT2_METRO_CONFIG = path.join(SHOOT2_TOOLS_DIR, '_shoot2_metro.config.js');
const SHOOT2_METRO_TMP = path.join(SHOOT2_TOOLS_DIR, '_shoot2_metro_tmp');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sleepSync = (ms) => { try { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); } catch {} };
const adbRaw = (args, opts = {}) =>
  execFileSync('adb', DEVICE ? ['-s', DEVICE, ...args] : args, { maxBuffer: 1 << 30, timeout: 12000, ...opts });
function adb(args, opts = {}) {
  let last;
  for (let i = 0; i < 4; i++) {
    try { return adbRaw(args, opts); }
    catch (e) { last = e; sleepSync(1500); try { execFileSync('adb', ['start-server'], { stdio: 'ignore' }); } catch {} try { execFileSync('adb', ['wait-for-device'], { timeout: 15000, stdio: 'ignore' }); } catch {} }
  }
  throw last;
}
function ensureAdbServer() {
  try { execFileSync('adb', ['start-server'], { stdio: 'ignore' }); } catch {}
  try { execFileSync('adb', ['wait-for-device'], { timeout: 20000, stdio: 'ignore' }); } catch {}
}

const stem = (f) => f.replace(/\.(jsx?)$/i, '');

function loadRetryData() {
  if (!fs.existsSync(RETRY_APPS_JSON)) {
    throw new Error(`retry apps json not found: ${RETRY_APPS_JSON}`);
  }
  const raw = fs.readFileSync(RETRY_APPS_JSON, 'utf8');
  return JSON.parse(raw);
}

function recomputeRetryCounts(data) {
  if (!data.counts) data.counts = {};
  const blankApps = data.apps_with_blank_or_near_blank_pages || [];
  const errorApps = data.apps_with_error_screens || [];
  const failedApps = data.apps_without_successful_output || [];
  const blankPages = blankApps.flatMap((item) => item.pages || []);
  const errorPages = errorApps.flatMap((item) => item.pages || []);
  data.counts.apps_with_blank_or_near_blank_pages = blankApps.length;
  data.counts.blank_or_near_blank_pages = blankPages.length;
  data.counts.strict_blank_pages = blankPages.filter((page) => page.blank_type === 'strict_blank').length;
  data.counts.near_blank_pages = blankPages.filter((page) => page.blank_type === 'near_blank').length;
  data.counts.apps_with_error_screens = errorApps.length;
  data.counts.error_screen_pages = errorPages.length;
  data.counts.apps_without_successful_output = failedApps.length;
  data.counts.compile_or_bundle_failed_apps = failedApps.filter((item) => item.compile_or_bundle_failed).length;
}

function saveRetryData(data) {
  recomputeRetryCounts(data);
  fs.writeFileSync(RETRY_APPS_JSON, JSON.stringify(data, null, 2), 'utf8');
}

function loadRetryAppSet() {
  const data = loadRetryData();
  const apps = new Set();
  for (const item of data.apps_with_blank_or_near_blank_pages || []) {
    if (item.category && item.app) apps.add(`${item.category}/${item.app}`);
  }
  for (const item of data.apps_with_error_screens || []) {
    if (item.category && item.app) apps.add(`${item.category}/${item.app}`);
  }
  for (const item of data.apps_without_successful_output || []) {
    if (item.category && item.app) apps.add(`${item.category}/${item.app}`);
  }
  return apps;
}

function removeRetryApp(category, app) {
  const data = loadRetryData();
  data.apps_with_blank_or_near_blank_pages = (data.apps_with_blank_or_near_blank_pages || [])
    .filter((item) => item.category !== category || item.app !== app);
  data.apps_with_error_screens = (data.apps_with_error_screens || [])
    .filter((item) => item.category !== category || item.app !== app);
  data.apps_without_successful_output = (data.apps_without_successful_output || [])
    .filter((item) => item.category !== category || item.app !== app);
  saveRetryData(data);
}

function classifyFailure(phase, errorText) {
  if (phase === 'bundle' || /bundle|BUNDLE|Unable to resolve module|TransformError|SyntaxError/i.test(errorText)) {
    return { failure_type: 'bundle_failed', compile_or_bundle_failed: true };
  }
  if (/adb|device|emulator|activity|screencap|am start|push/i.test(errorText)) {
    return { failure_type: 'runtime_or_adb_failed', compile_or_bundle_failed: false };
  }
  return { failure_type: 'unknown_failed', compile_or_bundle_failed: false };
}

function recordRetryFailure(category, app, phase, error) {
  const errorText = String(error && error.message ? error.message : error);
  const data = loadRetryData();
  const failedApps = data.apps_without_successful_output || [];
  const existing = failedApps.find((item) => item.category === category && item.app === app);
  const outDir = path.join(OUTPUT_ROOT, category, app);
  const classified = classifyFailure(phase, errorText);
  const entry = {
    category,
    app,
    page_cluster_exists: fs.existsSync(path.join(outDir, 'page_cluster.json')),
    png_count: fs.existsSync(outDir) ? fs.readdirSync(outDir).filter((f) => /\.png$/i.test(f)).length : 0,
    ...classified,
    error_summary: errorText.split('\n').find((line) => line.trim()) || errorText.slice(0, 300),
    last_error: errorText.slice(0, 4000),
    last_failed_at: new Date().toISOString(),
  };
  if (existing) Object.assign(existing, entry);
  else failedApps.push(entry);
  data.apps_without_successful_output = failedApps;
  saveRetryData(data);
}

function listApps() {
  const targetSet = TARGET.length ? new Set(TARGET) : null;
  const retryAppSet = loadRetryAppSet();
  const apps = [];
  for (const cat of fs.readdirSync(BASE2_ROOT, { withFileTypes: true }).filter((d) => d.isDirectory())) {
    if (FILTER_CATEGORY && cat.name !== FILTER_CATEGORY) continue;
    for (const app of fs.readdirSync(path.join(BASE2_ROOT, cat.name), { withFileTypes: true }).filter((d) => d.isDirectory())) {
      if (FILTER_APP && app.name !== FILTER_APP) continue;
      if (!retryAppSet.has(`${cat.name}/${app.name}`)) continue;
      if (targetSet && !targetSet.has(`${cat.name}/${app.name}`)) continue;
      apps.push({ category: cat.name, app: app.name, dir: path.join(BASE2_ROOT, cat.name, app.name) });
    }
  }
  return apps;
}

// 解析 App.js 本地默认 import → 顶层屏幕组件(各 page_id), 按出现顺序
function resolveScreens(appDir, allStems) {
  const appJs = path.join(appDir, 'App.js');
  if (!fs.existsSync(appJs)) return allStems;
  const code = fs.readFileSync(appJs, 'utf8');
  const known = new Set(allStems);
  const NON = new Set(['styles', 'theme', 'colors', 'constants', 'utils', 'helpers', 'data', 'config']);
  const re = /import\s+([A-Za-z_$][\w$]*)\s*(?:,\s*\{[^}]*\})?\s+from\s+['"](\.[^'"]+)['"]/g;
  const componentToStem = new Map();
  const imported = []; const importedSeen = new Set(); let m;
  while ((m = re.exec(code)) !== null) {
    const s = m[2].split('/').pop().replace(/\.(jsx?)$/i, '');
    if (NON.has(s.toLowerCase()) || !known.has(s) || importedSeen.has(s)) continue;
    if (!importedSeen.has(s)) { importedSeen.add(s); imported.push(s); }
    componentToStem.set(m[1], s);
  }

  const rendered = []; const renderedSeen = new Set();
  const addRendered = (componentName) => {
    const s = componentToStem.get(componentName);
    if (!s || renderedSeen.has(s)) return;
    renderedSeen.add(s); rendered.push(s);
  };

  const renderPatterns = [
    /case\s+['"][^'"]+['"]\s*:\s*return\s*<([A-Za-z_$][\w$]*)\b/g,
    /\{\s*[A-Za-z_$][\w$.]*\s*={2,3}\s*['"][^'"]+['"]\s*&&\s*<([A-Za-z_$][\w$]*)\b/g,
    /default\s*:\s*return\s*<([A-Za-z_$][\w$]*)\b/g,
  ];
  for (const pattern of renderPatterns) {
    let mm;
    while ((mm = pattern.exec(code)) !== null) addRendered(mm[1]);
  }

  return rendered.length ? rendered : (imported.length ? imported : allStems);
}

function inferScreenRoutes(appDir, screens) {
  const routes = Object.fromEntries(screens.map((s) => [s, s.replace(/Screen$/, '')]));
  const appJs = path.join(appDir, 'App.js');
  if (!fs.existsSync(appJs)) return routes;

  const code = fs.readFileSync(appJs, 'utf8');
  const componentToStem = new Map();
  const importRe = /import\s+([A-Za-z_$][\w$]*)\s*(?:,\s*\{[^}]*\})?\s+from\s+['"](\.[^'"]+)['"]/g;
  let m;
  while ((m = importRe.exec(code)) !== null) {
    const s = m[2].split('/').pop().replace(/\.(jsx?)$/i, '');
    if (Object.prototype.hasOwnProperty.call(routes, s)) componentToStem.set(m[1], s);
  }

  const patterns = [
    /case\s+['"]([^'"]+)['"]\s*:\s*return\s*<([A-Za-z_$][\w$]*)\b/g,
    /\{\s*[A-Za-z_$][\w$.]*\s*={2,3}\s*['"]([^'"]+)['"]\s*&&\s*<([A-Za-z_$][\w$]*)\b/g,
  ];
  for (const pattern of patterns) {
    let mm;
    while ((mm = pattern.exec(code)) !== null) {
      const screen = componentToStem.get(mm[2]);
      if (screen) routes[screen] = mm[1];
    }
  }

  return routes;
}

// 该屏幕的源码映射: 主文件 + 递归 import 的本地子组件 (相对路径解析, 排除三方库)
function resolveSourceJs(appDir, screenStem) {
  const result = [];
  const seen = new Set();
  const visit = (rel) => {
    let file = null;
    for (const ext of ['.js', '.jsx']) { if (fs.existsSync(path.join(appDir, rel + ext))) { file = rel + ext; break; } }
    if (!file || seen.has(file)) return;
    seen.add(file); result.push(file.split(path.sep).join('/'));
    const code = fs.readFileSync(path.join(appDir, file), 'utf8');
    const dir = path.posix.dirname(rel.split(path.sep).join('/'));
    const re = /from\s+['"](\.[^'"]+)['"]/g; let m;
    while ((m = re.exec(code)) !== null) {
      const resolved = path.posix.normalize(path.posix.join(dir, m[1])).replace(/^\.\//, '');
      visit(resolved.split('/').join(path.sep));
    }
  };
  visit(screenStem);
  return result;
}

// 解析一个屏幕组件的内部状态 -> [{label, set, props}], 第一个总是 default
// set: 注入到 globalThis.__HS (改 useState 初值); props: 注入为组件 prop (应对 useState(propName) + useEffect 重置 的 tab)
function detectStates(code) {
  const states = [{ label: 'default', set: {}, props: {} }];
  const declRe = /const\s*\[\s*([A-Za-z_$][\w$]*)\s*,\s*set[A-Za-z0-9_$]*\s*\]\s*=\s*useState\(([^()]*)\)/g;
  const vars = []; let m;
  while ((m = declRe.exec(code)) !== null) vars.push({ name: m[1], init: (m[2] || '').trim() });
  for (const v of vars) {
    const setName = 'set' + v.name.charAt(0).toUpperCase() + v.name.slice(1);
    const boolRe = new RegExp('\\{\\s*' + v.name + '\\s*(?:&&|\\?|\\})');
    const setTrueRe = new RegExp(setName + '\\(\\s*true\\s*\\)');
    if (/^(false|true)$/.test(v.init) && (boolRe.test(code) || setTrueRe.test(code))) {
      states.push({ label: v.name, set: { [v.name]: true }, props: {} });
      continue;
    }
    // 枚举该状态在代码里出现的字符串字面值(=== 'x' 或 setX('y'))
    const vals = new Set();
    let mm;
    const eqRe = new RegExp(v.name + "\\s*===?\\s*['\"]([^'\"]+)['\"]", 'g');
    while ((mm = eqRe.exec(code)) !== null) vals.add(mm[1]);
    const setRe = new RegExp(setName + "\\(\\s*['\"]([^'\"]+)['\"]\\s*\\)", 'g');
    while ((mm = setRe.exec(code)) !== null) vals.add(mm[1]);
    if (!vals.size) continue;
    // init 是否 prop 驱动: 取 init 开头的标识符(如 useState(initialTab) / useState(initialTab || 'all'))
    const idMatch = v.init.match(/^([A-Za-z_$][\w$]*)/);
    const propName = idMatch && !/^(true|false|null|undefined|NaN)$/.test(idMatch[1]) ? idMatch[1] : null;
    // 跳过默认值(避免和 default 截重复): init 里的字面量, 或 prop 的解构默认值(propName = 'x')
    let skipVal = (v.init.match(/['"]([^'"]+)['"]/) || [])[1];
    if (!skipVal && propName) skipVal = (code.match(new RegExp(propName + "\\s*=\\s*['\"]([^'\"]+)['\"]")) || [])[1];
    for (const val of [...vals].filter((x) => x !== skipVal).slice(0, CAP_TAB_VALUES)) {
      states.push({ label: `${v.name}=${val}`, set: { [v.name]: val }, props: propName ? { [propName]: val } : {} });
    }
  }
  return states;
}

// 把 useState(INIT) 改写为可被全局 __HS 注入初值
function transformComponent(code) {
  return code.replace(
    /const(\s*\[\s*)([A-Za-z_$][\w$]*)(\s*,\s*set[A-Za-z0-9_$]*\s*\]\s*=\s*)useState\(([^()]*)\)/g,
    (full, a, name, b, init) => `const${a}${name}${b}useState((globalThis.__HS && '${name}' in globalThis.__HS) ? globalThis.__HS['${name}'] : (${init}))`,
  );
}

function safeSegment(value) {
  return String(value).replace(/[^A-Za-z0-9_$-]/g, '_');
}
function resetComponents() {
  fs.mkdirSync(COMPONENTS_DIR, { recursive: true });
  for (const entry of fs.readdirSync(COMPONENTS_DIR, { withFileTypes: true })) {
    fs.rmSync(path.join(COMPONENTS_DIR, entry.name), { recursive: true, force: true });
  }
}
function copyTreeTransformed(src, dest, top = true) {
  fs.mkdirSync(dest, { recursive: true });
  for (const d of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, d.name);
    if (d.isDirectory()) copyTreeTransformed(s, path.join(dest, d.name), false);
    else if (/\.(jsx?)$/i.test(d.name) && !(top && d.name === 'App.js')) {
      fs.writeFileSync(path.join(dest, d.name), transformComponent(fs.readFileSync(s, 'utf8')), 'utf8');
    }
  }
}

function rewriteLocalImportsForPrefixedComponents(code, prefix, knownStems) {
  return code.replace(/(from\s+['"])(\.[^'"]+)(['"])/g, (full, left, spec, right) => {
    if (spec === '../App') return full;
    const normalized = spec.replace(/\\/g, '/');
    const parts = normalized.split('/');
    const last = parts[parts.length - 1].replace(/\.(jsx?)$/i, '');
    if (!knownStems.has(last)) return full;
    parts[parts.length - 1] = `${prefix}${last}`;
    return `${left}${parts.join('/')}${right}`;
  });
}

function copyTreeTransformedWithPrefix(src, dest, prefix) {
  fs.mkdirSync(dest, { recursive: true });
  const sourceFiles = [];
  const collect = (folder, relDir = '') => {
    for (const d of fs.readdirSync(folder, { withFileTypes: true })) {
      const abs = path.join(folder, d.name);
      const rel = path.join(relDir, d.name);
      if (d.isDirectory()) collect(abs, rel);
      else if (/\.(jsx?)$/i.test(d.name) && d.name !== 'App.js') sourceFiles.push({ abs, rel });
    }
  };
  collect(src);

  const knownStems = new Set(sourceFiles.map((item) => stem(path.basename(item.rel))));
  for (const item of sourceFiles) {
    const relDir = path.dirname(item.rel);
    const fileStem = stem(path.basename(item.rel));
    const ext = path.extname(item.rel);
    const targetDir = relDir === '.' ? dest : path.join(dest, relDir);
    fs.mkdirSync(targetDir, { recursive: true });
    let code = transformComponent(fs.readFileSync(item.abs, 'utf8'));
    code = rewriteLocalImportsForPrefixedComponents(code, prefix, knownStems);
    fs.writeFileSync(path.join(targetDir, `${prefix}${fileStem}${ext}`), code, 'utf8');
  }
}

function assertCaptureImportsExist(captures, componentFilePrefix) {
  const missing = [];
  for (const screen of [...new Set(captures.map((c) => c.screen))]) {
    const base = path.join(COMPONENTS_DIR, `${componentFilePrefix}${screen}`);
    if (!fs.existsSync(`${base}.js`) && !fs.existsSync(`${base}.jsx`)) {
      missing.push(`${componentFilePrefix}${screen}`);
    }
  }
  if (missing.length) {
    throw new Error(`generated component files missing before bundle: ${missing.join(', ')}`);
  }
}

// function generateAppJs(captures) {
//   const screens = [...new Set(captures.map((c) => c.screen))];
//   const imports = screens.map((s, i) => `import S${i} from './components/${s}';`).join('\n');
//   const compMap = screens.map((s, i) => `  ${JSON.stringify(s)}: S${i},`).join('\n');
//   const caps = captures.map((c) => `  { screen: ${JSON.stringify(c.screen)}, set: ${JSON.stringify(c.set)}, props: ${JSON.stringify(c.props || {})} },`).join('\n');
//   return `/* AUTO-GENERATED harness (baseline2: 多状态/聚类) */
// import React from 'react';
// import { View } from 'react-native';
// import { NavigationContainer } from '@react-navigation/native';
// import { createNativeStackNavigator } from '@react-navigation/native-stack';
// import { SafeAreaProvider } from 'react-native-safe-area-context';
//
// ${imports}
//
// const COMP = {
// ${compMap}
// };
// const CAPTURES = [
// ${caps}
// ];
// const Stack = createNativeStackNavigator();
// const noop = () => {};
// const stub = { navigate: noop, navigatePush: noop, navigateBack: noop, goBack: noop, replaceRoot: noop,
//   navigation: { navigate: noop, push: noop, goBack: noop, replace: noop, setOptions: noop, addListener: () => (() => {}) },
//   route: { params: {} } };
// const Blank = () => null;
//
// class Boundary extends React.Component {
//   state = { err: false };
//   static getDerivedStateFromError() { return { err: true }; }
//   componentDidCatch(e) { try { console.error('PAGERR|' + this.props.name + '|' + (e && e.message ? e.message : String(e))); } catch (x) {} }
//   render() { return this.state.err ? null : this.props.children; }
// }
//
// export default function App(props) {
//   const i = props && typeof props.pageIndex === 'number' ? props.pageIndex : -1;
//   const cap = i >= 0 && i < CAPTURES.length ? CAPTURES[i] : null;
//   globalThis.__HS = cap ? cap.set : {};
//   const capProps = cap && cap.props ? cap.props : {};
//   const Comp = cap ? (COMP[cap.screen] || Blank) : Blank;
//   const name = cap ? cap.screen : '__blank__';
//   return (
//     <SafeAreaProvider>
//       <NavigationContainer>
//         <Stack.Navigator screenOptions={{ headerShown: false, animation: 'none' }}>
//           <Stack.Screen name={name}>
//             {(p) => (<Boundary name={name}><Comp {...stub} {...p} {...capProps} /></Boundary>)}
//           </Stack.Screen>
//         </Stack.Navigator>
//       </NavigationContainer>
//     </SafeAreaProvider>
//   );
// }
// `;
// }
function generateAppJs(captures) {
  const screens = [...new Set(captures.map((c) => c.screen))];
  const imports = screens.map((s, i) => `import S${i} from './components/${s}';`).join('\n');
  const compMap = screens.map((s, i) => `  ${JSON.stringify(s)}: S${i},`).join('\n');
  const caps = captures.map((c) => `  { screen: ${JSON.stringify(c.screen)}, set: ${JSON.stringify(c.set)}, props: ${JSON.stringify(c.props || {})} },`).join('\n');

  return `/* AUTO-GENERATED harness (baseline2: 多状态/聚类) */
import React, { createContext, useMemo } from 'react';
import { SafeAreaView, View, StatusBar, Text } from 'react-native';

${imports}

export const NavigationContext = createContext({
  navigate: (_name, _params) => {},
  replace: (_name, _params) => {},
  goBack: () => {},
  current: 'Main',
  params: {},
});

const COMP = {
${compMap}
};

const CAPTURES = [
${caps}
];

const noop = () => {};
const Blank = () => null;

function normalizeRouteName(screen) {
  if (!screen) return '__blank__';
  return screen.replace(/Screen$/, '');
}

class Boundary extends React.Component {
  state = { err: false, msg: '' };

  static getDerivedStateFromError(e) {
    return { err: true, msg: e && e.message ? e.message : String(e) };
  }

  componentDidCatch(e) {
    try {
      console.log('PAGERR|' + this.props.name + '|' + (e && e.message ? e.message : String(e)));
    } catch (x) {}
  }

  render() {
    if (this.state.err) {
      return (
        <View style={{ flex: 1, padding: 24, justifyContent: 'center', backgroundColor: '#FFFFFF' }}>
          <Text style={{ fontSize: 16, color: '#D00', fontWeight: '700' }}>{this.props.name}</Text>
          <Text style={{ marginTop: 12, color: '#333' }}>{this.state.msg}</Text>
        </View>
      );
    }
    return this.props.children;
  }
}

export default function App(props) {
  const i = props && typeof props.pageIndex === 'number' ? props.pageIndex : -1;
  const cap = i >= 0 && i < CAPTURES.length ? CAPTURES[i] : null;

  globalThis.__HS = cap ? cap.set : {};

  const capProps = cap && cap.props ? cap.props : {};
  const Comp = cap ? (COMP[cap.screen] || Blank) : Blank;
  const screenName = cap ? cap.screen : '__blank__';
  const routeName = normalizeRouteName(screenName);

  const navValue = useMemo(
    () => ({
      navigate: noop,
      replace: noop,
      goBack: noop,
      current: routeName,
      params: {},
    }),
    [routeName]
  );

  const navigationProp = {
    navigate: noop,
    push: noop,
    goBack: noop,
    replace: noop,
    setOptions: noop,
    addListener: () => (() => {}),
  };

  const routeProp = {
    name: routeName,
    params: {},
  };

  return (
    <NavigationContext.Provider value={navValue}>
      <SafeAreaView style={{ flex: 1, backgroundColor: '#FFFFFF' }}>
        <StatusBar barStyle="dark-content" />
        <View style={{ flex: 1 }}>
          <Boundary name={screenName}>
            <Comp
              {...capProps}
              navigation={navigationProp}
              route={routeProp}
            />
          </Boundary>
        </View>
      </SafeAreaView>
    </NavigationContext.Provider>
  );
}
`;
}

function generateAdaptedAppJs(captures, componentImportRoot = './components', componentFilePrefix = '') {
  const screens = [...new Set(captures.map((c) => c.screen))];
  const imports = screens.map((s, i) => `import S${i} from '${componentImportRoot}/${componentFilePrefix}${s}';`).join('\n');
  const compMap = screens.map((s, i) => `  ${JSON.stringify(s)}: S${i},`).join('\n');
  const caps = captures.map((c) => `  { screen: ${JSON.stringify(c.screen)}, route: ${JSON.stringify(c.route || c.screen.replace(/Screen$/, ''))}, set: ${JSON.stringify(c.set)}, props: ${JSON.stringify(c.props || {})} },`).join('\n');
  const routeNames = [...new Set(captures.map((c) => c.route || c.screen.replace(/Screen$/, '')))];
  const routeHelpers = routeNames
    .filter((route) => /^[A-Za-z_$][\w$]*$/.test(route))
    .map((route) => `    goTo${route}: noop,`)
    .join('\n');

  return `/* AUTO-GENERATED harness for results/code_results/baseline/baseline_UTG+Component */
import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { Linking, SafeAreaView, View, StatusBar, Text } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

${imports}

export const NavigationContext = createContext({
  navigate: (_name, _params) => {},
  push: (_name, _params) => {},
  replace: (_name, _params) => {},
  resetTo: (_name, _params) => {},
  goBack: () => {},
  setOptions: () => {},
  addListener: () => (() => {}),
  current: '',
  params: {},
  route: { name: '', params: {} },
});

export const useNav = () => useContext(NavigationContext);

export const AppContext = createContext({
  selfCheckFrequency: 'Every day',
  setSelfCheckFrequency: () => {},
  autoStorageCleanup: 'Every day',
  setAutoStorageCleanup: () => {},
  autoVirusUpdate: 'Using Wi-Fi or mobile data',
  setAutoVirusUpdate: () => {},
  autoBlockingListUpdate: 'Using Wi-Fi or mobile data',
  setAutoBlockingListUpdate: () => {},
  showRecommendedTools: true,
  setShowRecommendedTools: () => {},
  showToolsSection: true,
  setShowToolsSection: () => {},
});

export const ThemeContext = createContext({
  isDark: false,
  toggleTheme: () => {},
  colors: {
    bg: '#f6f6f6',
    card: '#ffffff',
    text: '#111111',
    mutedText: '#666666',
    primary: '#3f51b5',
    accent: '#ff4081',
    border: '#e0e0e0',
    overlay: 'rgba(0,0,0,0.4)',
  },
});

const COMP = {
${compMap}
};

const CAPTURES = [
${caps}
];

const noop = () => {};
const Blank = () => null;
const Stack = createNativeStackNavigator();
const defaultThemeColors = {
  bg: '#f6f6f6',
  card: '#ffffff',
  text: '#111111',
  mutedText: '#666666',
  primary: '#3f51b5',
  accent: '#ff4081',
  border: '#e0e0e0',
  overlay: 'rgba(0,0,0,0.4)',
};

function indexFromUrl(url) {
  if (!url) return null;
  const match = String(url).match(/capture\\/(\\-?\\d+)/);
  return match ? Number(match[1]) : null;
}

class Boundary extends React.Component {
  state = { err: false, msg: '' };

  static getDerivedStateFromError(e) {
    return { err: true, msg: e && e.message ? e.message : String(e) };
  }

  componentDidCatch(e) {
    try {
      console.log('PAGERR|' + this.props.name + '|' + (e && e.message ? e.message : String(e)));
    } catch (x) {}
  }

  render() {
    if (this.state.err) {
      return (
        <View style={{ flex: 1, padding: 24, justifyContent: 'center', backgroundColor: '#FFFFFF' }}>
          <Text style={{ fontSize: 16, color: '#D00', fontWeight: '700' }}>{this.props.name}</Text>
          <Text style={{ marginTop: 12, color: '#333' }}>{this.state.msg}</Text>
        </View>
      );
    }
    return this.props.children;
  }
}

export default function App(props) {
  const propIndex = props && typeof props.pageIndex === 'number' ? props.pageIndex : null;
  const [urlIndex, setUrlIndex] = useState(null);

  useEffect(() => {
    let mounted = true;
    Linking.getInitialURL()
      .then((url) => {
        if (mounted) setUrlIndex(indexFromUrl(url));
      })
      .catch(() => {
        if (mounted) setUrlIndex(null);
      });
    return () => { mounted = false; };
  }, []);

  const i = propIndex !== null ? propIndex : (urlIndex !== null ? urlIndex : -1);
  const cap = i >= 0 && i < CAPTURES.length ? CAPTURES[i] : null;

  globalThis.__HS = cap ? cap.set : {};

  const capProps = cap && cap.props ? cap.props : {};
  const Comp = cap ? (COMP[cap.screen] || Blank) : Blank;
  const screenName = cap ? cap.screen : '__blank__';
  const routeName = cap ? (cap.route || screenName.replace(/Screen$/, '')) : '__blank__';
  const routeProp = { name: routeName, params: capProps };

  const navValue = useMemo(
    () => ({
      navigate: noop,
      push: noop,
      replace: noop,
      reset: noop,
      resetTo: noop,
      goBack: noop,
      go: noop,
      backToMain: noop,
      setTab: noop,
      setOptions: noop,
      addListener: () => (() => {}),
      current: routeName,
      currentTab: routeName,
      params: routeProp.params,
      route: routeProp,
${routeHelpers}
    }),
    [routeName]
  );

  const navigationProp = {
    ...navValue,
    navigate: noop,
    push: noop,
    goBack: noop,
    replace: noop,
    reset: noop,
    resetTo: noop,
    go: noop,
    backToMain: noop,
    setTab: noop,
    setOptions: noop,
    addListener: () => (() => {}),
${routeHelpers}
  };

  const appContextValue = useMemo(
    () => ({
      selfCheckFrequency: 'Every day',
      setSelfCheckFrequency: noop,
      autoStorageCleanup: 'Every day',
      setAutoStorageCleanup: noop,
      autoVirusUpdate: 'Using Wi-Fi or mobile data',
      setAutoVirusUpdate: noop,
      autoBlockingListUpdate: 'Using Wi-Fi or mobile data',
      setAutoBlockingListUpdate: noop,
      showRecommendedTools: true,
      setShowRecommendedTools: noop,
      showToolsSection: true,
      setShowToolsSection: noop,
    }),
    []
  );

  const themeContextValue = useMemo(
    () => ({
      isDark: false,
      toggleTheme: noop,
      colors: defaultThemeColors,
    }),
    []
  );

  if (propIndex === null && urlIndex === null) {
    return <View style={{ flex: 1, backgroundColor: '#FFFFFF' }} />;
  }

  return (
    <NavigationContext.Provider value={navValue}>
      <AppContext.Provider value={appContextValue}>
        <ThemeContext.Provider value={themeContextValue}>
          <NavigationContainer>
            <Stack.Navigator initialRouteName={routeName} screenOptions={{ headerShown: false, animation: 'none' }}>
              <Stack.Screen name={routeName}>
                {(p) => {
                  const mergedRoute = { ...p.route, name: routeName, params: routeProp.params };
                  const mergedNavigation = {
                    ...navigationProp,
                    ...p.navigation,
                    resetTo: noop,
                    go: noop,
                    backToMain: noop,
                    setTab: noop,
                    current: routeName,
                    currentTab: routeName,
                    params: routeProp.params,
                    route: mergedRoute,
${routeHelpers}
                  };
                  return (
                    <SafeAreaView style={{ flex: 1, backgroundColor: '#FFFFFF' }}>
                      <StatusBar barStyle="dark-content" />
                      <View style={{ flex: 1 }}>
                        <Boundary name={screenName}>
                          <Comp
                            {...capProps}
                            nav={navValue}
                            navigation={mergedNavigation}
                            navigate={mergedNavigation.navigate || noop}
                            push={mergedNavigation.push || noop}
                            replace={mergedNavigation.replace || noop}
                            resetTo={noop}
                            goBack={mergedNavigation.goBack || noop}
                            onNavigate={noop}
                            onBack={noop}
                            route={mergedRoute}
                            current={routeName}
                            params={routeProp.params}
                          />
                        </Boundary>
                      </View>
                    </SafeAreaView>
                  );
                }}
              </Stack.Screen>
            </Stack.Navigator>
          </NavigationContainer>
        </ThemeContext.Provider>
      </AppContext.Provider>
    </NavigationContext.Provider>
  );
}
`;
}

function buildCaptures(appDir, screens) {
  const captures = [];
  const routes = inferScreenRoutes(appDir, screens);
  for (const s of screens) {
    let code = '';
    for (const ext of ['.js', '.jsx']) { const p = path.join(appDir, s + ext); if (fs.existsSync(p)) { code = fs.readFileSync(p, 'utf8'); break; } }
    const states = code ? detectStates(code) : [{ label: 'default', set: {}, props: {} }];
    for (const st of states) captures.push({ screen: s, route: routes[s] || s.replace(/Screen$/, ''), label: st.label, set: st.set, props: st.props || {} });
  }
  return captures;
}

function installMissingDeps(output) {
  const found = new Set(); const re = /Unable to resolve module\s+['"]?([@\w][\w@/.\-]*)['"]?/g; let m;
  while ((m = re.exec(output)) !== null) { let p = m[1]; if (p.startsWith('.')) continue; p = p.startsWith('@') ? p.split('/').slice(0, 2).join('/') : p.split('/')[0]; if (!['react', 'react-native'].includes(p)) found.add(p); }
  if (!found.size) return [];
  const pkgs = [...found]; console.log(`    [deps] 安装: ${pkgs.join(', ')}`);
  try { execFileSync('npm', ['install', '--save', ...pkgs], { cwd: HOST_PROJECT, stdio: 'ignore', shell: true, timeout: 600000 }); } catch (e) {}
  return pkgs;
}

function ensureShoot2MetroConfig() {
  fs.mkdirSync(SHOOT2_TOOLS_DIR, { recursive: true });
  fs.mkdirSync(SHOOT2_METRO_TMP, { recursive: true });
  fs.writeFileSync(SHOOT2_METRO_CONFIG, `const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const projectRoot = path.resolve(__dirname, '..');

module.exports = mergeConfig(getDefaultConfig(projectRoot), {
  projectRoot,
  cacheStores: [],
  fileMapCacheDirectory: path.join(__dirname, '_shoot2_metro_file_map_cache'),
});
`, 'utf8');
}

function bundleApp() {
  ensureShoot2MetroConfig();
  const out = path.join(HOST_PROJECT, 'tools', '_b2.bundle');
  for (let attempt = 0; attempt < 3; attempt++) {
    const r = spawnSync(process.execPath,
      ['node_modules/react-native/cli.js', 'bundle', '--platform', 'android', '--dev', 'false',
        '--reset-cache', '--config', SHOOT2_METRO_CONFIG, '--entry-file', 'index.js', '--bundle-output', out, '--assets-dest', path.join(HOST_PROJECT, 'tools', '_b2assets')],
      {
        cwd: HOST_PROJECT,
        encoding: 'utf8',
        env: { ...process.env, TMP: SHOOT2_METRO_TMP, TEMP: SHOOT2_METRO_TMP },
        maxBuffer: 1 << 30,
        timeout: 300000,
        windowsHide: true,
      });
    if (r.status === 0 && fs.existsSync(out)) return out;
    const output = (r.stdout || '') + (r.stderr || '');
    if (!installMissingDeps(output).length) throw new Error('bundle 失败:\n' + output);
  }
  throw new Error('bundle 安装依赖后仍失败');
}
// 当前前台 app 包名: 用 ResumedActivity(本机 mCurrentFocus 不可靠, 会显示 AOD/NotificationShade)
function currentForeground() {
  let o = '';
  try { o = adb(['shell', 'dumpsys', 'activity', 'activities']).toString('utf8'); } catch { return null; }
  const m = o.match(/(?:top)?ResumedActivity[:=]\s*ActivityRecord\{[0-9a-fA-F]+\s+u\d+\s+([\w.]+)\//);
  return m ? m[1] : null;
}
function wakeScreen() {
  try { adb(['shell', 'input', 'keyevent', 'KEYCODE_WAKEUP']); } catch {}
  try { adb(['shell', 'wm', 'dismiss-keyguard']); } catch {}
}
// 启动指定页并截图. 截完校验前台确实是我们的 app(否则=崩溃/没起来截到桌面)→重试; 3次都不在前台则 ok:false(不保存桌面图)
async function launchPageAndShot(idx) {
  let png = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    adb([
      'shell', 'am', 'start', '-S', '-n', ACTIVITY,
      '-a', 'android.intent.action.VIEW',
      '-d', `shoot2://capture/${idx}`,
      '--ei', 'pageIndex', String(idx),
    ]);
    await sleep(COLD_START_MS);
    png = adb(['exec-out', 'screencap', '-p']); let prev = md5(png); const t0 = Date.now();
    while (Date.now() - t0 < WAIT_MS) {
      await sleep(400);
      const p = adb(['exec-out', 'screencap', '-p']);
      const h = md5(p);
      png = p;
      const elapsed = Date.now() - t0;
      if (elapsed >= MIN_RENDER_MS && h === prev) break;
      prev = h;
    }
    if (currentForeground() === PACKAGE) return { png, ok: true };
    wakeScreen(); await sleep(900);
  }
  return { png, ok: false };
}
function pushBundleTo(bundlePath) { adb(['push', bundlePath, DEVICE_BUNDLE_PATH]); }

async function main() {
  const apps = listApps();
  console.log(`[batch2] retry apps json: ${RETRY_APPS_JSON}`);
  console.log(`[batch2] 待处理 App: ${apps.length}  (BUNDLE_ONLY=${BUNDLE_ONLY})`);
  if (!BUNDLE_ONLY) { ensureAdbServer(); wakeScreen(); try { adb(['shell', 'svc', 'power', 'stayon', 'usb']); } catch {} try { adb(['shell', 'mkdir', '-p', DEVICE_BUNDLE_DIR]); } catch {} }
  const failures = []; const blanks = []; const dups = []; const renderFails = []; let done = 0;

  for (const { category, app, dir } of apps) {
    const allStems = fs.readdirSync(dir).filter((f) => /\.(jsx?)$/i.test(f) && f !== 'App.js').map(stem);
    const screens = resolveScreens(dir, allStems);
    const captures = buildCaptures(dir, screens);
    console.log(`\n[${done + failures.length + 1}/${apps.length}] ${category}/${app}  (${screens.length} 屏幕 / ${captures.length} 状态)`);
    const outDir = path.join(OUTPUT_ROOT, category, app);
    const componentPrefix = `__shoot2_${safeSegment(category)}_${safeSegment(app)}__`;
    let phase = 'prepare';
    try {
      phase = 'prepare';
      resetComponents();
      copyTreeTransformedWithPrefix(dir, COMPONENTS_DIR, componentPrefix);
      assertCaptureImportsExist(captures, componentPrefix);
      fs.writeFileSync(APP_JS, generateAdaptedAppJs(captures, './components', componentPrefix), 'utf8');
      sleepSync(300);
      phase = 'bundle';
      if (BUNDLE_ONLY) { bundleApp(); console.log('    [ok] bundle 通过'); done++; continue; }

      fs.rmSync(outDir, { recursive: true, force: true }); fs.mkdirSync(outDir, { recursive: true });
      const bundle = bundleApp();
      phase = 'push_bundle';
      pushBundleTo(bundle);
      phase = 'capture_blank';
      const blankRes = await launchPageAndShot(-1);
      const blankHash = blankRes.ok ? md5(blankRes.png) : null;

      const fileFor = (c) => c.label === 'default' ? `${c.screen}.png` : `${c.screen}__${c.label.replace(/[^\w=]/g, '_')}.png`;
      const clusters = {}; // screen -> [{state,file,hash,blank,render_failed}]
      phase = 'capture_pages';
      for (let i = 0; i < captures.length; i++) {
        const c = captures[i];
        const fname = fileFor(c);
        const { png, ok } = await launchPageAndShot(i);
        if (!ok) {
          // app 没在前台(崩溃/启动失败)→截到的是桌面/别的, 不保存, 标记渲染失败
          (clusters[c.screen] = clusters[c.screen] || []).push({ state: c.label, file: null, hash: null, render_failed: true });
          renderFails.push({ category, app, page: fname });
          process.stdout.write(`    [shot] ${fname}  (渲染失败: app未在前台/崩溃, 不保存)\n`);
          continue;
        }
        fs.writeFileSync(path.join(outDir, fname), png);
        const h = md5(png); const blank = blankHash !== null && h === blankHash;
        (clusters[c.screen] = clusters[c.screen] || []).push({ state: c.label, file: fname, hash: h, blank });
        if (blank) blanks.push({ category, app, page: fname });
        process.stdout.write(`    [shot] ${fname}${blank ? '  (空白/渲染失败)' : ''}\n`);
      }
      // 同屏幕内重复状态(注入未生效/本就一样)标记(忽略渲染失败的)
      for (const sc of Object.keys(clusters)) {
        const byH = {}; clusters[sc].forEach((s) => { if (s.hash) (byH[s.hash] = byH[s.hash] || []).push(s.state); });
        for (const grp of Object.values(byH)) if (grp.length > 1) dups.push({ category, app, screen: sc, states: grp });
      }
      // page_cluster.json: 每屏幕一 page_id + source_js 映射 + states
      const pages = Object.keys(clusters).map((screen, idx) => ({
        page_id: `P${idx + 1}`,
        screen,
        source_js: resolveSourceJs(dir, screen),
        states: clusters[screen].map((s) => s.render_failed
          ? { mockup: fileFor({ screen, label: s.state }).replace(/\.png$/, ''), state: s.state, screenshot: null, render_failed: true }
          : { mockup: s.file.replace(/\.png$/, ''), state: s.state, screenshot: s.file }),
      }));
      phase = 'write_page_cluster';
      fs.writeFileSync(path.join(outDir, 'page_cluster.json'), JSON.stringify({ pages }, null, 2), 'utf8');
      removeRetryApp(category, app);
      console.log(`    [ok] -> ${outDir}`);
      console.log(`    [queue] removed ${category}/${app} from ${RETRY_APPS_JSON}`);
      done++;
    } catch (e) {
      recordRetryFailure(category, app, phase, e);
      failures.push({ category, app, error: String(e.message || e).slice(0, 300) });
      console.log(`    [FAIL] ${category}/${app}: ${String(e.message || e).split('\n')[0]}`);
      console.log(`    [queue] saved failure for ${category}/${app} (${phase})`);
    }
  }
  console.log(`\n[batch2] 成功 ${done} / 失败 ${failures.length} / 空白页 ${blanks.length} / 渲染失败页 ${renderFails.length} / 重复状态项 ${dups.length}`);
  for (const f of failures) console.log(`  [FAIL] ${f.category}/${f.app}: ${f.error.split('\n')[0]}`);
  if (!BUNDLE_ONLY) {
    fs.mkdirSync(OUTPUT_ROOT, { recursive: true });
    fs.writeFileSync(path.join(OUTPUT_ROOT, '_summary.json'), JSON.stringify({ total: apps.length, done, failures, blank_pages: blanks, render_failed_pages: renderFails, duplicate_states: dups }, null, 2), 'utf8');
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
