// 推荐算法练习场：选函数 -> 写代码 -> 提交测试 -> 看结果和原因分析

const state = {
  modules: [],
  current: null,      // { module, path, func, summary }
  template: "",
  reference: "",
  editor: null,
  referenceViewer: null,
  errorLine: null,
  running: false,
};

const $ = (id) => document.getElementById(id);

// ---------- 本地存储：草稿、每道题的状态、上次打开的题 ----------

function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* 隐私模式下忽略 */ }
}
function storageRemove(key) {
  try { localStorage.removeItem(key); } catch { /* 忽略 */ }
}
function exerciseKey(ex) { return `${ex.module}|${ex.path}|${ex.func}`; }

// ---------- 编辑器 ----------

function createEditor(host, readOnly) {
  if (window.CodeMirror) {
    return CodeMirror(host, {
      mode: "python",
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      matchBrackets: true,
      readOnly,
      extraKeys: {
        Tab: (cm) => cm.somethingSelected() ? cm.indentSelection("add") : cm.replaceSelection("    "),
        "Shift-Tab": (cm) => cm.indentSelection("subtract"),
        "Cmd-Enter": submit,
        "Ctrl-Enter": submit,
      },
    });
  }
  // CDN 加载失败时退回普通文本框
  const area = document.createElement("textarea");
  area.className = "fallback";
  area.spellcheck = false;
  area.readOnly = readOnly;
  host.appendChild(area);
  return {
    getValue: () => area.value,
    setValue: (value) => { area.value = value; },
    on: (event, handler) => area.addEventListener(event === "change" ? "input" : event, handler),
    refresh: () => {},
    focus: () => area.focus(),
    setCursor: () => area.focus(),
    addLineClass: () => {},
    removeLineClass: () => {},
    scrollIntoView: () => {},
  };
}

function highlight(code) {
  // 测试代码等只读片段的语法高亮
  const container = document.createElement("div");
  if (window.CodeMirror && CodeMirror.runMode) {
    CodeMirror.runMode(code, "python", container);
  } else {
    container.textContent = code;
  }
  return container.innerHTML;
}

function codeBlock(code, { startLine = 1, hitLine = null } = {}) {
  const pre = document.createElement("pre");
  pre.className = "code-block";
  code.replace(/\n$/, "").split("\n").forEach((line, index) => {
    const number = startLine + index;
    const row = document.createElement("div");
    row.className = "ln" + (number === hitLine ? " hit" : "");
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = number;
    const text = document.createElement("span");
    text.className = "text";
    text.innerHTML = highlight(line) || " ";
    row.append(num, text);
    pre.appendChild(row);
  });
  return pre;
}

function markErrorLine(line) {
  if (state.errorLine !== null) {
    state.editor.removeLineClass(state.errorLine, "background", "error-line");
    state.errorLine = null;
  }
  if (line) {
    state.errorLine = line - 1;
    state.editor.addLineClass(state.errorLine, "background", "error-line");
  }
}

function jumpToLine(line) {
  state.editor.focus();
  state.editor.setCursor({ line: line - 1, ch: 0 });
  state.editor.scrollIntoView({ line: line - 1, ch: 0 }, 80);
}

// ---------- 左侧练习树 ----------

function functionStatus(ex) { return storageGet(`status:${exerciseKey(ex)}`); }

function renderTree() {
  const tree = $("tree");
  const query = $("search").value.trim().toLowerCase();
  tree.innerHTML = "";
  let chapter = null;

  for (const module of state.modules) {
    const files = module.files
      .map((file) => ({
        ...file,
        functions: file.functions.filter((fn) =>
          !query || fn.name.toLowerCase().includes(query) || module.name.includes(query) || fn.summary.toLowerCase().includes(query)),
      }))
      .filter((file) => file.functions.length);
    if (!files.length) continue;

    if (module.chapter !== chapter) {
      chapter = module.chapter;
      const heading = document.createElement("div");
      heading.className = "chapter";
      heading.textContent = chapter;
      tree.appendChild(heading);
    }

    const details = document.createElement("details");
    details.className = "module";
    const isCurrent = state.current && state.current.module === module.name;
    details.open = Boolean(query) || isCurrent;

    const all = module.files.flatMap((file) => file.functions.map((fn) => ({ module: module.name, path: file.path, func: fn.name })));
    const done = all.filter((ex) => functionStatus(ex) === "pass").length;

    const summary = document.createElement("summary");
    summary.innerHTML = `<span>${module.name}</span><span class="module-progress">${done}/${all.length}</span>`;
    details.appendChild(summary);

    for (const file of files) {
      if (module.files.length > 1) {
        const label = document.createElement("div");
        label.className = "file-label";
        label.textContent = file.path.split("/").pop();
        details.appendChild(label);
      }
      for (const fn of file.functions) {
        const ex = { module: module.name, path: file.path, func: fn.name, summary: fn.summary };
        const button = document.createElement("button");
        button.type = "button";
        button.className = "fn";
        button.title = fn.summary || fn.name;
        if (state.current && exerciseKey(state.current) === exerciseKey(ex)) button.classList.add("active");
        const status = functionStatus(ex);
        button.innerHTML = `<span class="dot ${status || ""}"></span><span class="fn-name"></span>`;
        button.querySelector(".fn-name").textContent = fn.name;
        button.addEventListener("click", () => openExercise(ex));
        details.appendChild(button);
      }
    }
    tree.appendChild(details);
  }

  if (!tree.children.length) {
    tree.innerHTML = `<p class="muted pad">没有找到包含“${query}”的函数。</p>`;
  }
}

// ---------- 打开一道题 ----------

async function openExercise(ex) {
  saveDraft();
  const params = new URLSearchParams({ path: ex.path, function: ex.func });
  const response = await fetch(`/api/exercise?${params}`);
  const data = await response.json();
  if (!response.ok) {
    showMessage(data.error || "读取练习失败");
    return;
  }

  state.current = ex;
  state.template = data.template;
  state.reference = data.reference;
  storageSet("last", JSON.stringify(ex));

  const draft = storageGet(`draft:${exerciseKey(ex)}`);
  state.editor.setValue(draft ?? data.template);
  markErrorLine(null);
  $("save-state").textContent = draft ? "已恢复上次的草稿" : "";
  $("function-name").textContent = ex.func;
  $("function-summary").textContent = ex.summary || "";
  $("crumbs").innerHTML = "";
  const parts = [...ex.path.split("/"), ex.func];
  parts.forEach((part, index) => {
    const last = index === parts.length - 1;
    const span = document.createElement(last ? "b" : "span");
    span.textContent = part;
    $("crumbs").append(span, last ? "" : " / ");
  });
  closeReference();
  renderEmptyResults();
  renderTree();
  state.editor.refresh();
}

let saveTimer = null;
function saveDraft() {
  if (!state.current) return;
  const code = state.editor.getValue();
  const key = `draft:${exerciseKey(state.current)}`;
  if (code === state.template) storageRemove(key);
  else storageSet(key, code);
}

// ---------- 结果区 ----------

function showMessage(text) {
  $("results").innerHTML = "";
  const banner = document.createElement("div");
  banner.className = "banner";
  banner.textContent = text;
  $("results").appendChild(banner);
}

function renderEmptyResults() {
  $("results-summary").textContent = "";
  const ex = state.current;
  $("results").innerHTML = `
    <div class="empty">
      <p>在上面实现 <code></code>，然后点“提交测试”（或按 ⌘/Ctrl + Enter）。</p>
      <p>提交时只替换这一个函数，同一个文件里的其他函数用参考实现，
      然后运行 <code></code> 里的全部题目。和这个函数无关的题目会直接通过。</p>
    </div>`;
  const codes = $("results").querySelectorAll("code");
  codes[0].textContent = ex.func;
  codes[1].textContent = `tests/test_${ex.module}.py`;
}

function renderRunning() {
  $("results-summary").textContent = "";
  $("results").innerHTML = `<div class="running"><span class="spinner"></span>正在运行 tests/test_${state.current.module}.py …</div>`;
}

function renderCompileError(error) {
  $("results-summary").textContent = "没有运行测试";
  $("results").innerHTML = "";
  const banner = document.createElement("div");
  banner.className = "banner";
  const text = document.createElement("span");
  text.textContent = `第 ${error.line} 行: ${error.message}`;
  const jump = document.createElement("button");
  jump.type = "button";
  jump.className = "btn small danger";
  jump.textContent = "跳到这一行";
  jump.addEventListener("click", () => jumpToLine(error.line));
  banner.append(text, jump);
  $("results").appendChild(banner);

  const lines = state.editor.getValue().split("\n");
  const start = Math.max(1, error.line - 2);
  $("results").appendChild(codeBlock(lines.slice(start - 1, error.line + 2).join("\n"), { startLine: start, hitLine: error.line }));
  markErrorLine(error.line);
}

function renderResults(data) {
  const results = $("results");
  results.innerHTML = "";
  const allPassed = data.passed === data.total && data.total > 0;
  $("results-summary").textContent = `${data.passed}/${data.total} 通过`;

  const summary = document.createElement("div");
  summary.className = `summary ${allPassed ? "pass" : "fail"}`;
  const percent = data.total ? (data.passed / data.total) * 100 : 0;
  summary.innerHTML = `<strong>${allPassed ? "全部通过" : `${data.total - data.passed} 道题没通过`}</strong>
    <span>${data.passed} / ${data.total}</span>
    <span class="bar"><span style="width:${percent}%"></span></span>`;
  results.appendChild(summary);

  const failures = data.tests.filter((test) => !test.passed);
  const userLines = state.editor.getValue().split("\n");
  failures.forEach((test, index) => results.appendChild(renderFailure(test, userLines, index === 0)));

  for (const test of data.tests.filter((t) => t.passed)) {
    const row = document.createElement("div");
    row.className = "test-row pass";
    row.innerHTML = `<span class="icon">✓</span><span></span>`;
    row.lastChild.textContent = test.title;
    results.appendChild(row);
  }

  const firstLine = failures.map((test) => test.analysis.line).find(Boolean);
  markErrorLine(firstLine || null);
}

function renderFailure(test, userLines, open) {
  const { analysis } = test;
  const details = document.createElement("details");
  details.className = "failure";
  details.open = open;

  const summary = document.createElement("summary");
  summary.innerHTML = `<span class="icon">✕</span><span class="title"></span><span class="chip"></span>`;
  summary.querySelector(".title").textContent = test.title;
  const chip = summary.querySelector(".chip");
  chip.textContent = analysis.category;
  if (analysis.category === "还没写完") chip.classList.add("todo");
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "failure-body";

  const explainBlock = document.createElement("div");
  explainBlock.innerHTML = `<div class="block-title">原因分析</div>`;
  const explain = document.createElement("p");
  explain.className = "explain";
  explain.textContent = analysis.explanation;
  explainBlock.appendChild(explain);
  body.appendChild(explainBlock);

  if (analysis.line) {
    const where = document.createElement("div");
    const title = document.createElement("div");
    title.className = "block-title";
    title.textContent = `你的代码 · 第 ${analysis.line} 行`;
    const jump = document.createElement("button");
    jump.type = "button";
    jump.className = "btn small ghost";
    jump.textContent = "跳到这一行";
    jump.addEventListener("click", () => jumpToLine(analysis.line));
    title.appendChild(jump);
    const start = Math.max(1, analysis.line - 2);
    where.append(title, codeBlock(userLines.slice(start - 1, analysis.line + 2).join("\n"), { startLine: start, hitLine: analysis.line }));
    body.appendChild(where);
  }

  if (test.source) {
    const testBlock = document.createElement("div");
    testBlock.innerHTML = `<div class="block-title">测试代码</div>`;
    testBlock.appendChild(codeBlock(test.source));
    body.appendChild(testBlock);
  }

  const raw = document.createElement("details");
  raw.className = "raw";
  raw.innerHTML = `<summary>完整报错</summary><pre></pre>`;
  raw.querySelector("pre").textContent = test.detail;
  body.appendChild(raw);

  details.appendChild(body);
  return details;
}

// ---------- 按钮 ----------

async function submit() {
  if (state.running || !state.current) return;
  saveDraft();
  state.running = true;
  $("btn-submit").disabled = true;
  renderRunning();
  const ex = state.current;
  try {
    const response = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ module: ex.module, path: ex.path, function: ex.func, code: state.editor.getValue() }),
    });
    const data = await response.json();
    if (exerciseKey(ex) !== exerciseKey(state.current)) return;  // 运行期间切换了题目
    if (!response.ok) {
      showMessage(data.error || "运行失败");
    } else if (data.compile_error) {
      renderCompileError(data.compile_error);
      storageSet(`status:${exerciseKey(ex)}`, "fail");
    } else {
      renderResults(data);
      storageSet(`status:${exerciseKey(ex)}`, data.passed === data.total ? "pass" : "fail");
    }
    renderTree();
  } catch (error) {
    showMessage("连不上本地服务器。确认终端里的 python -m practice 还在运行。");
  } finally {
    state.running = false;
    $("btn-submit").disabled = false;
  }
}

let resetArmed = null;
function reset() {
  const button = $("btn-reset");
  if (!resetArmed) {
    // 第一次点击只是确认，3 秒内再点一次才清空
    button.textContent = "再点一次确认";
    button.classList.add("danger");
    resetArmed = setTimeout(disarmReset, 3000);
    return;
  }
  disarmReset();
  state.editor.setValue(state.template);
  storageRemove(`draft:${exerciseKey(state.current)}`);
  $("save-state").textContent = "已重置为空白模板";
  markErrorLine(null);
  renderEmptyResults();
}
function disarmReset() {
  clearTimeout(resetArmed);
  resetArmed = null;
  $("btn-reset").textContent = "重置为空白";
  $("btn-reset").classList.remove("danger");
}

function toggleReference() {
  const panel = $("reference");
  if (!panel.hidden) { closeReference(); return; }
  panel.hidden = false;
  if (!state.referenceViewer) state.referenceViewer = createEditor($("reference-host"), true);
  state.referenceViewer.setValue(state.reference);
  state.referenceViewer.refresh();
  $("btn-reference").textContent = "隐藏参考实现";
}
function closeReference() {
  $("reference").hidden = true;
  $("btn-reference").textContent = "参考实现";
}

// ---------- 启动 ----------

async function start() {
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) $("submit-key").textContent = "Ctrl↵";
  state.editor = createEditor($("editor-host"), false);
  state.editor.on("change", () => {
    clearTimeout(saveTimer);
    $("save-state").textContent = "";
    saveTimer = setTimeout(() => { saveDraft(); $("save-state").textContent = "草稿已保存"; }, 600);
  });

  $("btn-submit").addEventListener("click", submit);
  $("btn-reset").addEventListener("click", reset);
  $("btn-reference").addEventListener("click", toggleReference);
  $("btn-close-reference").addEventListener("click", closeReference);
  $("search").addEventListener("input", renderTree);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); submit(); }
  });

  try {
    state.modules = await (await fetch("/api/modules")).json();
  } catch {
    $("tree").innerHTML = `<p class="muted pad">连不上本地服务器。在项目根目录运行 python -m practice。</p>`;
    return;
  }

  let last = null;
  try { last = JSON.parse(storageGet("last")); } catch { /* 忽略 */ }
  const exists = (ex) => ex && state.modules.some((m) => m.name === ex.module &&
    m.files.some((f) => f.path === ex.path && f.functions.some((fn) => fn.name === ex.func)));
  const first = exists(last) ? last : { module: "item_cf", path: "召回/item_cf.py", func: "item_similarity" };
  const module = state.modules.find((m) => m.name === first.module);
  first.summary = module.files.find((f) => f.path === first.path).functions.find((fn) => fn.name === first.func).summary;
  await openExercise(first);
}

start();
