"""
练习题: 从实现文件里找出每个函数，挖空成模板，再把用户写的代码拼回去。

每道练习题是一个函数，用 (文件, 函数名) 表示。类里的方法写成 "类名.方法名"。
"""
import ast
import os
import shutil
import tempfile
import textwrap

from quiz import MODULES, ROOT, run_pytest, test_file

# 运行测试需要的目录，复制到临时目录里，不动原文件
PACKAGES = ["common", "召回", "排序", "特征交叉", "序列", "多样性", "GNN", "transformer", "tests"]


def read_file(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as file:
        return file.read()


def module_files(source):
    """模块对应的实现文件。common/ 是一个目录，包含多个文件"""
    if source.endswith("/"):
        folder = os.path.join(ROOT, source)
        return [source + name for name in sorted(os.listdir(folder))
                if name.endswith(".py") and name != "__init__.py"]
    return [source]


def iter_functions(tree):
    """顶层函数和类里的方法: (函数名, 节点)。跳过 _private 方法"""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("__") or not item.name.startswith("_"):
                        yield f"{node.name}.{item.name}", item


def allowed_paths():
    return {path for _, _, source in MODULES for path in module_files(source)}


def find_function(path, qualname):
    # 只允许读实现文件，防止通过 path 读到别的文件
    if path not in allowed_paths():
        raise KeyError(f"不是练习文件: {path}")
    for name, node in iter_functions(ast.parse(read_file(path))):
        if name == qualname:
            return node
    raise KeyError(f"{path} 里没有 {qualname}")


def function_span(node):
    """函数的起止行号（从 1 开始，包含装饰器）"""
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return start, node.end_lineno


def summary(node):
    docstring = ast.get_docstring(node)
    return docstring.strip().splitlines()[0] if docstring else ""


def list_modules():
    modules = []
    for name, chapter, source in MODULES:
        files = []
        for path in module_files(source):
            functions = [{"name": qualname, "summary": summary(node)}
                         for qualname, node in iter_functions(ast.parse(read_file(path)))]
            files.append({"path": path, "functions": functions})
        modules.append({"name": name, "chapter": chapter, "source": source, "files": files})
    return modules


def reference_code(path, qualname):
    """参考实现，去掉缩进"""
    node = find_function(path, qualname)
    start, end = function_span(node)
    lines = read_file(path).splitlines()[start - 1:end]
    return textwrap.dedent("\n".join(lines)) + "\n"


def template_code(path, qualname):
    """挖空的模板: 保留函数签名和文档字符串，函数体换成 raise NotImplementedError"""
    node = find_function(path, qualname)
    start, _ = function_span(node)
    first = node.body[0]
    has_docstring = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                     and isinstance(first.value.value, str))
    header_end = first.end_lineno if has_docstring else first.lineno - 1
    lines = read_file(path).splitlines()
    header = lines[start - 1:header_end]
    body_indent = " " * first.col_offset
    header.append(f"{body_indent}# 在这里写你的实现")
    header.append(f"{body_indent}raise NotImplementedError")
    return textwrap.dedent("\n".join(header)) + "\n"


def check_user_code(qualname, code):
    """
    检查提交的代码能不能编译，并且定义了这个函数。
    返回 None 表示没问题，否则返回 {"line", "message"}
    """
    code = textwrap.dedent(code)
    try:
        # compile 比 ast.parse 多检查一些错误，比如函数外面的 return
        compile(code, "<练习代码>", "exec")
        tree = ast.parse(code)
    except SyntaxError as error:
        return {"line": error.lineno, "message": f"{type(error).__name__}: {error.msg}"}
    function_name = qualname.split(".")[-1]
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if function_name not in names:
        return {"line": 1, "message": f"代码里要保留函数 {function_name} 的定义（def {function_name}(...)）"}
    return None


def splice(path, qualname, code):
    """
    把用户的代码替换进原文件，返回 (新的文件内容, 用户代码第 1 行在文件里的行号)
    """
    node = find_function(path, qualname)
    start, end = function_span(node)
    indent = " " * node.col_offset
    user_lines = textwrap.dedent(code).rstrip().splitlines()
    new_lines = [indent + line if line.strip() else "" for line in user_lines]
    lines = read_file(path).splitlines()
    return "\n".join(lines[:start - 1] + new_lines + lines[end:]) + "\n", start


def make_workspace():
    """把代码复制到临时目录"""
    workspace = tempfile.mkdtemp(prefix="rec_practice_")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for package in PACKAGES:
        shutil.copytree(os.path.join(ROOT, package), os.path.join(workspace, package), ignore=ignore)
    return workspace


def run_exercise(module, path, qualname, code):
    """
    用用户的代码替换掉这个函数，运行模块的测试。
    返回 (测试结果, 用户代码第 1 行在文件里的行号)
    """
    new_text, code_start = splice(path, qualname, code)
    workspace = make_workspace()
    try:
        with open(os.path.join(workspace, path), "w", encoding="utf-8") as file:
            file.write(new_text)
        results = run_pytest([test_file(module)], cwd=workspace)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return results.get(f"test_{module}", []), code_start


def test_source(module, title):
    """测试函数的源码。title 是去掉 test_ 前缀的名字，参数化的题目带 [参数]"""
    function_name = "test_" + title.split("[")[0]
    path = test_file(module)
    source = read_file(path)
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start, end = function_span(node)
            return "\n".join(source.splitlines()[start - 1:end])
    return ""
