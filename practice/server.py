"""
练习用的本地网页服务器。只监听 127.0.0.1，提交的代码会在本机运行。

python -m practice
"""
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from practice.analysis import analyze
from practice.exercises import (
    check_user_code, list_modules, module_files, reference_code, run_exercise, template_code, test_source,
)
from quiz import MODULES

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
MODULE_NAMES = {name for name, _, _ in MODULES}
MODULE_SOURCES = {name: source for name, _, source in MODULES}

# 同一时间只跑一组测试
run_lock = threading.Lock()


def submit(payload):
    module, path, qualname, code = payload["module"], payload["path"], payload["function"], payload["code"]
    if module not in MODULE_NAMES:
        raise ValueError(f"没有这个模块: {module}")
    if path not in module_files(MODULE_SOURCES[module]):
        raise ValueError(f"{path} 不属于模块 {module}")

    compile_error = check_user_code(qualname, code)
    if compile_error:
        return {"compile_error": compile_error}

    with run_lock:
        cases, code_start = run_exercise(module, path, qualname, code)

    code_length = len(code.rstrip().splitlines())
    tests = []
    for title, passed, message, detail in cases:
        test = {"title": title, "passed": passed}
        if not passed:
            test["analysis"] = analyze(detail, message, path, code_start, code_length, import_failed=title == "导入失败")
            test["source"] = test_source(module, title)
            test["detail"] = detail
        tests.append(test)
    return {"tests": tests, "passed": sum(test["passed"] for test in tests), "total": len(tests)}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, name):
        path = os.path.join(STATIC, name)
        if os.path.dirname(os.path.abspath(path)) != STATIC or not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, "rb") as file:
            body = file.read()
        self.send_response(200)
        content_type = CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream")
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send_static("index.html")
            elif url.path.startswith("/static/"):
                self.send_static(url.path[len("/static/"):])
            elif url.path == "/api/modules":
                self.send_json(list_modules())
            elif url.path == "/api/exercise":
                self.send_json({
                    "template": template_code(query["path"], query["function"]),
                    "reference": reference_code(query["path"], query["function"]),
                })
            else:
                self.send_error(404)
        except (KeyError, ValueError) as error:
            self.send_json({"error": str(error)}, status=400)

    def do_POST(self):
        if self.path != "/api/submit":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            self.send_json(submit(payload))
        except (KeyError, ValueError) as error:
            self.send_json({"error": str(error)}, status=400)
        except RuntimeError as error:
            self.send_json({"error": str(error)}, status=500)

    def log_message(self, format, *args):
        pass  # 不在终端打印每个请求


def main(port=8765, open_browser=True):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"练习页面: {url}  按 Ctrl+C 退出")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
