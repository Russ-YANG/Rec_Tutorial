"""
推荐算法小测验。

python quiz.py                 查看所有模块的完成情况
python quiz.py item_cf         查看某个模块每道题的结果
python quiz.py item_cf --watch 保存代码后自动重新测试
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))

# (模块名, 章节, 实现文件)。测试文件统一是 tests/test_<模块名>.py
MODULES = [
    ("ml_to_csv", "召回", "召回/ml_to_csv.py"),
    ("item_cf", "召回", "召回/item_cf.py"),
    ("user_cf", "召回", "召回/user_cf.py"),
    ("swing", "召回", "召回/swing.py"),
    ("matrix_completion", "召回", "召回/matrix_completion.py"),
    ("dssm", "召回", "召回/dssm.py"),
    ("dr", "召回", "召回/dr.py"),
    ("mmoe", "排序", "排序/mmoe.py"),
    ("fm", "特征交叉", "特征交叉/fm.py"),
    ("dcn", "特征交叉", "特征交叉/dcn.py"),
    ("senet", "特征交叉", "特征交叉/senet.py"),
    ("bilinear_interaction", "特征交叉", "特征交叉/bilinear_interaction.py"),
    ("fibinet", "特征交叉", "特征交叉/fibinet.py"),
    ("lhuc_ppnet", "特征交叉", "特征交叉/lhuc_ppnet.py"),
    ("din", "序列", "序列/din.py"),
    ("sim", "序列", "序列/sim.py"),
    ("dpp", "多样性", "多样性/dpp.py"),
    ("lightgcn", "GNN", "GNN/lightgcn.py"),
    ("transformer", "Transformer", "transformer.py"),
]


def test_file(name):
    return os.path.join("tests", f"test_{name}.py")


def run_pytest(test_files):
    """运行 pytest，返回 {测试文件名: [(题目, 是否通过, 失败信息)]}"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "report.xml")
        # 某个模块导入失败时，其他模块照常测试
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--continue-on-collection-errors", f"--junitxml={report_path}", *test_files]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if not os.path.exists(report_path):
            print(process.stdout + process.stderr)
            sys.exit("pytest 没有生成结果，检查一下是否安装了 pytest")
        tree = ET.parse(report_path)

    results = {}
    for case in tree.iter("testcase"):
        # 正常的题目: classname = tests.test_item_cf
        # 导入失败: classname 为空，name = tests.test_item_cf
        module = case.get("classname") or case.get("name")
        module = module.split(".")[-1]
        title = case.get("name")
        if title.startswith("test_"):
            title = title[len("test_"):]
        if case.find("skipped") is not None:
            continue
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is None:
            results.setdefault(module, []).append((title, True, "", ""))
        else:
            message = problem.get("message") or ""
            detail = (problem.text or "").strip()
            if not case.get("classname"):
                title = "导入失败"
                # 取报错的最后一行，比如 SyntaxError: expected ':'
                error_lines = [line for line in detail.splitlines() if line.startswith("E ")]
                if error_lines:
                    message = error_lines[-1][1:].strip()
            results.setdefault(module, []).append((title, False, message, detail))
    return results


def is_not_started(cases):
    failed = [case for case in cases if not case[1]]
    return len(failed) == len(cases) and all("NotImplementedError" in case[2] for case in failed)


def show_board():
    results = run_pytest([test_file(name) for name, _, _ in MODULES])
    finished = 0
    chapter = None
    print()
    for name, module_chapter, source in MODULES:
        if module_chapter != chapter:
            chapter = module_chapter
            print(f"【{chapter}】")
        cases = results.get(f"test_{name}", [])
        passed = sum(1 for case in cases if case[1])
        total = len(cases)
        if total == 0:
            status = "⬜ 没有题目"
        elif any(case[0] == "导入失败" for case in cases):
            status = "💥 导入失败"
        elif passed == total:
            status = "✅ 全部通过"
            finished += 1
        elif is_not_started(cases):
            status = "⬜ 还没开始"
        else:
            status = "❌ 还没通过"
        print(f"  {status}  {name:<22} {passed:>2}/{total:<2}  {source}")
    print(f"\n完成 {finished}/{len(MODULES)} 个模块。用 python quiz.py <模块名> 查看详细结果。\n")


def show_module(name, verbose):
    results = run_pytest([test_file(name)])
    cases = results.get(f"test_{name}", [])
    passed = sum(1 for case in cases if case[1])
    print(f"\n{name}  {passed}/{len(cases)}\n")
    for case in cases:
        if case[1]:
            print(f"  ✅ {case[0]}")
        else:
            print(f"  ❌ {case[0]}")
            message = case[2].splitlines()[0] if case[2] else ""
            print(f"       {message[:200]}")
            if verbose:
                for line in case[3].splitlines():
                    print(f"       {line}")
    if cases and passed == len(cases):
        print("\n🎉 全部通过！")
    elif not verbose and passed < len(cases):
        print("\n加上 -v 可以看到完整的报错。")
    print()


def watched_files(source):
    """实现文件所在的目录里所有 .py，加上对应的测试文件"""
    folder = os.path.dirname(os.path.join(ROOT, source))
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".py")]
    if folder == ROOT:
        files = [os.path.join(ROOT, source)]
    return files


def watch_module(name, source, verbose):
    files = watched_files(source) + [os.path.join(ROOT, test_file(name))]
    last_state = None
    print(f"监听 {source}，保存后自动重新测试。按 Ctrl+C 退出。")
    while True:
        state = [os.path.getmtime(file) for file in files if os.path.exists(file)]
        if state != last_state:
            last_state = state
            print("\033[2J\033[H", end="")  # 清屏
            print(time.strftime("%H:%M:%S"), end="")
            show_module(name, verbose)
            sys.stdout.flush()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="推荐算法小测验")
    parser.add_argument("module", nargs="?", help="模块名，比如 item_cf")
    parser.add_argument("-w", "--watch", action="store_true", help="保存后自动重新测试")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示完整报错")
    args = parser.parse_args()

    if args.module is None:
        show_board()
        return

    sources = {name: source for name, _, source in MODULES}
    if args.module not in sources:
        sys.exit(f"没有这个模块: {args.module}\n可选: {', '.join(sources)}")
    try:
        if args.watch:
            watch_module(args.module, sources[args.module], args.verbose)
        else:
            show_module(args.module, args.verbose)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
