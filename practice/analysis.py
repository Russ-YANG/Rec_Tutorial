"""
把 pytest 的报错整理成: 错误类型、原因分析、出错位置。
"""
import re

FRAME = re.compile(r"^(\S+?\.py):(\d+):(?: (\w+))?", re.MULTILINE)

SHAPE_HINTS = ("size of tensor", "shape", "shapes cannot be multiplied", "dimension", "mat1 and mat2")


def error_lines(detail):
    """pytest 报错里以 E 开头的行，就是异常信息"""
    return [line[1:].strip() for line in detail.splitlines() if line.startswith("E ")]


def exception_type(detail):
    """最后一个 path:line: ErrorType 里的 ErrorType"""
    types = [match.group(3) for match in FRAME.finditer(detail) if match.group(3)]
    return types[-1] if types else ""


def user_error_line(detail, path, code_start, code_length):
    """出错位置如果在用户写的代码里，返回编辑器里的行号（从 1 开始）"""
    line = None
    for match in FRAME.finditer(detail):
        if match.group(1).replace("\\", "/").endswith(path):
            line_in_file = int(match.group(2))
            if code_start <= line_in_file < code_start + code_length:
                line = line_in_file - code_start + 1
    return line


def explain_assertion(lines, message):
    """断言失败: 尽量找出期望值和实际值"""
    first = lines[0] if lines else message
    if first.startswith("AssertionError:"):
        return "结果不对", first[len("AssertionError:"):].strip()

    compare = re.match(r"assert (.+?) == (.+)$", first)
    if compare:
        actual, expected = compare.group(1), compare.group(2)
        if "Size(" in actual or ".shape" in "\n".join(lines):
            return "形状不对", f"你的结果形状是 {actual}，期望是 {expected}。检查 view、transpose、unsqueeze 这些改形状的操作。"
        if "±" in expected:
            expected = expected.split("±")[0].strip()
            return "数值不对", f"你算出来的是 {actual}，期望是 {expected}。对照公式检查每一项。"
        return "结果不对", f"你的结果是 {actual}，期望是 {expected}。"

    joined = "\n".join(lines)
    if "allclose" in joined or "torch.equal" in joined:
        return "数值不对", "张量的数值和期望不一致。检查公式，也检查 dim 参数和广播是不是按预期进行。"
    if "isinstance" in joined:
        return "类型不对", "返回值或子模块的类型和期望不一样。"
    if "not in" in first or " in " in first:
        return "结果不对", f"检查没通过: {first}。看看是不是漏掉或多出了某些元素。"
    return "结果不对", f"检查没通过: {first}"


def analyze(detail, message, path, code_start, code_length, import_failed=False):
    """
    返回 {category, explanation, line, error}
    category: 给人看的错误类别
    explanation: 原因分析
    line: 用户代码里出错的行（编辑器行号），找不到时为 None
    error: 原始的异常信息
    """
    lines = error_lines(detail)
    error_type = exception_type(detail)
    error_text = lines[0] if lines else message
    line = user_error_line(detail, path, code_start, code_length)

    if import_failed:
        return {
            "category": "导入失败",
            "explanation": f"{message}。文件在导入时就出错了，测试一道都没跑。"
                           "检查缩进，以及是不是在函数外面写了代码。",
            "line": line,
            "error": message,
        }

    if "NotImplementedError" in detail and (error_type == "NotImplementedError" or "NotImplementedError" in message):
        category, explanation = "还没写完", "函数里还有 raise NotImplementedError，把它换成你的实现。"
    elif error_type == "AssertionError" or error_text.startswith("assert"):
        category, explanation = explain_assertion(lines, message)
        if line is None:
            explanation += " 代码能运行，但结果没通过测试的检查，看看下面的测试代码检查了什么。"
    elif error_type == "RuntimeError" and any(hint in error_text for hint in SHAPE_HINTS):
        category = "形状不匹配"
        explanation = ("两个张量的形状对不上。在出错的那一行前面 print(x.shape) 看看每个张量的形状，"
                       "常见原因是少了 unsqueeze、拼接用错了 dim，或者 view/reshape 的维度写错。")
    elif error_type == "IndexError":
        category, explanation = "下标越界", "访问了不存在的位置。检查循环范围和下标是从 0 还是从 1 开始。"
    elif error_type == "KeyError":
        category = "键不存在"
        explanation = f"字典里没有这个键（{error_text}）。检查用的是原始 id 还是映射后的下标，或者是不是没有先建好字典。"
    elif error_type == "NameError":
        name = re.search(r"name '(\w+)' is not defined", error_text)
        what = f" {name.group(1)}" if name else ""
        category = "名字没定义"
        explanation = f"用到了没定义的名字{what}。检查拼写，或者是不是需要用文件开头已经 import 的名字。"
    elif error_type == "AttributeError":
        category, explanation = "属性不存在", f"对象上没有这个属性或方法（{error_text}）。检查名字拼写和对象的类型。"
    elif error_type == "TypeError":
        category, explanation = "类型或参数不对", f"{error_text}。检查函数的参数个数、参数顺序和返回值的类型。"
    elif error_type == "ZeroDivisionError":
        category, explanation = "除以 0", "分母可能为 0。检查集合为空、长度为 0 这些边界情况。"
    elif error_type == "ValueError":
        category, explanation = "值不合法", f"{error_text}。检查输入的取值范围和形状。"
    elif error_type == "RuntimeError":
        category, explanation = "运行时错误", f"{error_text}。检查数据类型（float/long）和设备是否一致。"
    else:
        category = "运行出错"
        explanation = f"{error_type or '报错'}: {error_text}" if error_text else "运行时出错，看下面的完整报错。"

    if line is None and category not in ("还没写完", "数值不对", "结果不对", "形状不对", "类型不对"):
        explanation += " 报错发生在你写的函数之外，可能是你的函数返回值的形状或类型不对，导致后面用到它的代码出错。"

    return {"category": category, "explanation": explanation, "line": line, "error": error_text}
