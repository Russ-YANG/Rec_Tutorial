import argparse

from practice.server import main

parser = argparse.ArgumentParser(description="推荐算法练习页面")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
args = parser.parse_args()
main(port=args.port, open_browser=not args.no_browser)
