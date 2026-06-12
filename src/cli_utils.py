"""命令行工具辅助函数。"""

from __future__ import annotations

import argparse


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 的默认帮助标题替换为中文。"""

    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage:", "用法:")
            .replace("options:", "选项:")
            .replace("show this help message and exit", "显示帮助信息并退出")
        )

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:")
