# -*- coding: utf-8 -*-
"""engine.dll 构建脚本（MSVC cl，系统自带，无需 CMake）。

用法: python build_engine.py
输出: 与本脚本同目录的 engine.dll
"""

import subprocess
import sys
from pathlib import Path

VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")


def find_cl():
    out = subprocess.run(
        [str(VSWHERE), "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True).stdout.strip().splitlines()
    for root in out:
        vcvars = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if vcvars.exists():
            return vcvars
    raise RuntimeError("未找到 MSVC VC 工具链（vcvars64.bat）")


def main():
    here = Path(__file__).resolve().parent
    src = here / "engine.cpp"
    dll = here / "engine.dll"
    vcvars = find_cl()
    bat = here / "_build_engine.bat"
    bat.write_text(
        f'@echo off\r\n'
        f'call "{vcvars}" >nul 2>&1\r\n'
        f'cd /d "{here}"\r\n'
        f'cl /nologo /utf-8 /O2 /std:c++17 /MT /LD "{src}" /Fe:"{dll}"\r\n',
        encoding="ascii")
    print("构建 engine.dll ...")
    r = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True,
                       encoding="gbk", errors="replace")
    bat.unlink(missing_ok=True)
    if r.returncode != 0 or not dll.exists():
        print(r.stdout)
        print(r.stderr)
        sys.exit("构建失败")
    (here / "engine.obj").unlink(missing_ok=True)
    print(f"构建成功: {dll}")


if __name__ == "__main__":
    main()
