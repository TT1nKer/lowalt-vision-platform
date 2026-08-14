#!/usr/bin/env python3
"""Windows console launcher with environment and dependency checks."""

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def fail(message):
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def check_environment():
    print(f"Python : {sys.executable}", flush=True)
    print(f"Version: {sys.version.split()[0]}", flush=True)

    try:
        import yaml  # noqa: F401
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except Exception as exc:
        fail(f"Missing console dependency: {exc}")

    try:
        import torch
        import torchvision

        cuda_ok = torch.cuda.is_available()
        print(f"Torch  : {torch.__version__}", flush=True)
        print(f"Vision : {torchvision.__version__}", flush=True)
        print(f"CUDA   : {cuda_ok}", flush=True)
        if cuda_ok:
            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
            scores = torch.tensor([0.9], device="cuda")
            torchvision.ops.nms(boxes, scores, 0.5)
            print("CUDA NMS: OK", flush=True)
    except ImportError:
        print("[WARN] PyTorch/torchvision not installed; web console can start, YOLO training cannot.", flush=True)
    except Exception as exc:
        fail(f"PyTorch/CUDA check failed: {exc}")


def open_when_ready(url):
    for _ in range(30):
        try:
            with urllib.request.urlopen(url, timeout=1):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(1)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(script_dir))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true", help="热更新：文件修改后自动重启（开发用）")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    app_path = project_dir / "app.py"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path

    if not app_path.is_file():
        fail(f"app.py not found: {app_path}")
    if not config_path.is_file():
        example = project_dir / "config.example.yaml"
        if config_path.name == "config.yaml" and example.is_file():
            config_path.write_bytes(example.read_bytes())
            print(f"Created config: {config_path}", flush=True)
        else:
            fail(f"Config not found: {config_path}")

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    check_environment()

    url = f"http://{args.host}:{args.port}"
    print(f"Project: {project_dir}", flush=True)
    print(f"Config : {config_path}", flush=True)
    print(f"Open   : {url}", flush=True)

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    command = [
        sys.executable,
        "-u",
        str(app_path),
        "--project-dir",
        str(project_dir),
        "--config",
        str(config_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        _run_with_reload(command, project_dir)
    else:
        raise SystemExit(subprocess.call(command, cwd=project_dir, env=os.environ.copy()))


def _run_with_reload(command, project_dir):
    """把 --reload 传给 app.py，让 uvicorn 自己管理热更新。"""
    command.append("--reload")
    raise SystemExit(subprocess.call(command, cwd=project_dir, env=os.environ.copy()))


if __name__ == "__main__":
    main()
