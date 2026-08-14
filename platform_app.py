from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from lowalt_platform.api import create_platform_app
from lowalt_platform.settings import PlatformSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="低空遥感智能分析平台")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--allow-remote-without-auth", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote_without_auth:
        raise SystemExit("拒绝在无鉴权状态下监听远程地址")
    app = create_platform_app(PlatformSettings.from_project(args.project_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
