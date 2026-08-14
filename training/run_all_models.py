#!/usr/bin/env python3
"""
Autopilot - sequential training of 4 YOLO models
    python -m training.run_all_models
Automatically archives results after each model. Survive crashes/resume.
"""

import os, sys, shutil, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from console.pipeline3_yolo import train_yolo, load_config, run_dirs

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
DIRS = run_dirs(PROJECT_DIR, CFG)
BASE = DIRS["yolo"]  # yolo_obb dir
TRAIN = os.path.join(BASE, "train")

MODELS = [
    ("v8n",  "yolov8n-obb.pt", "train_v8n_3cls"),
    ("v8s",  "yolov8s-obb.pt", "train_v8s_3cls"),
    ("v26n", "yolo26n-obb.pt", "train_v26n_3cls"),
    ("v26s", "yolo26s-obb.pt", "train_v26s_3cls"),
]

LOG_FILE = os.path.join(PROJECT_DIR, "autopilot_log.txt")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def check_done(name, archive):
    """Return True if this model already has an archived result with best.pt"""
    best = os.path.join(BASE, archive, "weights", "best.pt")
    if os.path.exists(best):
        log(f"SKIP {name} — already archived at {archive}")
        return True
    return False

def archive(name, archive):
    if not os.path.exists(TRAIN):
        log(f"WARN {name}: train dir missing, nothing to archive")
        return
    dst = os.path.join(BASE, archive)
    if os.path.exists(dst):
        log(f"WARN {name}: {archive} exists, removing")
        shutil.rmtree(dst)
    shutil.move(TRAIN, dst)
    log(f"ARCHIVE {name} → {archive}")

def kill_server():
    """Stop any running server on port 7860"""
    try:
        import requests
        r = requests.get("http://127.0.0.1:7860/", timeout=3)
        log("Server running, requesting shutdown...")
    except:
        log("No server running")
        return
    # Just kill python processes if server doesn't support shutdown
    import psutil, signal
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "app.py" in " ".join(proc.cmdline() or []) and "7860" in " ".join(proc.cmdline() or []):
                log(f"Killing server process pid={proc.pid}")
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
        except:
            pass
    time.sleep(2)

def main():
    log("========== AUTOPILOT START ==========")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Kill stale server to free GPU
    kill_server()

    for name, model_path, archive_name in MODELS:
        if check_done(name, archive_name):
            continue

        log(f"TRAIN START {name} ({model_path})")

        # Clean any stale train dir
        if os.path.exists(TRAIN):
            log(f"Removing stale train dir")
            shutil.rmtree(TRAIN)

        try:
            train_yolo(
                project_dir=PROJECT_DIR,
                cfg=CFG,
                model=model_path,
                imgsz="1024",
                epochs="200",
                device="0",
                resume=False,
            )
            log(f"TRAIN DONE {name}")
            archive(name, archive_name)
        except Exception as e:
            log(f"TRAIN ERROR {name}: {e}")
            # Still try to archive partial results
            if os.path.exists(TRAIN):
                fail_archive = archive_name + "_FAILED"
                try:
                    archive(name, fail_archive)
                except:
                    log(f"  Failed to archive partial results")
            raise

    log("========== AUTOPILOT ALL DONE ==========")

if __name__ == "__main__":
    main()
