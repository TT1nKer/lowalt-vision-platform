#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低空遥感视觉AI自动化标注训练系统 — Web 后端

启动:
    python app.py --project-dir . --config config.yaml --port 7860

设计要点（针对旧版 log 显示崩坏的根因）：
- 页面 HTML 不再用 f-string 拼接动态内容。所有动态数据通过 /api/* 返回 JSON，
  前端用 textContent 写入，天然转义，日志里出现 < > & </pre> 都不会破坏页面。
- 子进程 stdout 实时按行收集；进度通过解析 "@@PROGRESS done total msg" 得到真实百分比，
  不再按行数瞎涨。
- 任务日志环形缓冲（保留末尾 N 行），前端增量轮询，长任务也不卡。
"""

import os
import sys
import json
import time
import html
import argparse
import threading
import subprocess
import queue
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse, RedirectResponse

from console.core import (load_config, cfg_get, run_dirs, image_dir, safe_load_json,
                  safe_write_json, ensure_dir, parse_progress)
from console.pipeline2_review import (
    load_index, count_index, save_label, bake_one_mp,
    prepare_target, target_layer_path, target_geom_path,
    LABELS, POSITIVE_LABELS, NEGATIVE_LABELS,
)
from console.acceptance_workflow import read_acceptance, save_acceptance_row, validate_acceptance
from console.golden_review import read_review_rows, save_review_row, validate_golden_package

APP = FastAPI(title="低空遥感视觉AI自动化标注训练系统")
ARGS = None
CFG = None
JOBS = {}
LOCK = threading.Lock()


def _safe_join(base, *parts):
    """Join a user-controlled relative path while keeping it inside base."""
    base_abs = os.path.abspath(base)
    candidate = os.path.abspath(os.path.join(base_abs, *parts))
    try:
        if os.path.commonpath([base_abs, candidate]) != base_abs:
            raise HTTPException(403, detail="path outside allowed directory")
    except ValueError:
        raise HTTPException(403, detail="invalid path")
    return candidate

LOG_KEEP = 50000  # 每个任务保留的日志行数上限

# ---- 进程内索引缓存 ----
# 92 万 target 时，每次 api_review 都全量读 jsonl 两遍 + O(n) 查找会很慢。
# 这里把索引加载一次缓存在内存，并预建几个查找表。文件变化（mtime）时自动失效重载。
_INDEX_CACHE = {"mtime": None, "list": [], "by_overlay": {}, "by_image": {}, "by_tid": {}}
_INDEX_LOCK = threading.Lock()

# ---- 内存状态存储 ----
# 打标状态用内存 dict 托管，避免每次打标全量重写 state.json（5万条时那要 0.5s+，
# 几十万条会更慢）。打标 = 更新内存 + 追加 jsonl（O(1)），state.json 周期性落盘。
class StateStore:
    def __init__(self):
        self.data = {}          # target_id -> rec
        self.loaded_key = None  # 当前已加载的 run 标识，切 run 时重载
        self.dirty = 0
        self.lock = threading.Lock()
        self.version = 0

    def _ensure(self, dirs):
        key = dirs["state"]
        if self.loaded_key == key:
            return
        # 从 state.json 恢复；再用 jsonl 追平（jsonl 是权威增量日志）
        data = safe_load_json(dirs["state"], {})
        jl = dirs["jsonl"]
        if os.path.exists(jl):
            try:
                with open(jl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                            tid = r.get("target_id")
                            if tid:
                                data[tid] = {k: v for k, v in r.items() if k != "target_id"}
                        except Exception:
                            continue
            except Exception:
                pass
        self.data = data
        self.loaded_key = key
        self.version += 1

    def all(self, dirs):
        with self.lock:
            self._ensure(dirs)
            return self.data

    def snapshot(self, dirs):
        with self.lock:
            self._ensure(dirs)
            return dict(self.data)

    def get_label(self, dirs, tid):
        with self.lock:
            self._ensure(dirs)
            return self.data.get(tid, {}).get("label", "")

    def set(self, dirs, tid, rec):
        with self.lock:
            self._ensure(dirs)
            self.data[tid] = rec
            self.version += 1
            # 追加 jsonl（增量，便宜）
            ensure_dir(dirs["review"])
            with open(dirs["jsonl"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"target_id": tid, **rec}, ensure_ascii=False) + "\n")
            self.dirty += 1
            # 每 100 次落盘一次 state.json（兜底快照；jsonl 已保证不丢）
            if self.dirty >= 100:
                self._flush(dirs)

    def _flush(self, dirs):
        try:
            safe_write_json(dirs["state"], self.data)
            self.dirty = 0
        except Exception:
            pass

    def flush(self, dirs):
        with self.lock:
            self._ensure(dirs)
            self._flush(dirs)


STATE = StateStore()

# ---- 后台预热队列状态（统一：预缓冲 + 全量铺底，一套线程，优先级取任务）----
# cursor: 前端汇报的当前浏览位置（当前 mode 列表里的下标）
# direction: 最近翻页方向(+1/-1/0)，顺序浏览时据此往前方多缓冲
# mode: 当前筛选模式，决定排序基于哪个列表
# random_ids: 随机模式下，前端预测的"接下来要跳到的目标 id"批，精确预缓冲
# window: 预缓冲窗口大小（前方/附近优先烤这么多）
_BAKE = {
    "running": False,
    "stop": False,
    "done": 0,
    "total": 0,
    "skipped": 0,
    "failed": 0,
    "cursor": 0,
    "direction": 1,
    "mode": "all",
    "random_ids": [],
    "window": 50,
    "thread": None,
}
_BAKE_LOCK = threading.Lock()

# ---- Gemma 自动打标状态 ----
# 后端独立线程：扫所有未筛选 target，让 Gemma 给标签，自动跳过已打标的。
# 可随时停止/恢复(下次启动从未打标继续)。每条记录会写 is_auto=True 区分。
_GAUTO = {
    "running": False,
    "stop": False,
    "done": 0,         # 本次启动以来成功打的数
    "skipped": 0,      # 因为 Gemma 没给出有效标签而跳过的数
    "failed": 0,       # Gemma 调用失败的数
    "total": 0,        # 启动时统计的待办数
    "last_label": "",
    "last_target": "",
    "last_error": "",
    "thread": None,
}
_GAUTO_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# 任务模型
# ----------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    action: str
    status: str = "pending"          # pending / running / done / error
    progress: int = 0                # 0-100，来自真实进度
    done: int = 0
    total: int = 0
    log: list = field(default_factory=list)
    _log_total: int = 0              # 单调递增的总行数（环形缓冲区截断也不回退）
    _log_start: int = 0              # 当前缓冲区第一行的绝对游标
    error: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def add(self, msg):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.log.append(entry)
        self._log_total += 1
        if len(self.log) > LOG_KEEP:
            removed = len(self.log) - LOG_KEEP
            self.log = self.log[removed:]
            self._log_start += removed


def cfg():
    global CFG
    if CFG is None:
        CFG = load_config(ARGS.config_path)
    return CFG


# 当前选定的 prompt(run)。None = 用 config 的默认 text_prompt。
# 网页可切换，从而只对某一个 prompt 的 target 打标。
CURRENT_PROMPT = None
_PROMPT_LOADED = False


def _prompt_persist_path():
    root = run_dirs(ARGS.project_dir, cfg())
    return os.path.join(os.path.dirname(root["base"]), ".current_prompt")


def current_prompt():
    """当前选定的 prompt。持久化到文件,服务器重启后保持上次选择,
    防止"标在 run A、重启后导出默默跑到 config 默认 run"的错位。"""
    global CURRENT_PROMPT, _PROMPT_LOADED
    if CURRENT_PROMPT is None and not _PROMPT_LOADED:
        _PROMPT_LOADED = True
        try:
            p = _prompt_persist_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    v = f.read().strip()
                if v:
                    CURRENT_PROMPT = v
        except Exception:
            pass
    return CURRENT_PROMPT if CURRENT_PROMPT is not None else cfg_get(cfg(), "sam3.text_prompt", "target")


def _cfg_with_prompt(prompt):
    import copy
    c = copy.deepcopy(cfg())
    c.setdefault("sam3", {})["text_prompt"] = prompt
    return c


def ds():
    return run_dirs(ARGS.project_dir, _cfg_with_prompt(current_prompt()))


def list_runs():
    """扫 results_root 下所有 run，返回 [{slug, prompt, has_index}]。"""
    root = run_dirs(ARGS.project_dir, cfg())
    results_root = os.path.dirname(root["base"])
    out = []
    if os.path.isdir(results_root):
        for name in sorted(os.listdir(results_root)):
            base = os.path.join(results_root, name)
            if not os.path.isdir(base):
                continue
            meta = safe_load_json(os.path.join(base, "run_meta.json"), {})
            prompt_mode = meta.get("prompt_mode", "joined")
            if prompt_mode == "batch":
                idx_jsonl = os.path.join(base, "merged", "review_cache", "target_index.jsonl")
                idx_json = os.path.join(base, "merged", "review_cache", "target_index.json")
            else:
                idx_jsonl = os.path.join(base, "review_cache", "target_index.jsonl")
                idx_json = os.path.join(base, "review_cache", "target_index.json")
            out.append({
                "slug": name,
                "prompt": meta.get("text_prompt", name),
                "prompt_mode": prompt_mode,
                "has_index": os.path.exists(idx_jsonl) or os.path.exists(idx_json),
            })
    return out



def _index_mtime():
    d = ds()
    for key in ("index", "index_legacy"):
        p = d.get(key)
        if p and os.path.exists(p):
            return os.path.getmtime(p)
    return None


def get_index():
    """返回缓存的索引列表。索引文件 mtime 变化时自动重载并重建查找表。"""
    mt = _index_mtime()
    with _INDEX_LOCK:
        if _INDEX_CACHE["mtime"] != mt:
            lst = load_index(ARGS.project_dir, _cfg_with_prompt(current_prompt()))
            for pos, x in enumerate(lst):
                x["_pos"] = pos
            by_overlay = {x["overlay_file"]: x for x in lst}
            by_tid = {x["target_id"]: x for x in lst}
            by_image = {}
            for x in lst:
                by_image.setdefault(x["image"], []).append(x)
            _INDEX_CACHE.update(mtime=mt, list=lst, by_overlay=by_overlay,
                                by_image=by_image, by_tid=by_tid)
        return _INDEX_CACHE["list"]


def index_by_tid(tid):
    get_index()
    return _INDEX_CACHE.get("by_tid", {}).get(tid)


def index_by_overlay(name):
    get_index()
    return _INDEX_CACHE["by_overlay"].get(name)


def index_by_image(image):
    get_index()
    return _INDEX_CACHE["by_image"].get(image, [])


# ----------------------------------------------------------------------------
# 子进程执行
# ----------------------------------------------------------------------------

def run_script(job, script, step, extra=None):
    # 子进程读磁盘状态,先把内存里的打标落盘(双保险,jsonl 合并是主修复)
    try:
        STATE.flush(ds())
    except Exception:
        pass
    cmd = [sys.executable, "-u", os.path.join(ARGS.app_dir, script), step,
           "--project-dir", ARGS.project_dir, "--config", ARGS.config_path]
    # 把当前选定的 prompt 传给子进程，使其操作正确的 run
    cur = current_prompt()
    cmd += ["--prompt", cur]
    if extra:
        cmd += extra
    job.add("运行命令: " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARGS.app_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    p = subprocess.Popen(cmd, cwd=ARGS.project_dir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env, bufsize=1)
    job._process = p  # 保存引用，异常时强制 kill
    import re as _re
    _ansi = _re.compile(r'\x1b\[[0-9;]*m')
    _ansik = _re.compile(r'\x1b\[K')
    output = queue.Queue()
    sentinel = object()

    def _read_output():
        try:
            for raw in p.stdout:
                output.put(raw)
        finally:
            output.put(sentinel)

    threading.Thread(target=_read_output, daemon=True).start()
    timeout_key = "web.train_timeout_seconds" if step == "train" else "web.job_timeout_seconds"
    timeout = int(cfg_get(cfg(), timeout_key, 0 if step == "train" else 1800))
    deadline = time.monotonic() + timeout if timeout > 0 else None
    if deadline is None:
        job.add("训练任务不设固定总时限，可随时手动取消")
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            p.kill()
            p.wait()
            raise RuntimeError(f"子进程超时（{timeout} 秒），已强制终止")
        try:
            line = output.get(timeout=0.5)
        except queue.Empty:
            if p.poll() is not None:
                continue
            continue
        if line is sentinel:
            break
        line = line.rstrip("\n")
        line = _ansik.sub('', line)
        line = _ansi.sub('', line)  # 直接删 ANSI，不做 HTML
        prog = parse_progress(line)
        if prog:
            done, total, msg = prog
            job.done, job.total = done, total
            job.progress = int(done * 100 / total) if total else 0
            last_milestone = int((job.done - 1) * 100 / total) if total else 0
            milestone = int(job.done * 100 / total) if total else 0
            if (done % 2000 == 0 or milestone // 10 > last_milestone // 10
                    or done == total):
                job.add(f"进度 {done}/{total} ({milestone}%) {msg}")
        else:
            job.add(line)
    rc = p.wait(timeout=5)
    job.add(f"子进程结束 rc={rc}")
    if rc != 0:
        raise RuntimeError(f"命令失败 rc={rc}")
    job.progress = 100


def runner(job_id, action, params):
    job = JOBS[job_id]
    with LOCK:
        if job.status == "cancelled":
            return
        job.status = "running"
    try:
        job.add(f"开始任务: {action}")
        if action.startswith("p1_"):
            step = action[3:]
            run_script(job, "console/pipeline1_sam3.py", step)
        elif action == "p2_index":
            run_script(job, "console/pipeline2_review.py", "index")
        elif action == "p2_overlay":
            extra = ["--force"] if params.get("force") else []
            run_script(job, "console/pipeline2_review.py", "overlay", extra)
        elif action == "p3_export":
            fmt = params.get("fmt", "obb")
            run_script(job, "console/pipeline3_yolo.py", "export", ["--fmt", str(fmt)])
        elif action == "p3_audit":
            run_script(job, "console/pipeline3_yolo.py", "audit", ["--no-fail"])
        elif action == "p3_sanitize":
            run_script(job, "console/pipeline3_yolo.py", "sanitize")
        elif action == "p3_train":
            extra = []
            for k in ("model", "imgsz", "epochs"):
                if params.get(k):
                    extra += [f"--{k}", str(params[k])]
            if params.get("resume"):
                extra.append("--resume")
            run_script(job, "console/pipeline3_yolo.py", "train", extra)
        elif action == "p3_predict":
            model = str(params.get("model", "")).strip()
            source = str(params.get("source", "")).strip()
            if not model or not source:
                raise ValueError("模型测试需要 model 和 source")
            extra = ["--model", model, "--source", source,
                     "--conf", str(params.get("conf", 0.25))]
            if params.get("imgsz"):
                extra += ["--imgsz", str(params["imgsz"])]
            run_script(job, "console/pipeline3_yolo.py", "predict", extra)
        elif action == "p3_eval":
            extra = ["--model", str(params.get("model", "")),
                     "--split", str(params.get("split", "test"))]
            if params.get("imgsz"):
                extra += ["--imgsz", str(params["imgsz"])]
            run_script(job, "console/pipeline3_yolo.py", "eval", extra)
        elif action == "p3_compare":
            extra = ["--baseline", str(params.get("baseline", "")),
                     "--trained", str(params.get("trained", "")),
                     "--split", str(params.get("split", "test"))]
            if params.get("imgsz"):
                extra += ["--imgsz", str(params["imgsz"])]
            run_script(job, "console/pipeline3_yolo.py", "compare", extra)
        elif action == "p4_predict_targets":
            extra = ["--model", str(params.get("model", "")),
                     "--conf", str(params.get("conf", 0.3))]
            if params.get("imgsz"):
                extra += ["--imgsz", str(params["imgsz"])]
            run_script(job, "console/pipeline3_yolo.py", "predict_targets", extra)
        elif action == "pipeline_full":
            fmt = params.get("fmt", "obb")
            job.add("===== 1/3: Pipeline1 识别 =====")
            run_script(job, "console/pipeline1_sam3.py", "all")
            job.add("===== 2/3: 构建索引 =====")
            run_script(job, "console/pipeline2_review.py", "index")
            index = get_index()
            if not index:
                raise RuntimeError("识别与索引完成后仍无 target，请检查图片目录和 SAM3 服务")
            job.add(f"索引已生成 {len(index)} 个 target")
            job.add(f"===== 3/3: 导出 YOLO-{fmt.upper()} =====")
            run_script(job, "console/pipeline3_yolo.py", "export", ["--fmt", str(fmt)])
        elif action == "pipeline_full_both":
            job.add("===== 1/4: Pipeline1 识别 =====")
            run_script(job, "console/pipeline1_sam3.py", "all")
            job.add("===== 2/4: 构建索引 =====")
            run_script(job, "console/pipeline2_review.py", "index")
            index = get_index()
            if not index:
                raise RuntimeError("识别与索引完成后仍无 target")
            job.add(f"索引已生成 {len(index)} 个 target")
            job.add("===== 3/4: 导出 YOLO-OBB =====")
            run_script(job, "console/pipeline3_yolo.py", "export", ["--fmt", "obb"])
            job.add("===== 4/4: 导出 YOLO-seg =====")
            run_script(job, "console/pipeline3_yolo.py", "export", ["--fmt", "seg"])
        elif action == "clip_scan":
            from console.clip_service import scan_images as _scan
            feature_id = params.get("feature_id", "")
            job.add(f"CLIP 全量扫描: feature_id={feature_id}")
            def _prog(done, total, msg=""):
                job.done = done
                job.total = total
                job.progress = int(done * 100 / total) if total else 0
                job.add(f"进度 {done}/{total} {msg}")
            result = _scan(ARGS.project_dir, cfg(), feature_id, progress=_prog)
            if result["ok"]:
                job.add(f"扫描完成: {result['matched_images']}/{result['total_images']} 张图匹配, 共 {result['total_targets']} 个 target")
                if result.get("errors"):
                    for err in result["errors"]:
                        job.add(f"  错误: {err}")
            else:
                raise RuntimeError(result.get("error", "CLIP 扫描失败"))
        else:
            raise ValueError(f"未知任务: {action}")
        job.status = "done"
        job.progress = 100
        job.add("任务完成")
    except Exception as e:
        if job.status == "cancelled":
            return
        job.status = "error"
        job.error = str(e)
        # 清理：强制终止子进程，防止僵尸进程吃内存
        _proc = getattr(job, '_process', None)
        if _proc is not None and _proc.poll() is None:
            try:
                _proc.kill()
                _proc.wait(timeout=5)
            except Exception:
                pass
        # 标出失败步骤
        for marker in ["===== 1/3:", "===== 2/3:", "===== 3/3:", "===== 1/4:", "===== 2/4:", "===== 3/4:", "===== 4/4:"]:
            if marker in str(job.log[-3:]) if len(job.log) >= 3 else "":
                job.add("❌ 上一步失败，后续步骤已跳过")
                break
        job.add(f"错误: {e}")


def start_job(action, params):
    jid = uuid.uuid4().hex[:12]
    job = Job(id=jid, action=action)
    with LOCK:
        finished = [jid for jid, old in JOBS.items()
                    if old.status not in ("pending", "running")]
        for old_id in finished[:-100]:
            JOBS.pop(old_id, None)
        JOBS[jid] = job
    threading.Thread(target=runner, args=(jid, action, params), daemon=True).start()
    return jid


# ----------------------------------------------------------------------------
# 页面：返回静态外壳，数据全走 API
# ----------------------------------------------------------------------------

def _read_web(name):
    path = os.path.join(ARGS.app_dir, "web", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@APP.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(_read_web("index.html"))


@APP.get("/review", response_class=HTMLResponse)
def review_page():
    return HTMLResponse(_read_web("review.html"))


@APP.get("/files", response_class=HTMLResponse)
def files_page():
    return HTMLResponse(_read_web("files.html"))


@APP.get("/test", response_class=HTMLResponse)
def test_page():
    return HTMLResponse(_read_web("test.html"))


def _golden_review_dir():
    return os.path.join(ARGS.project_dir, "quality", "golden_review_v2")


@APP.get("/golden-review", response_class=HTMLResponse)
def golden_review_page():
    workspace = os.path.join(getattr(ARGS, "app_dir", ARGS.project_dir), "web", "golden_review.html")
    if os.path.isfile(workspace):
        return HTMLResponse(_read_web("golden_review.html"))
    import re
    path = os.path.join(_golden_review_dir(), "index.html")
    if not os.path.isfile(path):
        raise HTTPException(404, detail="golden review package not generated")
    with open(path, "r", encoding="utf-8") as handle:
        document = handle.read()
    # /golden-review has no trailing slash, so browser-relative preview URLs
    # would otherwise resolve to /previews/... and bypass the asset route.
    document = document.replace('src="previews/', 'src="/golden-review/previews/')
    document = re.sub(r"<title>.*?</title>", "<title>金标测试集复核包</title>", document, count=1, flags=re.DOTALL)
    document = re.sub(
        r"<header>.*?</header>",
        "<header><h1>金标测试集复核包</h1><p>Gemma 已完成机器预审。请优先处理 fail/needs_review，再由人工抽检和独立批准；机器结果不等于人工签署。</p><p><a href=\"/golden-review/gemma_review.csv\">下载 Gemma 风险清单</a> · <a href=\"/api/report/golden\">查看审核状态</a></p></header>",
        document,
        count=1,
        flags=re.DOTALL,
    )
    return HTMLResponse(document)


@APP.get("/api/golden-review")
def api_golden_review():
    root = _golden_review_dir()
    rows = read_review_rows(root)
    counts = {status: 0 for status in ("pending", "approved", "needs_correction", "corrected")}
    for row in rows:
        status = str(row.get("review_status", "pending")).strip().lower()
        counts[status] = counts.get(status, 0) + 1
    return {
        "rows": rows,
        "counts": counts,
        "validation": validate_golden_package(root),
        "package": os.path.basename(root),
    }


@APP.put("/api/golden-review/{index}")
async def api_golden_review_save(index: int, request: Request):
    root = _golden_review_dir()
    try:
        row = save_review_row(root, index, await request.json())
    except (ValueError, IndexError, FileNotFoundError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    report = validate_golden_package(root)
    return {
        "row": row,
        "completed_count": report["completed_count"],
        "pending_count": report["pending_count"],
        "status": report["status"],
    }


@APP.get("/golden-review/{name:path}")
def golden_review_asset(name: str):
    path = _safe_join(_golden_review_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(404, detail="golden review asset not found")
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


def _recovery_comparison_dir():
    return os.path.join(ARGS.project_dir, "quality", "recovery_comparison_v1")


@APP.get("/recovery-comparison", response_class=HTMLResponse)
def recovery_comparison_page():
    path = os.path.join(_recovery_comparison_dir(), "index.html")
    if not os.path.isfile(path):
        raise HTTPException(404, detail="recovery comparison not generated")
    with open(path, "r", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


@APP.get("/recovery-comparison/{name:path}")
def recovery_comparison_asset(name: str):
    path = _safe_join(_recovery_comparison_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(404, detail="recovery comparison asset not found")
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


def _business_shadow_dir():
    return os.path.join(ARGS.project_dir, "quality", "business_shadow_v1")


def _calibrated_shadow_dir():
    return os.path.join(ARGS.project_dir, "quality", "business_shadow_calibrated_v1")


def _space_classifier_shadow_dir():
    return os.path.join(ARGS.project_dir, "quality", "space_classifier_shadow_v2")


@APP.get("/space-classifier-shadow", response_class=HTMLResponse)
def space_classifier_shadow_page():
    root = _space_classifier_shadow_dir()
    images = sorted(name for name in os.listdir(root) if name.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(root) else []
    if not images:
        raise HTTPException(404, detail="space classifier shadow results not generated")
    cards = "".join(
        f'<figure><img src="/space-classifier-shadow/{html.escape(name)}" loading="lazy"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in images
    )
    return HTMLResponse(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>固定车位分类候选</title><style>body{margin:0;padding:20px;background:#f3f5f7;color:#202934;font:14px "Microsoft YaHei",sans-serif}'
        'header{margin-bottom:16px}h1{font-size:20px;margin:0 0 6px}p{margin:0;color:#657180}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}'
        'figure{margin:0;padding:8px;background:#fff;border:1px solid #d8dde4}img{display:block;width:100%;height:auto}figcaption{padding:7px 2px 0;font-size:12px;color:#657180}'
        '@media(max-width:800px){main{grid-template-columns:1fr}}</style><header><h1>固定车位分类候选</h1><p>红=occupied，绿=empty，灰=unknown；这是当前应直接检查的业务结果。</p></header><main>'
        + cards + '</main></html>'
    )


@APP.get("/space-classifier-shadow/{name:path}")
def space_classifier_shadow_asset(name: str):
    path = _safe_join(_space_classifier_shadow_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(404, detail="space classifier shadow asset not found")
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


@APP.get("/business-shadow", response_class=HTMLResponse)
def business_shadow_page():
    root = _business_shadow_dir()
    path = os.path.join(root, "index.html")
    if not os.path.isfile(path):
        raise HTTPException(404, detail="business shadow package not generated")
    with open(path, "r", encoding="utf-8") as handle:
        document = handle.read()
    banner = ('<p style="padding:12px 5vw;background:#fff4d6;border-bottom:1px solid #ead39a">'
              '候选模型仅用于业务影子验证，尚未发布。请完成 30 张人工验收并由独立批准人签署。</p>')
    document = document.replace("<main>", banner + '<main>', 1)
    return HTMLResponse(document)


@APP.get("/business-shadow/{name:path}")
def business_shadow_asset(name: str):
    path = _safe_join(_business_shadow_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(404, detail="business shadow asset not found")
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


@APP.get("/business-shadow-calibrated", response_class=HTMLResponse)
def business_shadow_calibrated_page():
    path = os.path.join(_calibrated_shadow_dir(), "index.html")
    if not os.path.isfile(path):
        raise HTTPException(404, detail="calibrated business shadow package not generated")
    with open(path, "r", encoding="utf-8") as handle:
        document = handle.read()
    banner = ('<p style="padding:12px 5vw;background:#e9f5ff;border-bottom:1px solid #b9d6ee">'
              '候选模型使用已验证的置信度阈值 0.65，仅供影子验证和人工验收；尚未发布。</p>')
    return HTMLResponse(document.replace("<main>", banner + "<main>", 1))


@APP.get("/business-shadow-calibrated/{name:path}")
def business_shadow_calibrated_asset(name: str):
    path = _safe_join(_calibrated_shadow_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(404, detail="calibrated business shadow asset not found")
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


@APP.get("/api/report/business-shadow")
def api_report_business_shadow():
    root = _business_shadow_dir()
    report = safe_load_json(os.path.join(root, "shadow_report.json"), {})
    acceptance = validate_acceptance(os.path.join(root, "human_acceptance.csv"))
    release = safe_load_json(os.path.join(ARGS.project_dir, "quality", "model_release_manifest.json"), {})
    calibrated = safe_load_json(os.path.join(_calibrated_shadow_dir(), "shadow_report.json"), {})
    return {"status": report.get("status", "missing"), "sample_count": report.get("sample_count", 0),
            "acceptance": acceptance, "release_status": release.get("release_status", "blocked"),
            "blockers": release.get("blockers", []), "url": "/business-shadow",
            "calibrated": {"exists": bool(calibrated), "url": "/business-shadow-calibrated",
                           "candidate_confidence": calibrated.get("inference", {}).get("candidate_confidence")}}


@APP.get("/business-acceptance", response_class=HTMLResponse)
def business_acceptance_page():
    return HTMLResponse(_read_web("business_acceptance.html"))


@APP.get("/api/business-acceptance")
def api_business_acceptance():
    path = os.path.join(_business_shadow_dir(), "human_acceptance.csv")
    if not os.path.isfile(path):
        raise HTTPException(404, detail="human acceptance package not generated")
    machine = {}
    journal = os.path.join(_business_shadow_dir(), "gemma_business_review.jsonl")
    if os.path.isfile(journal):
        with open(journal, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if item.get("image") and item.get("status") == "completed":
                        machine[item["image"]] = item
                except (TypeError, ValueError):
                    continue
    return {"rows": read_acceptance(path), "validation": validate_acceptance(path),
            "gemma_pre_review": machine,
            "machine_review_notice": "Gemma suggestions are evidence only. They never complete human acceptance or release approval."}


@APP.put("/api/business-acceptance/{index}")
async def api_business_acceptance_save(index: int, request: Request):
    path = os.path.join(_business_shadow_dir(), "human_acceptance.csv")
    try:
        row = save_acceptance_row(path, index, await request.json())
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return {"row": row, "validation": validate_acceptance(path)}


@APP.get("/clip")
def clip_page():
    """CLIP 工具已归档：不再作为正式控制台页面提供。"""
    return RedirectResponse(url="/", status_code=307)


@APP.get("/report", response_class=HTMLResponse)
def report_page():
    return HTMLResponse(_read_web("report.html"))


@APP.get("/api/report/eval")
def api_report_eval():
    """返回训练对比结果供报告页使用。"""
    import json as _json
    d = ds()
    quality = safe_load_json(os.path.join(d["yolo"], "quality_report.json"), {})
    if quality.get("status") != "passed":
        return {"invalid": True, "quality_status": quality.get("status", "missing")}
    eval_dir = os.path.join(d["yolo"], "eval")
    report_path = os.path.join(eval_dir, "compare_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return {}


@APP.get("/api/report/summary")
def api_report_summary():
    """Small, presentation-oriented summary of generated training artifacts."""
    d = ds()
    meta = safe_load_json(d["yolo_meta"], {})
    stats = meta.get("stats", {}) if isinstance(meta, dict) else {}

    classes = []
    if os.path.isfile(d["yolo_yaml"]):
        try:
            import yaml as _yaml
            with open(d["yolo_yaml"], "r", encoding="utf-8") as handle:
                names = (_yaml.safe_load(handle) or {}).get("names", {})
            if isinstance(names, dict):
                classes = [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
            elif isinstance(names, list):
                classes = [str(value) for value in names]
        except Exception:
            classes = []

    models = []
    for path in api_models_find().get("models", []):
        try:
            models.append({
                "name": os.path.basename(path),
                "path": path,
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
                "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
            })
        except OSError:
            continue

    predict_root = os.path.join(d["yolo"], "predict")
    latest_prediction = safe_load_json(os.path.join(predict_root, "latest.json"), {})
    prediction_dir = latest_prediction.get("dir") if isinstance(latest_prediction, dict) else None
    prediction_images = 0
    if prediction_dir and os.path.isdir(prediction_dir):
        prediction_images = sum(
            1 for name in os.listdir(prediction_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )

    return {
        "dataset": {
            "ready": os.path.isfile(d["yolo_yaml"]),
            "format": meta.get("format", "obb") if isinstance(meta, dict) else "obb",
            "exported_at": meta.get("exported_at") if isinstance(meta, dict) else None,
            "exported_images": meta.get("exported_images", 0) if isinstance(meta, dict) else 0,
            "split_mode": meta.get("split_mode") if isinstance(meta, dict) else None,
            "stats": stats,
            "classes": classes,
        },
        "models": models,
        "latest_prediction": {
            "created_at": latest_prediction.get("created_at") if isinstance(latest_prediction, dict) else None,
            "source": latest_prediction.get("source") if isinstance(latest_prediction, dict) else None,
            "images": prediction_images,
        },
    }


@APP.get("/api/report/compare_images")
def api_report_compare_images():
    import glob as _glob
    d = ds()
    compare_dir = os.path.join(d["yolo"], "eval", "compare_images")
    result = {"images": [], "ready": False,
              "compare_dir": compare_dir, "dir_exists": os.path.isdir(compare_dir)}
    if not os.path.isdir(compare_dir):
        return result
    has_b = False
    has_t = False
    b_info = {}
    t_info = {}
    for tag in ("baseline", "trained"):
        tag_dir = os.path.join(compare_dir, tag)
        if os.path.isdir(tag_dir):
            subs = sorted(os.listdir(tag_dir))
            for sub in subs:
                sub_dir = os.path.join(tag_dir, sub)
                if os.path.isdir(sub_dir):
                    pngs = None
                    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"):
                        pngs = _glob.glob(os.path.join(sub_dir, ext))
                        if pngs: break
                    if pngs:
                        info = {"sub": sub, "count": len(pngs), "sample": pngs[0] if pngs else None}
                        if tag == "baseline":
                            has_b = True; b_info = info
                        else:
                            has_t = True; t_info = info
                        break
    if not (has_b and has_t):
        result["diagnostic"] = f"b: {b_info} t: {t_info}"
        return result
    result["ready"] = True
    for tag, sub in [("baseline", ["run"]), ("trained", ["run"])]:
        if sub:
            run_dir = os.path.join(compare_dir, tag, sub[0])
            if os.path.isdir(run_dir):
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"):
                    found = _glob.glob(os.path.join(run_dir, ext))
                    if found:
                        result[f"{tag}_files"] = [os.path.basename(f) for f in found[:10]]
                        break
    return result


@APP.get("/api/eval/metrics")
def api_eval_metrics(run: str = ""):
    """返回最新的评估/对比指标 + 训练曲线数据。run=参数可选指定训练目录名。"""
    d = ds()
    eval_dir = os.path.join(d["yolo"], "eval")
    quality = api_report_quality()
    result = {"eval": None, "compare": None, "curves": None, "runs": [], "quality": quality}

    # 扫所有训练结果
    import glob as _glob
    train_runs = []

    # 1. 固定路径
    fixed_csv = os.path.join(d["yolo"], "train", "results.csv")
    if os.path.exists(fixed_csv):
        train_runs.append({"key": "fixed", "label": "固定路径 (最新)", "csv": fixed_csv})

    # 2. Legacy runs/obb/train*
    runs_dir = os.path.join(ARGS.project_dir, "runs", "obb")
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir), reverse=True):
            if not name.startswith("train"):
                continue
            csv = os.path.join(runs_dir, name, "results.csv")
            if not os.path.exists(csv):
                continue
            args = os.path.join(runs_dir, name, "args.yaml")
            label = name
            if os.path.exists(args):
                try:
                    import yaml as _yaml
                    with open(args, "r", encoding="utf-8") as f:
                        a = _yaml.safe_load(f)
                    label += f" ({a.get('model', '')})"
                except Exception:
                    pass
            train_runs.append({"key": name, "label": label, "csv": csv})

    result["runs"] = train_runs

    # 3. 读取指定 run 的曲线 (默认第一条)
    selected = train_runs[0] if train_runs else None
    if run:
        for tr in train_runs:
            if tr["key"] == run:
                selected = tr
                break

    if selected and os.path.exists(selected["csv"]):
        try:
            curves = []
            with open(selected["csv"], "r", encoding="utf-8") as f:
                headers = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if headers is None:
                        headers = [h.strip() for h in parts]
                        continue
                    row = {}
                    for i, h in enumerate(headers):
                        if i < len(parts):
                            try:
                                row[h] = float(parts[i].strip())
                            except ValueError:
                                row[h] = parts[i].strip()
                    curves.append(row)
            if curves:
                result["curves"] = {"headers": headers, "data": curves,
                                     "run_key": selected["key"],
                                     "run_label": selected["label"],
                                     "epochs": len(curves)}
        except Exception:
            pass

    # 4. 对比/评估报告
    cmp_path = os.path.join(eval_dir, "compare_report.json")
    if quality.get("status") == "passed" and os.path.exists(cmp_path):
        result["compare"] = safe_load_json(cmp_path, None)
    em_path = os.path.join(eval_dir, "test_metrics.json")
    if os.path.exists(em_path):
        result["eval"] = safe_load_json(em_path, None)

    return result


@APP.get("/api/report/quality")
def api_report_quality(legacy: bool = False):
    """Audit for the active recovery candidate; legacy=true exposes quarantined history."""
    d = ds()
    official = os.path.join(ARGS.project_dir, "quality", "pklot_official_recovery_v2", "quality_report.json")
    report_path = os.path.join(d["yolo"], "quality_report.json") if legacy else official
    if legacy and not os.path.isfile(report_path):
        rejected_export = d["yolo"] + ".quality_report.json"
        report_path = rejected_export if os.path.isfile(rejected_export) else report_path
    report = safe_load_json(report_path, None)
    if not isinstance(report, dict):
        return {"status": "missing", "report_path": report_path,
                "errors": [{"code": "AUDIT_NOT_RUN", "message": "尚未运行数据质量审计"}]}
    report["report_path"] = report_path
    report["artifact_role"] = "quarantined_legacy" if legacy else "official_recovery_candidate"
    report["legacy_report_url"] = "/api/report/quality?legacy=true"
    return report


@APP.get("/api/report/recovery")
def api_report_recovery():
    from console.dataset_quality import build_recovery_plan
    quality = api_report_quality()
    if quality.get("status") == "missing":
        return {"status": "missing", "steps": []}
    result = {"status": "ready", **build_recovery_plan(quality)}
    candidate_dir = os.path.join(ARGS.project_dir, "quality", "pklot_official_recovery_v2")
    candidate_report = safe_load_json(os.path.join(candidate_dir, "quality_report.json"), None)
    candidate_meta = safe_load_json(os.path.join(candidate_dir, "export_meta.json"), {})
    if isinstance(candidate_report, dict):
        result["candidate"] = {
            "exists": True,
            "dir": candidate_dir,
            "quality_status": candidate_report.get("status"),
            "summary": candidate_report.get("summary", {}),
            "errors": candidate_report.get("errors", []),
            "review_queue_images": candidate_report.get("review_queue_images", 0),
            "sanitization": (candidate_meta.get("sanitized_candidate") or {}).get("stats", {}),
            "artifact_role": "official_recovery_candidate",
        }
    else:
        result["candidate"] = {"exists": False, "dir": candidate_dir}
    result["golden_review_url"] = "/golden-review"
    result["golden_status_url"] = "/api/report/golden"
    official_dir = os.path.join(ARGS.project_dir, "quality", "pklot_official_recovery_v2")
    official_report = safe_load_json(os.path.join(official_dir, "quality_report.json"), None)
    official_metrics = safe_load_json(os.path.join(
        ARGS.project_dir, "quality", "official_baseline_eval", "current_best", "metrics.json"
    ), None)
    if isinstance(official_report, dict):
        result["official_recovery"] = {
            "exists": True,
            "dir": official_dir,
            "summary": official_report.get("summary", {}),
            "splits": official_report.get("splits", {}),
            "errors": official_report.get("errors", []),
            "diagnostic_baseline": official_metrics,
            "finetune_config": os.path.join(ARGS.project_dir, "quality", "finetune_clean_v1.json"),
        }
    calibrated_profile = safe_load_json(os.path.join(
        ARGS.project_dir, "quality", "calibrated_candidate_profile_v1.json"
    ), None)
    if isinstance(calibrated_profile, dict):
        result["calibrated_candidate"] = calibrated_profile
    comparison = safe_load_json(os.path.join(
        _recovery_comparison_dir(), "comparison_report.json"
    ), None)
    if isinstance(comparison, dict):
        result["recovery_comparison"] = {
            **comparison,
            "url": "/recovery-comparison",
        }
    return result


@APP.get("/api/report/golden")
def api_report_golden():
    from console.golden_review import validate_golden_package
    package_dir = _golden_review_dir()
    if not os.path.isdir(package_dir):
        return {"status": "missing", "package_dir": package_dir, "review_url": "/golden-review"}
    gemma = safe_load_json(os.path.join(package_dir, "gemma_review_summary.json"), None)
    return {**validate_golden_package(package_dir), "review_url": "/golden-review", "gemma_review": gemma}


@APP.post("/api/train/explain")
async def api_train_explain(request: Request):
    """调用 Gemma 对训练曲线生成中文解读。"""
    data = await request.json()
    run_key = (data or {}).get("run", "")
    d = ds()

    # 扫描所有训练结果 (与 eval/metrics 相同逻辑)
    train_runs = []
    fixed_csv = os.path.join(d["yolo"], "train", "results.csv")
    if os.path.exists(fixed_csv):
        train_runs.append({"key": "fixed", "label": "固定路径 (最新)", "csv": fixed_csv})
    runs_dir = os.path.join(ARGS.project_dir, "runs", "obb")
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir), reverse=True):
            if not name.startswith("train"):
                continue
            csv = os.path.join(runs_dir, name, "results.csv")
            if not os.path.exists(csv):
                continue
            args = os.path.join(runs_dir, name, "args.yaml")
            label = name
            if os.path.exists(args):
                try:
                    import yaml as _yaml
                    with open(args, "r", encoding="utf-8") as f:
                        a = _yaml.safe_load(f)
                    label += f" ({a.get('model', '')})"
                except Exception:
                    pass
            train_runs.append({"key": name, "label": label, "csv": csv})

    selected = train_runs[0] if train_runs else None
    if run_key:
        for tr in train_runs:
            if tr["key"] == run_key:
                selected = tr
                break

    if not selected:
        return {"ok": False, "error": "没有训练数据"}

    csv_path = selected["csv"]

    # 读取曲线数据
    curves = []
    headers = None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if headers is None:
                    headers = [h.strip() for h in parts]
                    continue
                row = {}
                for i, h in enumerate(headers):
                    if i < len(parts):
                        try:
                            row[h] = float(parts[i].strip())
                        except ValueError:
                            row[h] = parts[i].strip()
                curves.append(row)
    except Exception as e:
        return {"ok": False, "error": f"读取 results.csv 失败: {e}"}

    if not curves:
        return {"ok": False, "error": "训练曲线数据为空"}

    # 提取关键指标的首末值
    total_epochs = len(curves)
    first = curves[0]
    last = curves[-1]
    key_metrics = [
        ('train/box_loss', '训练框损失', '越低越好'),
        ('train/cls_loss', '训练分类损失', '越低越好'),
        ('val/box_loss', '验证框损失', '越低越好'),
        ('val/cls_loss', '验证分类损失', '越低越好'),
        ('metrics/precision(B)', '精确率', '越高越好'),
        ('metrics/recall(B)', '召回率', '越高越好'),
        ('metrics/mAP50(B)', 'mAP@0.5', '越高越好'),
        ('metrics/mAP50-95(B)', 'mAP@0.5-0.95', '越高越好'),
    ]
    summary_lines = []
    for key, label, _ in key_metrics:
        v0 = first.get(key, None)
        vN = last.get(key, None)
        if v0 is not None and vN is not None:
            trend = "↑" if vN > v0 else "↓" if vN < v0 else "→"
            summary_lines.append(f"  {label}({key}): {v0:.4f} → {vN:.4f} {trend}")
    summary = "\n".join(summary_lines)

    # 检查是否有过拟合迹象
    val_loss_keys = ['val/box_loss', 'val/cls_loss']
    overfit_signs = []
    for key in val_loss_keys:
        if key in headers:
            mid_epoch = curves[total_epochs // 2].get(key, None)
            last_val = last.get(key, None)
            if mid_epoch is not None and last_val is not None and last_val > mid_epoch * 1.1:
                overfit_signs.append(f"{key} 在后期上升（可能过拟合）")

    # 从配置和 data.yaml 读取项目描述，避免写死
    concept = cfg_get(cfg(), "gemma.target_concept", "object detection")
    class_names = []
    train_count = val_count = "?"
    try:
        import yaml as _y
        yolo_yaml = d["yolo_yaml"]
        if os.path.exists(yolo_yaml):
            with open(yolo_yaml, "r", encoding="utf-8") as yf:
                yd = _y.safe_load(yf) or {}
            names = yd.get("names", {})
            if isinstance(names, dict):
                class_names = list(names.values()) or class_names
            if isinstance(names, list):
                class_names = names
        train_dir = os.path.join(d["yolo_labels"], "train")
        val_dir = os.path.join(d["yolo_labels"], "val")
        if os.path.isdir(train_dir):
            train_count = len([f for f in os.listdir(train_dir) if f.endswith(".txt")]) or "?"
        if os.path.isdir(val_dir):
            val_count = len([f for f in os.listdir(val_dir) if f.endswith(".txt")]) or "?"
    except Exception:
        pass
    classes_str = ", ".join(class_names[:8]) or "parking spaces, vehicles"

    prompt = f"""你是一个深度学习训练顾问。以下是 YOLO-OBB 模型在{concept}检测任务上的训练曲线摘要：

训练配置：{selected['label']}，{total_epochs} 个 epoch
任务：检测{concept}（{classes_str}等）
训练集: {train_count} 张，验证集: {val_count} 张

各指标首→末变化：
{summary}

{'⚠️ 过拟合预警: ' + '; '.join(overfit_signs) if overfit_signs else '✓ 无明显过拟合迹象'}

请用中文给出 3-5 句话的简洁解读：
1. 模型训练效果总体评价（学得如何）
2. 是否存在过拟合/欠拟合
3. 最值得关注的指标变化
4. 下一步优化建议（如需要更多数据、调整学习率等）

直接回复文字，不要 markdown 格式。"""

    # 调用 Gemma
    gcfg = cfg().get("gemma", {})
    if not gcfg.get("enabled"):
        return {"ok": False, "error": "Gemma 未启用"}

    url = gcfg.get("url")
    model = gcfg.get("model")
    api_key = gcfg.get("api_key") or ""
    timeout = int(cfg_get(cfg(), "gemma.timeout", 120))

    import urllib.request as _ur
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    headers_dict = {"Content-Type": "application/json"}
    if api_key:
        headers_dict["Authorization"] = f"Bearer {api_key}"

    try:
        req = _ur.Request(url, data=json.dumps(body).encode("utf-8"),
                          headers=headers_dict, method="POST")
        with _ur.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8", errors="ignore"))
        explanation = resp["choices"][0]["message"]["content"]
        return {"ok": True, "explanation": explanation, "epochs": total_epochs,
                "summary": summary}
    except Exception as e:
        return {"ok": False, "error": f"Gemma 调用失败: {e}"}


_HEALTH_CACHE = {"time": 0.0, "data": None}
_HEALTH_LOCK = threading.Lock()


def _probe_url(url, timeout=3):
    import urllib.error as _ue
    import urllib.request as _ur
    try:
        req = _ur.Request(url, headers={"User-Agent": "SX-AI-Console/1.0"})
        with _ur.urlopen(req, timeout=timeout) as response:
            return response.status < 500
    except _ue.HTTPError as exc:
        return exc.code in (400, 401, 403, 404, 405, 422)
    except Exception:
        return False


@APP.get("/api/health")
def api_health():
    """快速健康检查：SAM3、Gemma 服务可达性。"""
    now = time.monotonic()
    with _HEALTH_LOCK:
        if _HEALTH_CACHE["data"] is not None and now - _HEALTH_CACHE["time"] < 10:
            return dict(_HEALTH_CACHE["data"])
    health = {"sam3": False, "gemma": False, "gemma_enabled": False}

    # SAM3
    sam3_url = cfg_get(cfg(), "sam3.api_url", "")
    if sam3_url:
        health["sam3"] = _probe_url(sam3_url)

    # Gemma
    health["gemma_enabled"] = bool(cfg_get(cfg(), "gemma.enabled", False))
    if health["gemma_enabled"]:
        gemma_url = cfg_get(cfg(), "gemma.url", "")
        if gemma_url:
            base = gemma_url.split("/v1/", 1)[0].rstrip("/")
            health["gemma"] = _probe_url(base + "/v1/models") or _probe_url(base)
    with _HEALTH_LOCK:
        _HEALTH_CACHE.update(time=now, data=dict(health))
    return health


@APP.get("/static/{name:path}")
def static_file(name: str):
    path = _safe_join(os.path.join(ARGS.app_dir, "web"), name)
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


# ----------------------------------------------------------------------------
# API：run（prompt）切换
# ----------------------------------------------------------------------------

@APP.get("/api/runs")
def api_runs():
    return {"runs": list_runs(), "current": current_prompt()}


@APP.post("/api/runs/select")
async def api_runs_select(request: Request):
    """切换当前打标的 prompt（run）。前端传 {prompt: ...}。"""
    global CURRENT_PROMPT
    with LOCK:
        if any(job.status in ("pending", "running") for job in JOBS.values()):
            return JSONResponse({"ok": False, "error": "有任务运行时不能切换任务空间"}, status_code=409)
    with _GAUTO_LOCK:
        if _GAUTO["running"]:
            return JSONResponse({"ok": False, "error": "请先停止 Gemma 自动审核"}, status_code=409)
    with _BAKE_LOCK:
        if _BAKE["running"]:
            return JSONResponse({"ok": False, "error": "请先停止审核缓存任务"}, status_code=409)
    data = await request.json()
    p = data.get("prompt")
    if not p:
        return JSONResponse({"ok": False, "error": "missing prompt"})
    CURRENT_PROMPT = p
    try:
        with open(_prompt_persist_path(), "w", encoding="utf-8") as f:
            f.write(p)
    except Exception:
        pass
    # 切 run 后清掉索引缓存，让下次读取按新路径重载（StateStore 按路径 key 自动重载）
    with _INDEX_LOCK:
        _INDEX_CACHE.update(mtime=None, list=[], by_overlay={}, by_image={}, by_tid={})
    return {"ok": True, "current": current_prompt()}


# ----------------------------------------------------------------------------
# API：项目概览
# ----------------------------------------------------------------------------

@APP.get("/api/debug")
def api_debug():
    """诊断端点：返回当前路径、索引状态、prompt 模式等。"""
    d = ds()
    cur = current_prompt()
    cfg_dict = _cfg_with_prompt(cur)
    index_path = d.get("index", "")
    sam3_dir = d.get("merged") or d.get("sam3", "")
    merged_exists = os.path.isdir(sam3_dir) if sam3_dir else False
    merged_count = 0
    if merged_exists:
        try:
            merged_count = len([f for f in os.listdir(sam3_dir) if f.endswith(".json")])
        except Exception:
            pass
    index_exists = os.path.exists(index_path)
    index_size = os.path.getsize(index_path) if index_exists else 0
    # 统计 merged JSON 里有多少 target
    target_count = 0
    if merged_exists:
        try:
            for jf in sorted(os.listdir(sam3_dir))[:200]:
                if not jf.endswith(".json"):
                    continue
                data = safe_load_json(os.path.join(sam3_dir, jf), None)
                if data:
                    target_count += len(data.get("targets", []))
        except Exception:
            pass
    return {
        "current_prompt": cur,
        "prompt_mode": cfg_get(cfg_dict, "sam3.prompt_mode", "joined"),
        "batch_slug": d.get("batch_slug", ""),
        "run_base": d["base"],
        "sam3_dir": sam3_dir,
        "merged_exists": merged_exists,
        "merged_json_count": merged_count,
        "sample_target_count": target_count,
        "index_path": index_path,
        "index_exists": index_exists,
        "index_size": index_size,
        "image_dir": image_dir(ARGS.project_dir, cfg_dict),
    }


@APP.get("/api/overview")
def api_overview():
    try:
        return _api_overview_inner()
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[:300]}, status_code=500)


@APP.get("/api/files")
def api_files():
    """返回当前任务空间的有效文件与目录，供文件页和报告页使用。"""
    items = []
    paths = dict(ds())
    paths.setdefault("project", ARGS.project_dir)
    paths.setdefault("config", ARGS.config_path)
    paths.setdefault("images", image_dir(ARGS.project_dir, cfg()))
    for key, value in paths.items():
        if not isinstance(value, (str, os.PathLike)):
            continue
        path = os.path.abspath(os.fspath(value))
        items.append({"key": key, "path": path, "exists": os.path.exists(path)})
    return {"items": items}


# overview 缓存 + sem 值缓存：重计算5秒内复用
_OVERVIEW_CACHE = {"time": 0, "data": {}}
_OVERVIEW_LOCK = threading.Lock()
_SEM_CACHE = {"time": 0, "object_type": [], "issue_type": []}
_FILTER_CACHE = {}
_FILTER_CACHE_LOCK = threading.Lock()


def _api_overview_inner():
    global _OVERVIEW_CACHE
    now = time.time()
    with _OVERVIEW_LOCK:
        if now - _OVERVIEW_CACHE["time"] < 5 and _OVERVIEW_CACHE["data"]:
            return dict(_OVERVIEW_CACHE["data"])
    d = ds()
    cur = current_prompt()
    index = get_index()
    target_count = len(index)
    state = STATE.snapshot(d)
    labeled_records = [rec for rec in state.values() if rec.get("label")]
    label_counts = {}
    for rec in labeled_records:
        lab = rec.get("label")
        if lab:
            label_counts[lab] = label_counts.get(lab, 0) + 1
    min_conf = float(cfg_get(cfg(), "gemma.min_confidence", 0.0))
    min_hits = int(cfg_get(cfg(), "gemma.min_prompt_hits", 1))
    total = len(index)
    gemma_eligible = sum(1 for x in index
                          if (x.get("confidence") is not None and float(x.get("confidence", 0)) >= (min_conf - 1e-9))
                          and (int(x.get("prompt_hits") or 1) >= min_hits))
    gemma_reviewed = sum(1 for rec in labeled_records if rec.get("source") == "gemma" or rec.get("is_auto"))
    human_reviewed = len(labeled_records) - gemma_reviewed
    result = {
        "project_dir": ARGS.project_dir,
        "config_path": ARGS.config_path,
        "image_dir": image_dir(ARGS.project_dir, cfg()),
        "run_base": d["base"],
        "text_prompt": cur,
        "prompt_mode": d.get("prompt_mode", cfg_get(cfg(), "sam3.prompt_mode", "joined")),
        "target_count": target_count,
        "reviewed_count": len(labeled_records),
        "gemma_eligible": gemma_eligible,
        "gemma_reviewed": gemma_reviewed,
        "human_reviewed": human_reviewed,
        "filtered_out": total - gemma_eligible,
        "min_confidence": min_conf,
        "min_prompt_hits": min_hits,
        "total_index": total,
        "label_counts": label_counts,
        "yolo_ready": os.path.exists(d["yolo_yaml"]),
        "index_path": d["index"],
        "index_exists": os.path.exists(d["index"]),
        "index_size": os.path.getsize(d["index"]) if os.path.exists(d["index"]) else 0,
        "train_defaults": {
            "model": cfg_get(cfg(), "train.model", "yolo26n-obb.pt"),
            "imgsz": cfg_get(cfg(), "train.imgsz", 1024),
            "epochs": cfg_get(cfg(), "train.epochs", 100),
        },
    }
    with _OVERVIEW_LOCK:
        _OVERVIEW_CACHE = {"time": now, "data": result}
    return result




def filtered_targets(mode="all", sem_filters=None):
    """按模式 + 语义筛选返回 target 列表，保留 _pos 字段。
    mode: all / unreviewed / accept / reject / hard_positive / hard_negative / bad_mask / needs_review / empty_ok
    sem_filters: {sem_object_type, sem_issue_type, sem_verifiable, min_prompt_hits, min_confidence}
    """
    index = get_index()
    if mode == "all" and not sem_filters:
        return index
    state = STATE.snapshot(ds())
    cache_key = (
        ds()["state"], _index_mtime(), STATE.version, mode,
        json.dumps(sem_filters or {}, sort_keys=True, ensure_ascii=False),
    )
    with _FILTER_CACHE_LOCK:
        cached = _FILTER_CACHE.get(cache_key)
        if cached is not None:
            return cached
    result = []
    for x in index:
        tid = x["target_id"]
        rec = state.get(tid, {})
        lab = rec.get("label", "")
        # mode 筛选
        if mode == "unreviewed":
            if lab:
                continue
        elif mode and mode != "all":
            if lab != mode:
                continue
        # semantic 筛选
        if sem_filters:
            sem = rec.get("semantic", {})
            st = sem_filters.get("sem_object_type")
            if st and str(sem.get("object_type", "")) != str(st):
                continue
            si = sem_filters.get("sem_issue_type")
            if si and str(sem.get("issue_type", "")) != str(si):
                continue
            sv = sem_filters.get("sem_verifiable")
            if sv is not None and bool(sem.get("verifiable")) != bool(sv):
                continue
            mph = sem_filters.get("min_prompt_hits")
            if mph and int(x.get("prompt_hits") or 0) < int(mph):
                continue
            mc = sem_filters.get("min_confidence")
            if mc and float(x.get("confidence") or 0) < float(mc):
                continue
        result.append(x)
    with _FILTER_CACHE_LOCK:
        if len(_FILTER_CACHE) >= 12:
            _FILTER_CACHE.clear()
        _FILTER_CACHE[cache_key] = result
    return result


def _collect_sem_values(state, key):
    now = time.time()
    with _BAKE_LOCK:
        if now - _SEM_CACHE["time"] < 30 and _SEM_CACHE.get(key):
            return _SEM_CACHE[key]
    vals = set()
    for rec in state.values():
        sem = rec.get("semantic", {})
        v = sem.get(key)
        if v:
            vals.add(str(v))
    result = sorted(vals)
    with _BAKE_LOCK:
        _SEM_CACHE["time"] = now
        _SEM_CACHE[key] = result
    return result


@APP.get("/api/review")
def api_review(mode: str = "all", i: int = 0,
               sem_object_type: str = "", sem_issue_type: str = "",
               sem_verifiable: str = "", min_prompt_hits: int = 0,
               min_confidence: float = 0.0):
    try:
        sem_filters = {}
        if sem_object_type:
            sem_filters["sem_object_type"] = sem_object_type
        if sem_issue_type:
            sem_filters["sem_issue_type"] = sem_issue_type
        if sem_verifiable:
            sem_filters["sem_verifiable"] = sem_verifiable.lower() in ("true", "1", "yes")
        if min_prompt_hits > 0:
            sem_filters["min_prompt_hits"] = int(min_prompt_hits)
        if min_confidence > 0:
            sem_filters["min_confidence"] = float(min_confidence)
        items = filtered_targets(mode, sem_filters if sem_filters else None)
        if not items:
            return {"empty": True, "total": 0, "labels": LABELS}
        i = max(0, min(i, len(items) - 1))
        item = items[i]
        tid = item["target_id"]
        state = STATE.all(ds())
        rec = state.get(tid, {})
        with _BAKE_LOCK:
            _BAKE["cursor"] = i
        pos_in_items = None
        if mode != "all":
            pos_in_items = {x["target_id"]: j for j, x in enumerate(items)}
        image_targets = []
        for x in index_by_image(item["image"]):
            if mode == "all":
                jump = x.get("_pos")
            else:
                jump = pos_in_items.get(x["target_id"])
            ims = {
                "target_id": x["target_id"], "target_index": x["target_index"],
                "class_name": x.get("class_name"), "confidence": x.get("confidence"),
                "bbox": x.get("bbox"),
                "label": state.get(x["target_id"], {}).get("label", ""),
                "jump": jump,
            }
            if x.get("source_prompts"): ims["source_prompts"] = x["source_prompts"]
            if x.get("prompt_hits") is not None: ims["prompt_hits"] = x["prompt_hits"]
            if x.get("prompt_mode"): ims["prompt_mode"] = x["prompt_mode"]
            image_targets.append(ims)
        sem = rec.get("semantic", {})
        target_info = {
            "target_id": tid, "image": item["image"],
            "target_index": item["target_index"],
            "class_name": item.get("class_name"), "confidence": item.get("confidence"),
            "overlay_file": item["overlay_file"],
            "label": rec.get("label", ""), "note": rec.get("note", ""),
        }
        if item.get("source_prompts"): target_info["source_prompts"] = item["source_prompts"]
        if item.get("prompt_hits") is not None: target_info["prompt_hits"] = item["prompt_hits"]
        if item.get("source_target_ids"): target_info["source_target_ids"] = item["source_target_ids"]
        if item.get("prompt_mode"): target_info["prompt_mode"] = item["prompt_mode"]
        if rec.get("semantic"): target_info["semantic"] = rec["semantic"]
        if rec.get("gemma_raw"): target_info["gemma_raw"] = rec["gemma_raw"]
        return {
            "empty": False, "mode": mode, "i": i, "total": len(items),
            "labels": LABELS, "target": target_info, "image_targets": image_targets,
            "available_sem_object_types": _collect_sem_values(state, "object_type"),
            "available_sem_issue_types": _collect_sem_values(state, "issue_type"),
        }
    except Exception as e:
        import traceback
        return JSONResponse({"empty": True, "total": 0, "labels": LABELS,
                             "error": str(e), "traceback": traceback.format_exc()[:300]}, status_code=500)


@APP.get("/api/review/prefetch")
def api_review_prefetch(mode: str = "all", i: int = 0, radius: int = 3,
                        sem_object_type: str = "", sem_issue_type: str = "",
                        sem_verifiable: str = "", min_prompt_hits: int = 0,
                        min_confidence: float = 0.0):
    """返回当前位置前后 radius 项的 overlay 文件名，供前端预取（触发懒加载渲染+浏览器缓存）。"""
    sem_filters = {}
    if sem_object_type: sem_filters["sem_object_type"] = sem_object_type
    if sem_issue_type: sem_filters["sem_issue_type"] = sem_issue_type
    if sem_verifiable: sem_filters["sem_verifiable"] = sem_verifiable.lower() in ("true", "1", "yes")
    if min_prompt_hits > 0: sem_filters["min_prompt_hits"] = min_prompt_hits
    if min_confidence > 0: sem_filters["min_confidence"] = min_confidence
    items = filtered_targets(mode, sem_filters or None)
    if not items:
        return {"files": []}
    n = len(items)
    i = max(0, min(i, n - 1))
    files = []
    for j in range(max(0, i - radius), min(n, i + radius + 1)):
        if j == i:
            continue
        files.append(items[j]["overlay_file"])
    return {"files": files}


@APP.get("/api/review/random_unreviewed")
def api_review_random_unreviewed(mode: str = "all",
                                  sem_object_type: str = "", sem_issue_type: str = "",
                                  sem_verifiable: str = "", min_prompt_hits: int = 0,
                                  min_confidence: float = 0.0):
    """随机跳到一个未筛选(无 label)的 target。

    返回 {mode, i}：前端据此跳转。优先在当前 mode 列表里定位；若当前 mode 是某个
    具体标签（列表里本就没有未筛选项），则切到 unreviewed 模式返回随机下标。
    没有任何未筛选项时返回 {done:true}。
    """
    import random
    # apply same sem filters
    sem_filters = {}
    if sem_object_type: sem_filters["sem_object_type"] = sem_object_type
    if sem_issue_type: sem_filters["sem_issue_type"] = sem_issue_type
    if sem_verifiable: sem_filters["sem_verifiable"] = sem_verifiable.lower() in ("true", "1", "yes")
    if min_prompt_hits > 0: sem_filters["min_prompt_hits"] = int(min_prompt_hits)
    if min_confidence > 0: sem_filters["min_confidence"] = float(min_confidence)

    index = filtered_targets("all", sem_filters if sem_filters else None)
    state = STATE.all(ds())
    # 全部未筛选 target 的 target_id
    unreviewed_ids = [x["target_id"] for x in index
                      if not state.get(x["target_id"], {}).get("label")]
    if not unreviewed_ids:
        return {"done": True}

    pick = random.choice(unreviewed_ids)

    # 尝试在当前 mode 列表里定位该 target
    items = filtered_targets(mode, sem_filters if sem_filters else None)
    pos = next((j for j, x in enumerate(items) if x["target_id"] == pick), None)
    if pos is not None:
        return {"done": False, "mode": mode, "i": pos}

    # 当前 mode 列表里没有它（比如 mode 是 accept），切到 unreviewed 模式定位
    un_items = filtered_targets("unreviewed", sem_filters if sem_filters else None)
    pos = next((j for j, x in enumerate(un_items) if x["target_id"] == pick), 0)
    return {"done": False, "mode": "unreviewed", "i": pos}


@APP.get("/api/review/random_batch")
def api_review_random_batch(n: int = 50):
    """返回一批随机未筛选目标，供随机模式预缓冲。
    返回 {files:[overlay_file...], ids:[target_id...]}。
    思路：不预测伪随机序列（脆弱），而是预先采样一批未筛选目标，前端精确预热它们；
    随机跳转时大概率命中其中之一，没命中也只是少缓存一张。
    """
    import random
    n = max(1, min(int(n), 200))
    index = get_index()
    state = STATE.all(ds())
    unreviewed = [x for x in index if not state.get(x["target_id"], {}).get("label")]
    if not unreviewed:
        return {"files": [], "ids": []}
    sample = random.sample(unreviewed, min(n, len(unreviewed)))
    return {
        "files": [x["overlay_file"] for x in sample],
        "ids": [x["target_id"] for x in sample],
    }


@APP.get("/api/gemma/suggest")
def api_gemma_suggest(target_id: str):
    item = index_by_tid(target_id)
    if not item:
        return JSONResponse({"ok": False, "error": "target 不存在"})
    try:
        from console.gemma_review import suggest_label
        cfg_dict = _cfg_with_prompt(current_prompt())
        return suggest_label(ARGS.project_dir, cfg_dict, item, LABELS)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@APP.get("/api/gemma/prompt")
def api_gemma_prompt_get():
    """返回当前生效的提示词、源、是否结构化。"""
    from console.gemma_review import get_active_prompt_template, BUILTIN_PROMPT, prompt_file_path, _build_structured_prompt
    cfg_dict = _cfg_with_prompt(current_prompt())
    template, source = get_active_prompt_template(ARGS.project_dir, cfg_dict)
    editable = bool(cfg_get(cfg(), "gemma.editable_prompt", False))
    enabled = bool(cfg_get(cfg(), "gemma.enabled", False))
    structured = bool(cfg_get(cfg(), "gemma.structured_output", False))
    file_path = prompt_file_path(ARGS.project_dir, cfg_dict)
    file_exists = os.path.exists(file_path) if file_path else False
    result = {
        "editable": editable,
        "enabled": enabled,
        "source": source,
        "template": template,
        "builtin": BUILTIN_PROMPT,
        "file_path": file_path,
        "file_exists": file_exists,
        "concept": cfg_get(cfg_dict, "sam3.text_prompt"),
        "labels": LABELS,
        "structured_output": structured,
    }
    if structured:
        concept = (cfg_get(cfg_dict, "gemma.target_concept") or
                   cfg_get(cfg_dict, "sam3.text_prompt") or "the target concept").strip()
        result["structured_prompt"] = _build_structured_prompt(cfg_dict, concept)
        result["object_types"] = cfg_get(cfg_dict, "gemma.object_types", [])
        result["issue_types"] = cfg_get(cfg_dict, "gemma.issue_types", [])
        result["condition_types"] = cfg_get(cfg_dict, "gemma.condition_types", [])
    return result


@APP.post("/api/gemma/prompt")
async def api_gemma_prompt_set(request: Request):
    """保存提示词到当前 run 的 gemma_prompt.txt。需 editable_prompt: true。"""
    if not bool(cfg_get(cfg(), "gemma.editable_prompt", False)):
        return JSONResponse({"ok": False, "error": "editable_prompt 未开启"})
    data = await request.json()
    template = data.get("template", "")
    from console.gemma_review import prompt_file_path
    cfg_dict = _cfg_with_prompt(current_prompt())
    p = prompt_file_path(ARGS.project_dir, cfg_dict)
    ensure_dir(os.path.dirname(p))
    try:
        if template.strip():
            with open(p, "w", encoding="utf-8") as f:
                f.write(template)
            return {"ok": True, "saved": p}
        else:
            # 空内容 = 重置(删覆盖文件,回退到 builtin/config)
            if os.path.exists(p):
                os.remove(p)
            return {"ok": True, "reset": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@APP.get("/api/gemma/preview")
def api_gemma_preview(target_id: str):
    """返回喂给 Gemma 的实际内容预览：渲染后的提示词 + 视图图片URL,不真调 LLM。"""
    item = index_by_tid(target_id)
    if not item:
        return JSONResponse({"ok": False, "error": "target 不存在"})
    try:
        from console.gemma_review import (compose_views, _render_prompt,
                                  get_active_prompt_template)
        import base64 as _b64
        cfg_dict = _cfg_with_prompt(current_prompt())
        padding = float(cfg_get(cfg_dict, "gemma.crop_padding", 0.7))
        views = compose_views(ARGS.project_dir, cfg_dict, item, padding=padding)
        template, source = get_active_prompt_template(ARGS.project_dir, cfg_dict)
        concept = cfg_get(cfg_dict, "sam3.text_prompt") or "the target concept"
        rendered = _render_prompt(template, concept, LABELS)
        # 视图返回 dataURL 直接前端展示
        view_urls = [{"name": n, "url": f"data:image/png;base64,{b}"} for n, b in views]
        return {"ok": True, "prompt": rendered, "source": source, "views": view_urls}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


# ----------------------------------------------------------------------------
# Gemma 自动打标 (后端线程)
# ----------------------------------------------------------------------------

def _gauto_save_label(tid, item, label, raw, prompt_used):
    """以 is_auto=True 构建 record (不写磁盘，由调用方决定)。"""
    from datetime import datetime as _dt
    image = (item or {}).get("image", "")
    target_index = (item or {}).get("target_index", "")
    rec = {
        "label": label,
        "note": f"[gemma_auto] {raw[:80] if raw else ''}",
        "image": image,
        "target_index": target_index,
        "is_auto": True,
        "source": "gemma",
        "updated_at": _dt.now().isoformat(timespec="seconds"),
    }
    if prompt_used:
        rec["prompt_version"] = prompt_used
    return rec


def _gauto_worker_impl():
    from console.gemma_review import suggest_label
    project_dir = ARGS.project_dir
    cfg_dict = _cfg_with_prompt(current_prompt())
    dirs = run_dirs(project_dir, cfg_dict)
    if not cfg_dict.get("gemma", {}).get("enabled"):
        with _GAUTO_LOCK:
            _GAUTO["running"] = False
            _GAUTO["last_error"] = "Gemma 未启用 (config.gemma.enabled=false)"
            _GAUTO["thread"] = None
        return

    # 起点：所有未打标的 target（跳过已打的，所以"恢复"自然就续上）
    index = get_index()
    state = STATE.all(dirs)
    # 粗筛：只送 Gemma 那些 SAM3 有把握的 target，节省时间
    min_conf = float(cfg_get(cfg_dict, "gemma.min_confidence", 0.0))
    min_hits = int(cfg_get(cfg_dict, "gemma.min_prompt_hits", 1))
    pending = [
        x for x in index
        if not state.get(x["target_id"], {}).get("label")
        and (x.get("confidence") or 0) >= min_conf
        and (x.get("prompt_hits") or 1) >= min_hits
    ]
    with _GAUTO_LOCK:
        _GAUTO["total"] = len(pending)
        _GAUTO["done"] = 0
        _GAUTO["skipped"] = 0
        _GAUTO["failed"] = 0

    for item in pending:
        if _GAUTO["stop"]:
            break
        # 实时再查一次状态，防止人工同时在前端打标
        if STATE.get_label(dirs, item["target_id"]):
            continue
        tid = item["target_id"]
        try:
            r = suggest_label(project_dir, cfg_dict, item, LABELS)
        except Exception as e:
            with _GAUTO_LOCK:
                _GAUTO["failed"] += 1
                _GAUTO["last_error"] = str(e)[:200]
            continue
        if not r.get("ok") or not r.get("label"):
            with _GAUTO_LOCK:
                _GAUTO["skipped"] += 1
                _GAUTO["last_error"] = r.get("error", "无法解析标签")[:200]
            continue
        try:
            rec = _gauto_save_label(tid, item, r["label"], r.get("raw", ""), None)
            if r.get("semantic"):
                rec["semantic"] = r["semantic"]
            if r.get("gemma_raw"):
                rec["gemma_raw"] = r["gemma_raw"]
            STATE.set(dirs, tid, rec)
        except Exception as e:
            with _GAUTO_LOCK:
                _GAUTO["failed"] += 1
                _GAUTO["last_error"] = f"打标失败: {e}"[:200]
            continue
        with _GAUTO_LOCK:
            _GAUTO["done"] += 1
            _GAUTO["last_label"] = r["label"]
            _GAUTO["last_target"] = tid

    with _GAUTO_LOCK:
        _GAUTO["running"] = False
        _GAUTO["thread"] = None


def _gauto_worker():
    try:
        _gauto_worker_impl()
    except Exception as exc:
        with _GAUTO_LOCK:
            _GAUTO["last_error"] = str(exc)[:200]
    finally:
        with _GAUTO_LOCK:
            _GAUTO["running"] = False
            _GAUTO["thread"] = None


@APP.post("/api/gemma/auto/start")
def api_gauto_start():
    """启动 Gemma 自动打标(后端线程)。已打标的自动跳过,所以也是"恢复"。"""
    with _GAUTO_LOCK:
        if _GAUTO["running"]:
            return {"ok": True, "already": True}
        if not bool(cfg_get(cfg(), "gemma.enabled", False)):
            return JSONResponse({"ok": False, "error": "Gemma 未启用,请在 config 里把 gemma.enabled 设为 true"})
        _GAUTO["stop"] = False
        _GAUTO["running"] = True
        _GAUTO["last_error"] = ""
        t = threading.Thread(target=_gauto_worker, daemon=True)
        _GAUTO["thread"] = t
        t.start()
    return {"ok": True}


@APP.post("/api/gemma/auto/stop")
def api_gauto_stop():
    """请求停止 Gemma 自动打标(当前这条跑完就停)。"""
    with _GAUTO_LOCK:
        _GAUTO["stop"] = True
    return {"ok": True}


@APP.get("/api/gemma/auto/status")
def api_gauto_status():
    with _GAUTO_LOCK:
        return dict(_GAUTO, thread=None)


@APP.post("/api/gemma/reset")
def api_gemma_reset():
    """清除所有 Gemma 自动标注，保留人工标注。"""
    import json as _json
    d = ds()
    with _GAUTO_LOCK:
        if _GAUTO["running"]:
            return JSONResponse({"ok": False, "error": "请先停止 Gemma 自动审核再重置"}, status_code=409)
    jl = d["jsonl"]
    st = d["state"]
    with STATE.lock:
        kept = 0
        removed = 0
        lines = []
        if os.path.exists(jl):
            with open(jl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = _json.loads(line)
                    except Exception:
                        continue
                    if record.get("source") == "gemma" or record.get("is_auto"):
                        removed += 1
                    else:
                        lines.append(line)
                        kept += 1
            tmp = jl + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, jl)
        state = {}
        for line in lines:
            try:
                record = _json.loads(line)
                tid = record.pop("target_id", None)
                if tid:
                    state[tid] = record
            except Exception:
                continue
        safe_write_json(st, state)
        STATE.loaded_key = None
        STATE.data = {}
        STATE.version += 1
    with _FILTER_CACHE_LOCK:
        _FILTER_CACHE.clear()
    return {"ok": True, "kept": kept, "removed": removed}


# ----------------------------------------------------------------------------
# CLIP 视觉参考匹配 (端口 8004)
# ----------------------------------------------------------------------------

@APP.get("/api/images/browse")
def api_images_browse(offset: int = 0, limit: int = 50):
    """浏览图片目录，支持分页。"""
    from console.core import image_dir as _idir
    import math
    idir = _idir(ARGS.project_dir, cfg())
    files = []
    if os.path.isdir(idir):
        for f in sorted(os.listdir(idir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(idir, f)
                try:
                    st = os.stat(p)
                    files.append({"name": f, "size": st.st_size, "mtime": st.st_mtime})
                except Exception:
                    files.append({"name": f, "size": 0, "mtime": 0})
    total = len(files)
    page = files[offset : offset + limit]
    return {"ok": True, "images": page, "total": total, "offset": offset, "limit": limit}


@APP.get("/api/images/file/{name:path}")
def api_images_file(name: str):
    """返回图片原始文件。"""
    from console.core import image_dir as _idir
    idir = _idir(ARGS.project_dir, cfg())
    p = _safe_join(idir, name)
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@APP.post("/api/clip/register")
async def api_clip_register(request: Request):
    """用图片名 + bbox 注册 CLIP 参考特征。"""
    raise HTTPException(410, detail="CLIP 功能已归档")
    from console.clip_service import register_reference
    data = await request.json()
    image_name = (data.get("image") or "").strip()
    bbox = data.get("bbox")
    if not image_name:
        return JSONResponse({"ok": False, "error": "image 不能为空"})
    if not bbox or len(bbox) != 4:
        return JSONResponse({"ok": False, "error": "bbox 必须是 [x1,y1,x2,y2]"})
    return register_reference(ARGS.project_dir, cfg(), image_name, bbox)


@APP.get("/api/clip/features")
def api_clip_features():
    """列出所有已注册的 CLIP 参考特征。"""
    raise HTTPException(410, detail="CLIP 功能已归档")
    from console.clip_service import list_features as _lf
    return _lf(ARGS.project_dir, cfg())


@APP.delete("/api/clip/feature/{feature_id}")
def api_clip_delete(feature_id: str):
    """删除指定 CLIP 参考特征。"""
    raise HTTPException(410, detail="CLIP 功能已归档")
    from console.clip_service import delete_feature as _df
    return _df(ARGS.project_dir, cfg(), feature_id)


@APP.post("/api/clip/scan")
async def api_clip_scan(request: Request):
    """全量扫描图片（后台任务）。"""
    raise HTTPException(410, detail="CLIP 功能已归档")
    data = await request.json()
    feature_id = (data.get("feature_id") or "").strip()
    if not feature_id:
        return JSONResponse({"ok": False, "error": "feature_id 不能为空"})
    params = {"action": "clip_scan", "feature_id": feature_id}
    jid = start_job("clip_scan", params)
    return {"ok": True, "job_id": jid}


# ----------------------------------------------------------------------------
# Gemma config
# ----------------------------------------------------------------------------

@APP.post("/api/gemma/generate-config")
async def api_generate_config(request: Request):
    """根据用户描述调用 Gemma 生成 SAM3 prompts + review 配置。"""
    from console.gemma_review import generate_config
    data = await request.json()
    domain = (data.get("domain") or "").strip()
    if not domain:
        return JSONResponse({"ok": False, "error": "domain 不能为空"})
    result = generate_config(ARGS.project_dir, cfg(), domain)
    return result


@APP.post("/api/gemma/apply-config")
async def api_apply_config(request: Request):
    """将生成的配置写入 config.yaml。"""
    import yaml as _yaml
    data = await request.json()
    config_updates = data.get("config", {})
    if not isinstance(config_updates, dict) or not config_updates:
        return JSONResponse({"ok": False, "error": "config 不能为空"})
    allowed_sections = {"sam3", "gemma", "yolo_obb"}
    unexpected = sorted(set(config_updates) - allowed_sections)
    if unexpected:
        return JSONResponse({"ok": False, "error": f"不允许修改配置段: {', '.join(unexpected)}"}, status_code=400)
    try:
        STATE.flush(ds())
        existing = {}
        if os.path.exists(ARGS.config_path):
            with open(ARGS.config_path, "r", encoding="utf-8") as f:
                existing = _yaml.safe_load(f) or {}
        # 深度合并: 嵌套 dict 递归合并, 列表直接覆盖
        def _merge(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _merge(base[k], v)
                else:
                    base[k] = v
        _merge(existing, config_updates)
        tmp = ARGS.config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _yaml.dump(existing, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ARGS.config_path)
        global CFG
        CFG = load_config(ARGS.config_path)
        with STATE.lock:
            STATE.loaded_key = None
            STATE.data = {}
            STATE.version += 1
        with _INDEX_LOCK:
            _INDEX_CACHE.update(mtime=None, list=[], by_overlay={}, by_image={}, by_tid={})
        with _FILTER_CACHE_LOCK:
            _FILTER_CACHE.clear()
        return {"ok": True, "message": "config.yaml 已更新并重新加载", "reloaded": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"写入 config.yaml 失败: {e}"})


@APP.post("/api/label")
async def api_label(request: Request):
    from datetime import datetime as _dt
    data = await request.json()
    try:
        tid = data.get("target_id")
        label = data.get("label")
        from console.pipeline2_review import LABELS as _L
        if label not in _L and label not in ("", None):
            return JSONResponse({"ok": False, "error": f"非法标签: {label}"})
        item = index_by_tid(tid) or {}
        if not tid or not item:
            return JSONResponse({"ok": False, "error": "target 不存在"}, status_code=404)
        image = item.get("image", "")
        target_index = item.get("target_index", "")
        if not image and tid and "::" in tid:
            image, _, ti = tid.rpartition("::")
            try:
                target_index = int(ti)
            except Exception:
                target_index = ""
        rec = {
            "label": label or "",
            "note": data.get("note", ""),
            "image": image,
            "target_index": target_index,
            "updated_at": _dt.now().isoformat(timespec="seconds"),
            "is_auto": bool(data.get("is_auto", False)),
            "source": data.get("source", "human"),   # human / gemma / sam3 etc
        }
        if data.get("semantic"):
            rec["semantic"] = data["semantic"]
        if data.get("gemma_raw"):
            rec["gemma_raw"] = data["gemma_raw"]
        if data.get("prompt_version"):
            rec["prompt_version"] = data["prompt_version"]
        STATE.set(ds(), tid, rec)
        return {"ok": True, "record": rec}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@APP.post("/api/export_csv")
def api_export_csv():
    """按需导出打标结果 CSV（不再每次打标都导，避免卡顿）。"""
    try:
        from console.pipeline2_review import export_state_csv
        export_state_csv(ARGS.project_dir, cfg())
        return {"ok": True, "path": ds()["csv"]}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


# ----------------------------------------------------------------------------
# 后台烤缓存（懒加载之外的并行预热）
# ----------------------------------------------------------------------------
# 策略（按你的选择）：
#   - 统一队列：随机预测批 > 前方窗口 > 后方窗口 > 全量铺底
#   - 未筛选优先于已筛选
#   - 每批重算优先级，实时响应光标/方向/随机预取的变化
#   - 进程数读 config: web.bake_workers（默认 4），与实时渲染并行

def _pending_items():
    """所有还没生成掩膜层的 item，连同位置和是否已筛选。"""
    from console.pipeline2_review import target_layer_path
    index = get_index()
    d = ds()
    state = STATE.all(d)
    out = []
    for pos, x in enumerate(index):
        if os.path.exists(target_layer_path(d, x)):
            continue
        reviewed = bool(state.get(x["target_id"], {}).get("label"))
        out.append((pos, reviewed, x))
    return out


def _priority_key(pos, reviewed, tid, snap):
    """优先级排序键，越小越先烤。
    tier 0: 随机模式预测的下一批目标（精确命中）
    tier 1: 当前位置前方窗口内（按方向）
    tier 2: 后方/反方向窗口内
    tier 3: 其余全量铺底
    同 tier 内未筛选优先、离光标近优先。
    """
    cursor = snap["cursor"]
    direction = snap["direction"] or 1
    window = snap["window"]
    random_ids = snap["random_set"]

    if tid in random_ids:
        tier = 0
    else:
        delta = pos - cursor
        fwd = delta * direction          # >0 表示在前进方向上
        if 0 <= fwd <= window:
            tier = 1
        elif -window <= fwd < 0:
            tier = 2
        else:
            tier = 3
    return (tier, 0 if not reviewed else 1, abs(pos - cursor))


def _snapshot_bake():
    with _BAKE_LOCK:
        return {
            "cursor": _BAKE["cursor"],
            "direction": _BAKE["direction"],
            "window": _BAKE["window"],
            "random_set": set(_BAKE["random_ids"] or []),
        }


def _bake_worker_impl():
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    try:
        workers = int(cfg_get(cfg(), "web.bake_workers", 4))
    except Exception:
        workers = 4
    workers = max(1, workers)

    project_dir = ARGS.project_dir
    cfg_dict = _cfg_with_prompt(current_prompt())

    def strip(it):
        return {k: v for k, v in it.items() if k != "_pos"}

    def process_with(executor, use_mp):
        # 每批：重新取待办、按当前光标/方向/随机重排、取最高优先级的一批烤
        batch = max(workers * 8, 64)
        while True:
            if _BAKE["stop"]:
                return
            pend = _pending_items()
            with _BAKE_LOCK:
                _BAKE["total"] = _BAKE["done"] + len(pend)
            if not pend:
                return
            snap = _snapshot_bake()
            pend.sort(key=lambda t: _priority_key(t[0], t[1], t[2]["target_id"], snap))
            chunk = [t[2] for t in pend[:batch]]
            if use_mp:
                args = [(project_dir, cfg_dict, strip(it)) for it in chunk]
                results = executor.map(bake_one_mp, args, chunksize=8)
            else:
                def one(it):
                    try:
                        prepare_target(project_dir, cfg_dict, strip(it), force=False)
                        return True
                    except Exception:
                        return False
                results = executor.map(one, chunk)
            for ok in results:
                with _BAKE_LOCK:
                    if ok:
                        _BAKE["done"] += 1
                    else:
                        _BAKE["failed"] += 1

    with _BAKE_LOCK:
        _BAKE["done"] = 0
        _BAKE["failed"] = 0

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            process_with(ex, True)
    except Exception:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            process_with(ex, False)

    with _BAKE_LOCK:
        _BAKE["running"] = False
        _BAKE["thread"] = None


def _bake_worker():
    try:
        _bake_worker_impl()
    finally:
        with _BAKE_LOCK:
            _BAKE["running"] = False
            _BAKE["thread"] = None


@APP.post("/api/bake/start")
def api_bake_start():
    with _BAKE_LOCK:
        if _BAKE["running"]:
            return {"ok": True, "already": True}
        _BAKE["stop"] = False
        _BAKE["running"] = True
        t = threading.Thread(target=_bake_worker, daemon=True)
        _BAKE["thread"] = t
        t.start()
    return {"ok": True}


@APP.post("/api/bake/stop")
def api_bake_stop():
    with _BAKE_LOCK:
        _BAKE["stop"] = True
    return {"ok": True}


@APP.get("/api/bake/status")
def api_bake_status():
    with _BAKE_LOCK:
        return {
            "running": _BAKE["running"],
            "done": _BAKE["done"],
            "total": _BAKE["total"],
            "failed": _BAKE["failed"],
            "cursor": _BAKE["cursor"],
            "direction": _BAKE["direction"],
            "random_n": len(_BAKE["random_ids"] or []),
        }


@APP.post("/api/bake/cursor")
async def api_bake_cursor(request: Request):
    """前端汇报浏览状态，驱动预缓冲优先级。
    body: {i, direction(+1/-1), mode, random_ids[]}
    """
    data = await request.json()
    with _BAKE_LOCK:
        try:
            _BAKE["cursor"] = int(data.get("i", _BAKE["cursor"]))
        except Exception:
            pass
        d = data.get("direction")
        if d in (1, -1, 0):
            _BAKE["direction"] = d
        if data.get("mode"):
            _BAKE["mode"] = data["mode"]
        if isinstance(data.get("random_ids"), list):
            _BAKE["random_ids"] = data["random_ids"][:200]
    return {"ok": True}


# ----------------------------------------------------------------------------
# API：媒体
# ----------------------------------------------------------------------------

@APP.get("/media/image/{name:path}")
def media_image(name: str):
    p = _safe_join(image_dir(ARGS.project_dir, cfg()), name)
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@APP.get("/cache/layer/{name:path}")
def cache_layer(name: str):
    """目标的掩膜彩色层 PNG（懒加载：不存在则现场生成）。name 为 overlay_file。"""
    item = index_by_overlay(name)
    if not item:
        raise HTTPException(404)
    try:
        layer, _ = prepare_target(ARGS.project_dir, _cfg_with_prompt(current_prompt()),
                                  item, force=False)
        return FileResponse(layer)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@APP.get("/eval/image/{name:path}")
def eval_image(name: str):
    """对比用的预测结果图。"""
    d = ds()
    base = os.path.join(d["yolo"], "eval", "compare_images")
    p = _safe_join(base, name.replace('\\', '/'))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@APP.get("/api/models/find")
def api_models_find():
    """自动搜索训练好的模型权重，规范路径置顶。"""
    models = []
    # 0. 规范路径 (训练固定输出)
    yolo_dir = ds()["yolo"]
    canonical = os.path.join(yolo_dir, "train", "weights", "best.pt")
    if os.path.exists(canonical):
        models.append(canonical)
        # 滚动备份
        for i in range(1, 10):
            p = os.path.join(yolo_dir, "train", "weights", f"best.{i}.pt")
            if os.path.exists(p):
                models.append(p)
            else:
                break
    # 1. project/train 及其变体 (train, train2, train-5, ...)
    for base in [yolo_dir, os.path.dirname(yolo_dir)]:
        if os.path.isdir(base):
            for entry in os.scandir(base):
                if entry.is_dir() and entry.name.startswith("train"):
                    pt = os.path.join(entry.path, "weights", "best.pt")
                    if os.path.exists(pt) and pt not in models:
                        models.append(pt)
    # 2. runs/ 目录全量扫 (YOLO 默认输出)
    runs_dir = os.path.join(ARGS.project_dir, "runs")
    if os.path.isdir(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            if "best.pt" in files:
                pt = os.path.join(root, "best.pt")
                if pt not in models:
                    models.append(pt)
            if len(models) > 30:
                break
    # 3. 按修改时间排序，规范路径始终第一
    canonical_set = {canonical} if os.path.exists(canonical) else set()
    backup_models = [m for m in models if m not in canonical_set]
    backup_models.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    models = [m for m in models if m in canonical_set] + backup_models
    return {"models": models[:8]}


@APP.get("/api/models/baselines")
def api_models_baselines():
    """返回项目根目录和 models/ 下的预训练权重，作为 baseline 候选。"""
    import glob as _glob
    baselines = []
    for d in [ARGS.project_dir, os.path.join(ARGS.project_dir, "models")]:
        if os.path.isdir(d):
            for pt in _glob.glob(os.path.join(d, "*.pt")):
                name = os.path.basename(pt)
                # 排除明显是训练产出的（含 train 路径的）
                if "train" not in pt.lower():
                    baselines.append(pt)
    # 按文件名去重，优先 OBB + yolov8
    seen = set()
    deduped = []
    for pt in baselines:
        name = os.path.basename(pt)
        if name not in seen:
            seen.add(name)
            deduped.append(pt)
    deduped.sort(key=lambda p: (
        not os.path.basename(p).lower().endswith("-obb.pt"),  # OBB 优先
        "v8" not in os.path.basename(p).lower(),              # yolov8 优先于 yolo26
        os.path.basename(p)))
    return {"baselines": deduped[:10]}


def _test_result_dir():
    root = os.path.join(ds()["yolo"], "predict")
    latest = safe_load_json(os.path.join(root, "latest.json"), {})
    candidate = latest.get("dir") if isinstance(latest, dict) else None
    if candidate:
        candidate = os.path.abspath(candidate)
        try:
            if os.path.commonpath([os.path.abspath(root), candidate]) == os.path.abspath(root) and os.path.isdir(candidate):
                return candidate
        except ValueError:
            pass
    return os.path.join(root, "run")


@APP.get("/api/test/models")
def api_test_models():
    """Model candidates used by the model-test page."""
    paths = api_models_find().get("models", [])
    canonical = os.path.abspath(os.path.join(ds()["yolo"], "train", "weights", "best.pt"))
    models = []
    for path in paths:
        try:
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
        except OSError:
            size_mb = 0
        absolute = os.path.abspath(path)
        filename = os.path.basename(path)
        run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if absolute == canonical:
            role, display_name = "current", "当前主模型"
        elif filename.startswith("best.") and filename != "best.pt":
            role, display_name = "backup", f"历史版本 {filename.removeprefix('best.').removesuffix('.pt')}"
        else:
            role, display_name = "experiment", run_name or filename
        models.append({
            "name": filename,
            "display_name": display_name,
            "role": role,
            "run_name": run_name,
            "path": absolute,
            "size_mb": size_mb,
            "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        })

    # Pretrained weights are true baselines. Keep them separate from rolling
    # best.N.pt backups, which are historical trained-model versions.
    meta = safe_load_json(ds()["yolo_meta"], {})
    fmt = meta.get("format", "obb") if isinstance(meta, dict) else "obb"
    suffix = "-seg.pt" if fmt == "seg" else "-obb.pt"
    baseline_models = []
    for path in api_models_baselines().get("baselines", []):
        filename = os.path.basename(path)
        if not filename.lower().endswith(suffix):
            continue
        baseline_models.append({
            "name": filename,
            "display_name": f"原始基准 {filename}",
            "role": "reference",
            "run_name": "pretrained_reference",
            "path": os.path.abspath(path),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        })

    latest = next((model for model in models if model["role"] == "current"), None)
    baseline = next((model for model in models if model["role"] == "backup"), None)
    if baseline is None:
        baseline = next((model for model in models if model["role"] == "experiment"), None)
    return {
        "models": models + baseline_models,
        "default_comparison": {"baseline": baseline, "latest": latest},
    }


@APP.get("/api/test/results")
def api_test_results(limit: int = 200):
    result_dir = _test_result_dir()
    files = []
    if os.path.isdir(result_dir):
        files = sorted([
            name for name in os.listdir(result_dir)
            if os.path.isfile(os.path.join(result_dir, name))
            and name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ], key=lambda name: os.path.getmtime(os.path.join(result_dir, name)), reverse=True)
    total = len(files)
    limit = max(1, min(int(limit), 500))
    return {"dir": result_dir, "files": files[:limit], "total": total, "truncated": total > limit}


@APP.get("/api/test/image/{name:path}")
def api_test_image(name: str):
    path = _safe_join(_test_result_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


@APP.get("/cache/geom/{name:path}")
def cache_geom(name: str):
    """目标的几何坐标 JSON（OBB + 轮廓 + bbox + 原始宽高）。"""
    item = index_by_overlay(name)
    if not item:
        raise HTTPException(404)
    try:
        _, geom = prepare_target(ARGS.project_dir, _cfg_with_prompt(current_prompt()),
                                 item, force=False)
        return FileResponse(geom, media_type="application/json")
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ----------------------------------------------------------------------------
# API：任务
# ----------------------------------------------------------------------------

def _extract_job_params(data):
    nested = data.get("params")
    if isinstance(nested, dict):
        return dict(nested)
    return {key: value for key, value in data.items() if key != "action"}

def _validate_job_params(action, params):
    if action in {"p3_export", "pipeline_full"}:
        fmt = str(params.get("fmt", "obb"))
        if fmt not in {"obb", "seg"}:
            raise ValueError("fmt 必须是 obb 或 seg")
    if action == "p3_train":
        if params.get("imgsz") not in (None, ""):
            imgsz = int(params["imgsz"])
            if not 64 <= imgsz <= 8192:
                raise ValueError("imgsz 必须在 64 到 8192 之间")
        if params.get("epochs") not in (None, ""):
            epochs = int(params["epochs"])
            if not 1 <= epochs <= 10000:
                raise ValueError("epochs 必须在 1 到 10000 之间")
    if action in {"p3_predict", "p4_predict_targets"}:
        if not str(params.get("model", "")).strip():
            raise ValueError("模型测试需要 model")
        conf = float(params.get("conf", 0.25))
        if not 0 <= conf <= 1:
            raise ValueError("conf 必须在 0 到 1 之间")
    if action == "p3_predict" and not str(params.get("source", "")).strip():
        raise ValueError("模型测试需要 source")
    if action == "p3_compare":
        baseline = str(params.get("baseline", "")).strip()
        trained = str(params.get("trained", "")).strip()
        if not baseline or not trained:
            raise ValueError("模型对比需要基线模型和候选模型")
        if os.path.abspath(baseline) == os.path.abspath(trained):
            raise ValueError("基线模型和候选模型不能相同")
        if not os.path.isfile(baseline) or not os.path.isfile(trained):
            raise ValueError("对比模型文件不存在")
        if str(params.get("split", "test")) not in {"train", "val", "test"}:
            raise ValueError("split 必须是 train、val 或 test")
    if action.startswith("p1_") or action.startswith("pipeline_full"):
        source = image_dir(ARGS.project_dir, cfg())
        if not os.path.isdir(source):
            raise ValueError(f"图片目录不存在: {source}")

@APP.post("/api/job/start")
async def api_job_start(request: Request):
    data = await request.json()
    action = data.get("action")
    if not action:
        return JSONResponse({"ok": False, "error": "missing action"})
    allowed = {
        "p1_infer", "p1_infer_batch", "p1_coords", "p1_aggregate", "p1_merge", "p1_all",
        "p2_index", "p2_overlay", "p3_export", "p3_audit", "p3_sanitize", "p3_train", "p3_predict",
        "p3_eval", "p3_compare", "p4_predict_targets", "pipeline_full",
        "pipeline_full_both",
    }
    if action not in allowed:
        return JSONResponse({"ok": False, "error": f"unknown action: {action}"}, status_code=400)
    runner_params = _extract_job_params(data)
    try:
        _validate_job_params(action, runner_params)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    with LOCK:
        running = next((j for j in JOBS.values() if j.status in ("pending", "running")), None)
    if running:
        return JSONResponse({
            "ok": False,
            "error": f"已有任务正在运行：{running.action}，请先取消或等待完成",
            "job_id": running.id,
        }, status_code=409)
    if action in {"p1_infer", "p1_infer_batch", "p1_coords", "p1_aggregate", "p1_merge", "p1_all", "p2_index"} or action.startswith("pipeline_full"):
        with _BAKE_LOCK:
            if _BAKE["running"]:
                return JSONResponse({"ok": False, "error": "请先停止审核缓存任务再修改识别或索引数据"}, status_code=409)
    if action == "p3_export":
        with _GAUTO_LOCK:
            if _GAUTO["running"]:
                return JSONResponse({"ok": False, "error": "请先停止 Gemma 自动审核再导出数据集"}, status_code=409)
    jid = start_job(action, runner_params)
    return {"ok": True, "job_id": jid}


@APP.get("/api/job/status/{jid}")
def api_job_status(jid: str, since: int = 0):
    """增量返回日志：since 是前端已收到的行数，只回传新增部分。"""
    with LOCK:
        job = JOBS.get(jid)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"})
    since = max(0, int(since))
    start = job._log_start
    relative = max(0, since - start)
    log_slice = job.log[relative:]
    return {
        "ok": True,
        "id": job.id,
        "action": job.action,
        "status": job.status,
        "progress": job.progress,
        "done": job.done,
        "total": job.total,
        "log_start": start,
        "log_total": job._log_total,
        "log": log_slice,
        "error": job.error,
    }


@APP.post("/api/job/cancel/{jid}")
def api_job_cancel(jid: str):
    job = JOBS.get(jid)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    with LOCK:
        if job.status not in ("pending", "running"):
            return {"ok": True, "already_finished": True, "status": job.status}
        job.status = "cancelled"
        job.error = "用户取消"
    process = getattr(job, "_process", None)
    if process is not None and process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"取消失败: {exc}"}, status_code=500)
    job.add("任务已由用户取消")
    return {"ok": True, "status": job.status}


@APP.get("/api/job/latest")
def api_job_latest():
    with LOCK:
        if not JOBS:
            return {"ok": True, "job_id": None}
        latest = JOBS[next(reversed(JOBS))]
    return {"ok": True, "job_id": latest.id}


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------

@APP.on_event("shutdown")
def _on_shutdown():
    try:
        STATE.flush(ds())
    except Exception:
        pass


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", default=".")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--reload", action="store_true", help="热更新：文件修改后自动重启")
    args = p.parse_args()
    args.project_dir = os.path.abspath(args.project_dir)
    args.app_dir = os.path.dirname(os.path.abspath(__file__))
    args.config_path = args.config if os.path.isabs(args.config) else os.path.join(args.project_dir, args.config)
    # 把关键参数写入环境变量，uvicorn reload 子进程通过 _init_from_env() 恢复
    for k in ("project_dir", "app_dir", "config_path", "host", "port"):
        os.environ[f"SAM3_{k.upper()}"] = str(getattr(args, k))
    ARGS = args
    cfg()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not bool(
            cfg_get(cfg(), "web.allow_remote_without_auth", False)):
        raise SystemExit(
            "拒绝在无鉴权状态下监听远程地址；如已评估内网风险，请显式设置 "
            "web.allow_remote_without_auth: true"
        )
    print("Project:", ARGS.project_dir)
    print("Config :", ARGS.config_path)
    print("Open   :", f"http://{ARGS.host}:{ARGS.port}")
    import logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"plain": {"format": "%(levelname)s: %(message)s"}},
        "handlers": {"plain": {"formatter": "plain", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"}},
        "loggers": {
            "uvicorn": {"handlers": ["plain"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["plain"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["plain"], "level": "INFO", "propagate": False},
        },
    }
    if ARGS.reload:
        print("[reload] 热更新已启用 — 保存 .py / web/ 文件自动重启", flush=True)
        uvicorn.run("app:APP", host=ARGS.host, port=ARGS.port, log_level="info",
                    log_config=log_config, reload=True,
                    reload_dirs=[os.path.join(ARGS.app_dir, "web")])
    else:
        uvicorn.run(APP, host=ARGS.host, port=ARGS.port, log_level="info",
                    log_config=log_config)


if __name__ == "__main__":
    main()
