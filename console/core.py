#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py — 全链路共用的基础设施。

设计原则：
- 配置缺字段不崩。所有读取都走 cfg_get(path, default)，永不 KeyError。
- 所有路径集中在 run_dirs() 一处定义，三个 pipeline 共用同一份，杜绝重复实现。
- JSON 读写原子化（写临时文件再 os.replace），避免半截文件。
- 日志同时写文件和 stdout，stdout 行首加标记便于子进程解析。
"""

import os
import re
import json
import hashlib
from datetime import datetime

import yaml

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------

# 全部默认值集中在此。load_config 会把用户 config.yaml 深度合并到这上面，
# 因此用户文件即使只写了几行，下游代码取任何键都不会崩。
DEFAULTS = {
    "paths": {
        "merged_dir": "imagery",
        "results_root": "sam3_runs",
        "log_file": "sam3_console.log",
    },
    "sam3": {
        "api_url": "http://127.0.0.1:8000/sam3",
        "text_prompt": "target object",
        "show_box": True,
        "timeout": 120,
        "retry": 2,
        "prompt_mode": "joined",
        "prompts": [],
        "batch_name": "",
    },
    "merge": {
        "mask_iou_threshold": 0.55,
        "bbox_iou_threshold": 0.65,
        "representative": "confidence",
        "min_confidence": 0.0,
    },
    "infer": {
        "workers": 8,
        "save_render": True,
        "save_mask": True,
        "batch_concurrency": 4,
    },
    "geo": {"zoom": 18},
    "block": {"pixel_size": 1024},
    "aggregate": {"min_confidence": 0.0, "only_classes": []},
    "web": {
        "workers": 8,
        "bake_workers": 4,
        "job_timeout_seconds": 1800,
        "train_timeout_seconds": 0,
        "allow_remote_without_auth": False,
    },
    "yolo_obb": {
        "min_confidence": 0.0,
        "min_mask_area": 10,
        "class_names": [],
        "split": {"mode": "stable_hash", "train": 0.8, "val": 0.1, "test": 0.1, "fixed_test_list": ""},
        "class_source": "",
        "require": {},
        "exclude_classes": [],
        "quality": {
            "enforce": True,
            "require_fixed_test": True,
            "require_golden_manifest": True,
            "require_review_provenance": True,
            "min_human_review_fraction": 0.05,
            "golden_manifest": "quality/golden_test_manifest.json",
            "min_test_images": 100,
            "max_object_area_ratio": 0.08,
            "max_oversized_fraction": 0.005,
            "max_aspect_ratio": 8.0,
            "max_extreme_aspect_fraction": 0.01,
            "max_duplicate_fraction": 0.005,
            "max_class_conflict_fraction": 0.002,
            "max_objects_per_image": 120,
            "geometry_exempt_classes": ["parking_area"],
        },
    },
    "evaluation": {
        "require_quality_pass": True,
        "require_class_compatibility": True,
        "min_precision": 0.75,
        "min_recall": 0.75,
        "min_map50_95": 0.60,
        "max_regression": 0.02,
    },
    "train": {"model": "yolo26n-obb.pt", "imgsz": 1024, "epochs": 100, "model_mirrors": []},
    "clip": {
        "api_url": "http://127.0.0.1:8004",
        "similarity_threshold": 0.7,
        "timeout": 120,
    },
    "gemma": {
        "enabled": False,
        "url": "http://127.0.0.1:9999/v1/chat/completions",
        "schema": "openai",
        "model": "/home/admin/models/gemma-4-31b-nvfp4",
        "api_key": "",
        "image_mode": "both",
        "crop_padding": 0.7,
        "mask_style": "outline",
        "timeout": 120,
        "prompt": "",
        "editable_prompt": True,
        "allowed_labels": [],
        "structured_output": False,
        "target_concept": "",
        "object_types": [],
        "issue_types": [],
        "positive_rules": [],
        "negative_rules": [],
        "unverifiable_rules": [],
        "min_confidence": 0.0,
        "min_prompt_hits": 1,
    },
}


def _deep_merge(base, override):
    """把 override 深合并进 base 的副本并返回。dict 递归合并，其它类型直接覆盖。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path="config.yaml"):
    user = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    # save_mask 必须为 True：后续 review 叠加和 YOLO 导出都依赖掩码文件。
    cfg["infer"]["save_mask"] = True
    return cfg


def cfg_get(cfg, dotted, default=None):
    """安全读取嵌套配置，例如 cfg_get(cfg, 'yolo_obb.split.train', 0.8)。"""
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------

def prompt_slug(prompt):
    """把 prompt 变成安全目录名：'building, car' -> 'building_car'。过长则截断加哈希。"""
    s = re.sub(r"[^\w]+", "_", (prompt or "").strip()).strip("_").lower()
    if len(s) > 60:
        h = hashlib.md5((prompt or "").encode("utf-8")).hexdigest()[:6]
        s = s[:50] + "_" + h
    return s or "default"


def batch_slug(cfg):
    """batch 模式的目录名：来自配置或 md5 生成。"""
    name = cfg_get(cfg, "sam3.batch_name", "") or ""
    if name.strip():
        return prompt_slug(name.strip())
    prompts = cfg_get(cfg, "sam3.prompts", []) or []
    joined = "_".join(p.strip() for p in prompts if p and p.strip())
    if not joined:
        joined = cfg_get(cfg, "sam3.text_prompt", "batch")
    h = hashlib.md5(joined.encode("utf-8")).hexdigest()[:6]
    n = len(prompts) or 1
    return f"batch_{n}prompts_{h}"


def prompt_mode(cfg):
    return cfg_get(cfg, "sam3.prompt_mode", "joined") or "joined"


def get_nested(obj, dotted, default=None):
    """读取嵌套字段，如 get_nested(obj, 'semantic.object_type')。"""
    if not obj or not dotted:
        return default
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def project_path(project_dir, *parts):
    return os.path.abspath(os.path.join(project_dir, *parts))


def run_dirs(project_dir, cfg):
    """当前 prompt 对应 run 的所有路径。唯一权威定义，所有模块都从这里取。
    
    joined 模式：sam3_runs/{prompt_slug}/...
    batch 模式：sam3_runs/{batch_slug}/raw_prompts/{prompt_slug}/... + merged/...
    """
    root = project_path(project_dir, cfg_get(cfg, "paths.results_root", "sam3_runs"))
    mode = prompt_mode(cfg)
    
    if mode == "batch":
        bslug = batch_slug(cfg)
        base = os.path.join(root, bslug)
        merged_base = os.path.join(base, "merged")
        return {
            "base": base,
            "batch_slug": bslug,
            "prompt_mode": "batch",
            "merged": os.path.join(merged_base, "sam3_results"),
            "render": os.path.join(merged_base, "render"),
            "mask": os.path.join(merged_base, "mask"),
            "geo_file": os.path.join(merged_base, "targets_geo.jsonl"),
            "stats_file": os.path.join(merged_base, "stats.json"),
            "geojson": os.path.join(merged_base, "targets.geojson"),
            "meta": os.path.join(base, "run_meta.json"),
            "review": os.path.join(merged_base, "review"),
            "cache": os.path.join(merged_base, "review_cache"),
            "index": os.path.join(merged_base, "review_cache", "target_index.jsonl"),
            "index_legacy": os.path.join(merged_base, "review_cache", "target_index.json"),
            "overlay": os.path.join(merged_base, "review_cache", "overlay"),
            "state": os.path.join(merged_base, "review", "target_state.json"),
            "jsonl": os.path.join(merged_base, "review", "target_reviews.jsonl"),
            "csv": os.path.join(merged_base, "review", "target_export.csv"),
            "yolo": os.path.join(merged_base, "yolo_obb"),
            "yolo_images": os.path.join(merged_base, "yolo_obb", "images"),
            "yolo_labels": os.path.join(merged_base, "yolo_obb", "labels"),
            "yolo_yaml": os.path.join(merged_base, "yolo_obb", "data.yaml"),
            "yolo_meta": os.path.join(merged_base, "yolo_obb", "export_meta.json"),
            "clip_results": os.path.join(merged_base, "clip_results"),
            "clip_features": os.path.join(merged_base, "clip_results", "features.json"),
        }
    
    slug = prompt_slug(cfg_get(cfg, "sam3.text_prompt", "target"))
    base = os.path.join(root, slug)
    return {
        "base": base,
        "sam3": os.path.join(base, "sam3_results"),
        "render": os.path.join(base, "render"),
        "mask": os.path.join(base, "mask"),
        "geo_file": os.path.join(base, "targets_geo.jsonl"),
        "stats_file": os.path.join(base, "stats.json"),
        "geojson": os.path.join(base, "targets.geojson"),
        "meta": os.path.join(base, "run_meta.json"),
        "review": os.path.join(base, "review"),
        "cache": os.path.join(base, "review_cache"),
        "index": os.path.join(base, "review_cache", "target_index.jsonl"),
        "index_legacy": os.path.join(base, "review_cache", "target_index.json"),
        "overlay": os.path.join(base, "review_cache", "overlay"),
        "state": os.path.join(base, "review", "target_state.json"),
        "jsonl": os.path.join(base, "review", "target_reviews.jsonl"),
        "csv": os.path.join(base, "review", "target_export.csv"),
        "yolo": os.path.join(base, "yolo_obb"),
        "yolo_images": os.path.join(base, "yolo_obb", "images"),
        "yolo_labels": os.path.join(base, "yolo_obb", "labels"),
        "yolo_yaml": os.path.join(base, "yolo_obb", "data.yaml"),
        "yolo_meta": os.path.join(base, "yolo_obb", "export_meta.json"),
        "clip_results": os.path.join(base, "clip_results"),
        "clip_features": os.path.join(base, "clip_results", "features.json"),
    }


def raw_prompt_dirs(project_dir, cfg, prompt):
    """batch 模式下单个 raw prompt 的目录。"""
    root = project_path(project_dir, cfg_get(cfg, "paths.results_root", "sam3_runs"))
    bslug = batch_slug(cfg)
    pslug = prompt_slug(prompt)
    base = os.path.join(root, bslug, "raw_prompts", pslug)
    return {
        "base": base,
        "sam3": os.path.join(base, "sam3_results"),
        "render": os.path.join(base, "render"),
        "mask": os.path.join(base, "mask"),
    }


def image_dir(project_dir, cfg):
    p = cfg_get(cfg, "paths.merged_dir", "imagery")
    return p if os.path.isabs(p) else project_path(project_dir, p)


# ----------------------------------------------------------------------------
# 文件 IO
# ----------------------------------------------------------------------------

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def safe_load_json(path, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_write_json(path, obj, retries=5, retry_wait=0.4):
    """原子写 JSON。

    Windows 上 os.replace 若目标文件被其它进程占用（控制台/review 页正在读、
    杀软或同步盘正在扫描）会抛 PermissionError(WinError 5)。这里做几次短重试，
    仍失败则回退为直接覆盖写——宁可牺牲一次原子性，也不让整个建索引任务白跑。
    """
    import time
    ensure_dir(os.path.dirname(path))
    tmp = path + "." + str(os.getpid()) + ".tmp"   # 带 pid，避免并发任务撞同一临时文件
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    last = None
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last = e
            time.sleep(retry_wait * (i + 1))   # 递增退避，给占用方时间释放
        except Exception as e:
            last = e
            break

    # 回退：直接覆盖目标文件（非原子，但能落盘）
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return
    except Exception:
        # 连覆盖都失败：保留 .tmp 不删，至少数据没丢，抛出原始错误
        raise last if last else RuntimeError("safe_write_json failed")


def open_jsonl_writer(path):
    """打开一个 JSONL 写入句柄（覆盖模式）。配合 write_jsonl_record 逐行写。

    相比 safe_write_json，这条路径：
      - 不用临时文件 + os.replace，绕开 Windows 替换锁（WinError 5）。
      - 逐行 flush，不在内存里攒整个大列表（92 万条也不爆内存）。
    用 with 管理：
        with open_jsonl_writer(p) as w:
            for rec in ...: write_jsonl_record(w, rec)
    """
    ensure_dir(os.path.dirname(path))
    return open(path, "w", encoding="utf-8")


def write_jsonl_record(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path, default=None):
    """逐行读 JSONL，返回列表。坏行跳过，不让单条脏数据毁掉整次读取。"""
    if not path or not os.path.exists(path):
        return default if default is not None else []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return default if default is not None else []
    return out


# ----------------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------------

def log(cfg, msg, project_dir="."):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file = cfg_get(cfg, "paths.log_file", "sam3_console.log")
    if not os.path.isabs(log_file):
        log_file = project_path(project_dir, log_file)
    try:
        ensure_dir(os.path.dirname(log_file))
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 目标 ID
# ----------------------------------------------------------------------------

def target_id_for(image, idx):
    return f"{image}::{idx}"


def merged_target_id_for(image, prompt_slug, idx):
    return f"{image}::{prompt_slug}::{idx}"


def overlay_filename(target_id):
    return hashlib.md5(target_id.encode("utf-8")).hexdigest() + ".jpg"


# ----------------------------------------------------------------------------
# 进度协议
# ----------------------------------------------------------------------------
# 子进程通过 stdout 打印 "@@PROGRESS {done} {total} {msg}" 这种行，
# 控制台据此显示真实进度，而不是按行数瞎涨。emit_progress 统一格式。

PROGRESS_PREFIX = "@@PROGRESS"


def emit_progress(done, total, msg=""):
    print(f"{PROGRESS_PREFIX} {done} {total} {msg}", flush=True)


def parse_progress(line):
    """从一行 stdout 解析进度。返回 (done, total, msg) 或 None。"""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        parts = line[len(PROGRESS_PREFIX):].strip().split(None, 2)
        done = int(parts[0])
        total = int(parts[1])
        msg = parts[2] if len(parts) > 2 else ""
        return done, total, msg
    except Exception:
        return None
