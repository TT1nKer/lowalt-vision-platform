#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 1 — SAM3 识别链路

用法:
    python -m console.pipeline1_sam3 infer      只跑推理
    python -m console.pipeline1_sam3 coords     只算坐标
    python -m console.pipeline1_sam3 aggregate  只汇总
    python -m console.pipeline1_sam3 all        全跑
    python -m console.pipeline1_sam3 status     查看进度

所有行为由 config.yaml 控制。每个 text_prompt 的结果隔离在独立目录，
改 prompt 不会覆盖旧结果。支持断点续跑（已有结果自动跳过）和 Ctrl+C 优雅停止。

【重要】SAM3 服务的请求字段与返回解析是固定契约，不可更改：
    请求:  POST api_url, files={'img_file':...}, data={'text_prompt','show_box'}
    返回:  {'result': {'render_image': b64, 'targets': [{'bbox','confidence',
                       'class_name','single_mask'(b64)}]}}
"""

import os
import sys
import json
import base64
import shutil
import signal
import threading
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from console.core import (
    load_config, cfg_get, run_dirs, raw_prompt_dirs, image_dir, ensure_dir, log,
    safe_write_json, safe_load_json, emit_progress, prompt_slug, prompt_mode,
    merged_target_id_for,
)
from console.pipeline2_review import mask_binary

def _noext(fname):
    """Strip extension, safe for .png / .jpg / .jpeg."""
    return os.path.splitext(fname)[0]

def _is_image(fname):
    return fname.lower().endswith((".png", ".jpg", ".jpeg"))

# 优雅停止：第一次 Ctrl+C 置位停止派发，第二次强制退出。
STOP = threading.Event()


def _handle_sigint(signum, frame):
    if not STOP.is_set():
        print("\n>>> 收到停止信号，正在优雅退出：停止派发新任务，等当前请求结束...", flush=True)
        print(">>> （再按一次 Ctrl+C 可强制退出）", flush=True)
        STOP.set()
    else:
        print("\n>>> 强制退出", flush=True)
        os._exit(1)


# ============================================================
#  坐标换算（EPSG:4326 Web 墨卡托瓦片，与下载脚本一致）
# ============================================================

def parse_block_filename(fname):
    """从 '..._r{行}_c{列}.png' 解析瓦片行列。"""
    import re
    m = re.search(r"_r(\d+)_c(\d+)\.(png|jpg|jpeg)$", fname, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def pixel_to_lonlat(cfg, br, bc, px, py):
    zoom = cfg_get(cfg, "geo.zoom", 18)
    tile_px = 256
    total_cols = 2 ** (zoom + 1)
    total_rows = 2 ** zoom
    abs_col = bc + px / tile_px
    abs_row = br + py / tile_px
    lon = abs_col / total_cols * 360 - 180
    lat = 90 - abs_row / total_rows * 180
    return lon, lat


# ============================================================
#  步骤1：推理  —— SAM3 调用为固定契约，逐字保留
# ============================================================

def call_sam3(cfg, img_path):
    """调用 SAM3 服务。请求字段与字段名固定，不可改。"""
    return call_sam3_with_prompt(cfg, img_path,
                                 cfg_get(cfg, "sam3.text_prompt", "target object"))


def call_sam3_with_prompt(cfg, img_path, prompt):
    """调用 SAM3 服务，使用显式传入的 prompt。请求契约不变。"""
    last = "unknown"
    retry = max(1, int(cfg_get(cfg, "sam3.retry", 2)))
    for _ in range(retry):
        try:
            with open(img_path, "rb") as f:
                files = {"img_file": f}
                data = {
                    "text_prompt": prompt,
                    "show_box": str(cfg_get(cfg, "sam3.show_box", True)).lower(),
                }
                r = requests.post(
                    cfg_get(cfg, "sam3.api_url"),
                    files=files, data=data,
                    timeout=cfg_get(cfg, "sam3.timeout", 120),
                )
            if r.status_code == 200:
                return r.json()
            last = f"http_{r.status_code}"
        except Exception as e:
            last = str(e)
    return {"error": last}


def infer_one(cfg, dirs, fname):
    if STOP.is_set():
        return "stopped"
    out_json = os.path.join(dirs["sam3"], _noext(fname) + ".json")
    if os.path.exists(out_json):
        return "skip"

    resp = call_sam3(cfg, os.path.join(image_dir_global, fname))
    if "error" in resp:
        return f"fail:{resp['error']}"

    result = resp.get("result", {})

    # 渲染图（可选）
    if cfg_get(cfg, "infer.save_render", True) and result.get("render_image"):
        try:
            ensure_dir(dirs["render"])
            with open(os.path.join(dirs["render"], _noext(fname) + "_render.jpg"), "wb") as f:
                f.write(base64.b64decode(result["render_image"]))
        except Exception:
            pass

    targets = []
    for i, t in enumerate(result.get("targets", [])):
        rec = {
            "bbox": t.get("bbox"),
            "confidence": t.get("confidence"),
            "class_name": t.get("class_name"),
        }
        # 掩码存单独文件，json 只记路径
        if cfg_get(cfg, "infer.save_mask", True) and t.get("single_mask"):
            try:
                ensure_dir(dirs["mask"])
                mask_name = _noext(fname) + f"_t{i}.png"
                with open(os.path.join(dirs["mask"], mask_name), "wb") as f:
                    f.write(base64.b64decode(t["single_mask"]))
                rec["mask_file"] = mask_name
            except Exception:
                pass
        targets.append(rec)

    slim = {
        "source_file": fname,
        "text_prompt": cfg_get(cfg, "sam3.text_prompt"),
        "targets": targets,
    }
    safe_write_json(out_json, slim)
    return "ok"


# image_dir 在 step 内解析一次后塞到模块级，供线程函数复用（避免每次重算）。
image_dir_global = None


def step_infer(cfg, dirs, project_dir):
    global image_dir_global
    image_dir_global = image_dir(project_dir, cfg)
    if not os.path.isdir(image_dir_global):
        log(cfg, f"[推理] 图片目录不存在: {image_dir_global}", project_dir)
        return

    ensure_dir(dirs["sam3"])
    files = sorted(f for f in os.listdir(image_dir_global) if _is_image(f))
    total = len(files)
    log(cfg, f"[推理] prompt='{cfg_get(cfg,'sam3.text_prompt')}' 待处理 {total} 块 -> {dirs['base']}", project_dir)
    emit_progress(0, total, "starting")

    safe_write_json(dirs["meta"], {
        "text_prompt": cfg_get(cfg, "sam3.text_prompt"),
        "block_pixel_size": cfg_get(cfg, "block.pixel_size", 1024),
        "zoom": cfg_get(cfg, "geo.zoom", 18),
        "started": datetime.now().isoformat(),
    })

    ok = skip = fail = stopped = done = 0
    workers = int(cfg_get(cfg, "infer.workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        batch_size = max(workers * 4, 20)
        i = 0
        while i < total:
            if STOP.is_set():
                break
            batch = files[i:i + batch_size]
            i += batch_size
            futs = {ex.submit(infer_one, cfg, dirs, f): f for f in batch}
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                if r == "ok":
                    ok += 1
                elif r == "skip":
                    skip += 1
                elif r == "stopped":
                    stopped += 1
                else:
                    fail += 1
                    log(cfg, f"  失败 {futs[fut]}: {r}", project_dir)
                if done % 5 == 0 or done == total:
                    emit_progress(done, total, f"ok={ok} skip={skip} fail={fail}")

    emit_progress(done, total, f"ok={ok} skip={skip} fail={fail}")
    if STOP.is_set():
        log(cfg, f"[推理] 已停止 成功:{ok} 跳过:{skip} 失败:{fail}（重跑可续）", project_dir)
    else:
        log(cfg, f"[推理] 完成 成功:{ok} 跳过:{skip} 失败:{fail}", project_dir)


# ============================================================
#  batch 模式：逐 prompt 推理 + 合并
# ============================================================

def infer_one_batch(cfg, img_path, fname, prompt, pslug, raw_dirs):
    """batch 模式下单个 (image, prompt) 的推理。"""
    if STOP.is_set():
        return "stopped"
    out_json = os.path.join(raw_dirs["sam3"], _noext(fname) + ".json")
    if os.path.exists(out_json):
        return "skip"

    resp = call_sam3_with_prompt(cfg, img_path, prompt)
    if "error" in resp:
        return f"fail:{resp['error']}"

    result = resp.get("result", {})

    if cfg_get(cfg, "infer.save_render", True) and result.get("render_image"):
        try:
            ensure_dir(raw_dirs["render"])
            with open(os.path.join(raw_dirs["render"], _noext(fname) + "_render.jpg"), "wb") as f:
                f.write(base64.b64decode(result["render_image"]))
        except Exception:
            pass

    targets = []
    for i, t in enumerate(result.get("targets", [])):
        rec = {
            "bbox": t.get("bbox"),
            "confidence": t.get("confidence"),
            "class_name": t.get("class_name"),
            "source_prompt": prompt,
            "source_prompt_slug": pslug,
            "source_target_id": merged_target_id_for(fname, pslug, i),
        }
        if cfg_get(cfg, "infer.save_mask", True) and t.get("single_mask"):
            try:
                ensure_dir(raw_dirs["mask"])
                mask_name = _noext(fname) + f"_t{i}.png"
                with open(os.path.join(raw_dirs["mask"], mask_name), "wb") as f:
                    f.write(base64.b64decode(t["single_mask"]))
                rec["mask_file"] = mask_name
            except Exception:
                pass
        targets.append(rec)

    slim = {
        "source_file": fname,
        "text_prompt": prompt,
        "prompt_mode": "batch",
        "targets": targets,
    }
    safe_write_json(out_json, slim)
    return "ok"


def step_infer_batch(cfg, dirs, project_dir):
    """batch 模式：逐 prompt 调 SAM3，每条 prompt 独立缓存。
    
    策略：逐图串行，每张图的 prompts 并发提交（受 workers 限制），
    启动前检查缓存跳过，避免无谓创建 future。"""
    global image_dir_global
    image_dir_global = image_dir(project_dir, cfg)
    if not os.path.isdir(image_dir_global):
        log(cfg, f"[推理-batch] 图片目录不存在: {image_dir_global}", project_dir)
        return

    files = sorted(f for f in os.listdir(image_dir_global) if _is_image(f))
    prompts = cfg_get(cfg, "sam3.prompts", []) or []
    if not prompts:
        prompts = [cfg_get(cfg, "sam3.text_prompt", "target object")]

    n_images = len(files)
    n_prompts = len(prompts)
    total = n_images * n_prompts
    log(cfg, f"[推理-batch] {n_images} 图 x {n_prompts} prompts = {total} 任务 -> {dirs['base']}", project_dir)
    emit_progress(0, total, "starting")

    safe_write_json(dirs["meta"], {
        "prompt_mode": "batch",
        "prompts": prompts,
        "block_pixel_size": cfg_get(cfg, "block.pixel_size", 1024),
        "zoom": cfg_get(cfg, "geo.zoom", 18),
        "started": datetime.now().isoformat(),
    })

    ok = skip = fail = stopped = done = 0
    # batch 并发：限制同时打服务器的 prompt 数，默认 4
    batch_workers = int(cfg_get(cfg, "infer.batch_concurrency", 4))
    batch_workers = max(1, min(batch_workers, len(prompts)))
    # 进度节流：至少每 2000 个任务或每 2 秒报一次
    _last_emit = done
    _last_time = datetime.now()

    with ThreadPoolExecutor(max_workers=batch_workers) as ex:
        for fname in files:
            if STOP.is_set():
                stopped = total - done
                break

            img_path = os.path.join(image_dir_global, fname)
            futs = {}
            for prompt in prompts:
                pslug = prompt_slug(prompt)
                raw_dirs = raw_prompt_dirs(project_dir, cfg, prompt)
                ensure_dir(raw_dirs["sam3"])
                out_json = os.path.join(raw_dirs["sam3"], _noext(fname) + ".json")

                if os.path.exists(out_json):
                    skip += 1
                    done += 1
                    # skip 阶段也周期性报进度（大量缓存时不会黑屏）
                    if (done - _last_emit >= 2000 or done == total
                            or (datetime.now() - _last_time).total_seconds() >= 2):
                        emit_progress(done, total, f"ok={ok} skip={skip} fail={fail}")
                        _last_emit = done
                        _last_time = datetime.now()
                    continue

                ensure_dir(raw_dirs["mask"])
                if cfg_get(cfg, "infer.save_render", True):
                    ensure_dir(raw_dirs["render"])
                fut = ex.submit(infer_one_batch, cfg, img_path, fname, prompt, pslug, raw_dirs)
                futs[fut] = (fname, prompt)

            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                if r == "ok":
                    ok += 1
                elif r == "skip":
                    skip += 1
                elif r == "stopped":
                    stopped += 1
                else:
                    fail += 1
                    log(cfg, f"  失败 {futs[fut][0]} / {futs[fut][1]}: {r}", project_dir)
            if (done - _last_emit >= 2000 or done == total
                    or (datetime.now() - _last_time).total_seconds() >= 2):
                emit_progress(done, total, f"ok={ok} skip={skip} fail={fail}")
                _last_emit = done
                _last_time = datetime.now()

    if STOP.is_set():
        log(cfg, f"[推理-batch] 已停止 成功:{ok} 跳过:{skip} 失败:{fail}", project_dir)
    else:
        log(cfg, f"[推理-batch] 完成 成功:{ok} 跳过:{skip} 失败:{fail}", project_dir)


# ============================================================
#  步骤2：算坐标
# ============================================================

def step_coords(cfg, dirs, project_dir):
    sam3_dir = dirs.get("merged") or dirs.get("sam3")
    if not os.path.isdir(sam3_dir):
        log(cfg, "[坐标] 没有识别结果，请先跑 infer", project_dir)
        return
    files = sorted(f for f in os.listdir(sam3_dir) if f.endswith(".json"))
    total = len(files)
    log(cfg, f"[坐标] 处理 {total} 个结果文件", project_dir)
    emit_progress(0, total, "starting")

    count = 0
    ensure_dir(os.path.dirname(dirs["geo_file"]))
    with open(dirs["geo_file"], "w", encoding="utf-8") as out:
        for n, fname in enumerate(files, 1):
            try:
                with open(os.path.join(sam3_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            block = parse_block_filename(data.get("source_file", ""))
            if not block:
                continue
            br, bc = block
            for t in data.get("targets", []):
                bbox = t.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                lon, lat = pixel_to_lonlat(cfg, br, bc, cx, cy)
                lon1, lat1 = pixel_to_lonlat(cfg, br, bc, x1, y1)
                lon2, lat2 = pixel_to_lonlat(cfg, br, bc, x2, y2)
                out.write(json.dumps({
                    "class_name": t.get("class_name"),
                    "confidence": t.get("confidence"),
                    "lon": lon, "lat": lat,
                    "bbox_lonlat": [lon1, lat1, lon2, lat2],
                    "source_block": data["source_file"],
                }, ensure_ascii=False) + "\n")
                count += 1
            if n % 50 == 0 or n == total:
                emit_progress(n, total, f"targets={count}")
    log(cfg, f"[坐标] 写出 {count} 个目标 -> {dirs['geo_file']}", project_dir)


# ============================================================
#  步骤3：汇总
# ============================================================

def step_aggregate(cfg, dirs, project_dir):
    min_conf = cfg_get(cfg, "aggregate.min_confidence", 0.0)
    only_list = cfg_get(cfg, "aggregate.only_classes", []) or []
    only = set(only_list) if only_list else None

    if not os.path.exists(dirs["geo_file"]):
        log(cfg, f"[汇总] 找不到 {dirs['geo_file']}，请先跑 coords", project_dir)
        return

    stats, features = {}, []
    total = kept = 0
    with open(dirs["geo_file"], "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            conf = rec.get("confidence")
            if conf is not None and conf < min_conf:
                continue
            if only and rec.get("class_name") not in only:
                continue
            kept += 1
            cls = rec.get("class_name")
            stats[cls] = stats.get(cls, 0) + 1
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
                "properties": {
                    "class_name": cls,
                    "confidence": conf,
                    "bbox_lonlat": rec.get("bbox_lonlat"),
                },
            })

    safe_write_json(dirs["stats_file"], {
        "total_raw": total, "kept": kept,
        "min_confidence": min_conf, "by_class": stats,
    })
    safe_write_json(dirs["geojson"], {"type": "FeatureCollection", "features": features})

    log(cfg, f"[汇总] 总目标:{total} 保留:{kept}", project_dir)
    for cls, n in sorted(stats.items(), key=lambda x: -x[1]):
        log(cfg, f"    {cls}: {n}", project_dir)
    log(cfg, f"[汇总] -> {dirs['stats_file']} / {dirs['geojson']}", project_dir)


# ============================================================
#  合并（batch 模式）
# ============================================================

def _bbox_iou(a, b):
    """两个 bbox [x1,y1,x2,y2] 的 IoU。"""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    if xa >= xb or ya >= yb:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b) > 0 else 0.0


def _mask_iou(mask_a, mask_b, bbox_a=None, bbox_b=None):
    """两个布尔掩膜的 IoU（自动裁剪到 bbox 区域加速）。形状不同时返回 0。"""
    import cv2, numpy as np
    try:
        if mask_a.shape != mask_b.shape:
            return 0.0
        # 裁剪到 union bbox 大幅减少像素运算
        if bbox_a and bbox_b:
            xa = int(max(0, min(bbox_a[0], bbox_b[0])))
            ya = int(max(0, min(bbox_a[1], bbox_b[1])))
            xb = int(min(mask_a.shape[1], max(bbox_a[2], bbox_b[2])))
            yb = int(min(mask_a.shape[0], max(bbox_a[3], bbox_b[3])))
            if xb > xa and yb > ya:
                mask_a = mask_a[ya:yb, xa:xb]
                mask_b = mask_b[ya:yb, xa:xb]
        ba = mask_a.astype(np.uint8) * 255
        bb = mask_b.astype(np.uint8) * 255
        inter = np.sum((ba > 0) & (bb > 0))
        union = np.sum((ba > 0) | (bb > 0))
        return float(inter) / float(union) if union > 0 else 0.0
    except Exception:
        return 0.0


def step_merge(cfg, dirs, project_dir):
    """batch 模式：合并所有 raw prompt 的 target，去重输出到 merged/。"""
    root = dirs["base"]
    raw_base = os.path.join(root, "raw_prompts")
    if not os.path.isdir(raw_base):
        log(cfg, "[合并] 没有 raw_prompts 目录，请先跑 infer_batch", project_dir)
        return

    mask_th = float(cfg_get(cfg, "merge.mask_iou_threshold", 0.55))
    bbox_th = float(cfg_get(cfg, "merge.bbox_iou_threshold", 0.65))
    representative = cfg_get(cfg, "merge.representative", "confidence") or "confidence"
    min_conf = float(cfg_get(cfg, "merge.min_confidence", 0.0))

    # 收集所有 raw prompt 子目录
    prompt_dirs = sorted(d for d in os.listdir(raw_base) if os.path.isdir(os.path.join(raw_base, d)))
    log(cfg, f"[合并] 找到 {len(prompt_dirs)} 个 prompt 子目录: {prompt_dirs}", project_dir)

    # 按 image 收集所有 target
    all_by_image = {}
    for pdir in prompt_dirs:
        sam3_dir = os.path.join(raw_base, pdir, "sam3_results")
        mask_dir = os.path.join(raw_base, pdir, "mask")
        if not os.path.isdir(sam3_dir):
            continue
        for jf in sorted(os.listdir(sam3_dir)):
            if not jf.endswith(".json"):
                continue
            data = safe_load_json(os.path.join(sam3_dir, jf), None)
            if not data:
                continue
            image = data.get("source_file")
            if not image:
                continue
            for t in data.get("targets", []):
                if t.get("confidence", 0) is not None and t["confidence"] < min_conf:
                    continue
                rec = dict(t)
                rec["_source_prompt_dir"] = pdir
                rec["_mask_dir"] = mask_dir
                all_by_image.setdefault(image, []).append(rec)

    total_images = len(all_by_image) or 0
    total_raw = sum(len(v) for v in all_by_image.values()) or 0
    log(cfg, f"[合并] {total_images} 张图 {total_raw} 个 raw target，开始去重...", project_dir)
    emit_progress(0, total_images, "starting merge")

    ensure_dir(dirs["merged"])
    ensure_dir(dirs["mask"])
    merged_count = 0
    done = 0

    for image, targets in sorted(all_by_image.items()):
        done += 1

        # 跳过已合并的图像（断点续跑）
        merged_json = os.path.join(dirs["merged"], _noext(image) + ".json")
        if os.path.exists(merged_json):
            if done % 100 == 0 or done == total_images:
                emit_progress(done, total_images, "merged_targets=skip")
            continue

        # 纯 bbox IoU 合并，不加载 mask 避免 I/O 瓶颈
        groups = []  # list of [rep_target, rep_bbox]

        for t in targets:
            t_bbox = t.get("bbox")
            matched = False
            for gi, (grp_rep, rep_bbox) in enumerate(groups):
                if _bbox_iou(t_bbox, rep_bbox) >= bbox_th:
                    grp_rep["_targets"].append(t)
                    if t.get("confidence", 0) > grp_rep.get("confidence", 0):
                        t["_targets"] = grp_rep["_targets"]
                        groups[gi] = (t, t_bbox)
                    matched = True
                    break
            if not matched:
                t["_targets"] = [t]
                groups.append((t, t_bbox))

        # 每组选出 representative，拷贝 mask
        merged_targets = []
        global_merge_idx = 0
        for grp_rep, _ in groups:
            all_in_group = grp_rep.get("_targets", [grp_rep])
            global_merge_idx += 1
            if representative == "area":
                def _area(r):
                    b = r.get("bbox")
                    return (b[2] - b[0]) * (b[3] - b[1]) if b and len(b) == 4 else 0
                best = max(all_in_group, key=_area)
            else:
                best = max(all_in_group, key=lambda r: r.get("confidence", 0) or 0)

            # 拷贝 mask 到 merged/mask/（唯一命名避免冲突）
            mask_file = None
            if best.get("mask_file") and best.get("_mask_dir"):
                src_mask = os.path.join(best["_mask_dir"], best["mask_file"])
                if os.path.exists(src_mask):
                    merged_mask_name = _noext(image) + f"_m{global_merge_idx}.png"
                    dst_mask = os.path.join(dirs["mask"], merged_mask_name)
                    try:
                        shutil.copy2(src_mask, dst_mask)
                        mask_file = merged_mask_name
                    except Exception:
                        mask_file = best["mask_file"]

            source_prompts = []
            source_target_ids = []
            for r in all_in_group:
                sp = r.get("source_prompt", "")
                if sp and sp not in source_prompts:
                    source_prompts.append(sp)
                stid = r.get("source_target_id", "")
                if stid and stid not in source_target_ids:
                    source_target_ids.append(stid)

            merged_targets.append({
                "bbox": best.get("bbox"),
                "confidence": best.get("confidence"),
                "class_name": best.get("class_name"),
                "mask_file": mask_file,
                "source_prompts": source_prompts,
                "prompt_hits": len(source_prompts),
                "source_target_ids": source_target_ids,
            })
            merged_count += 1

        slim = {
            "source_file": image,
            "text_prompt": "__BATCH_MERGED__",
            "prompt_mode": "batch",
            "targets": merged_targets,
        }
        safe_write_json(os.path.join(dirs["merged"], _noext(image) + ".json"), slim)

        if done % 10 == 0 or done == total_images:
            emit_progress(done, total_images, f"merged_targets={merged_count}")

    log(cfg, f"[合并] 完成 {total_images} 张图 → {merged_count} 个合并 target", project_dir)


# ============================================================
#  status
# ============================================================

def step_status(cfg, dirs, project_dir):
    idir = image_dir(project_dir, cfg)
    merged = len([f for f in os.listdir(idir) if _is_image(f)]) if os.path.isdir(idir) else 0
    sam3_dir = dirs.get("merged") or dirs.get("sam3")
    inferred = len([f for f in os.listdir(sam3_dir) if f.endswith(".json")]) if os.path.isdir(sam3_dir) else 0
    mode = dirs.get("prompt_mode") or prompt_mode(cfg)
    print(f"当前模式   : {mode}")
    print(f"当前prompt : {cfg_get(cfg,'sam3.text_prompt')}")
    print(f"结果目录   : {dirs['base']}")
    print(f"待处理块   : {merged}")
    print(f"已推理块   : {inferred}")
    print(f"剩余       : {merged - inferred}")


# ============================================================
#  入口
# ============================================================

def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["infer", "infer_batch", "merge", "coords", "aggregate", "all", "status"])
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config)
    cfg = load_config(cfg_path)
    if args.prompt:
        cfg.setdefault("sam3", {})["text_prompt"] = args.prompt
    dirs = run_dirs(project_dir, cfg)
    mode = prompt_mode(cfg)

    log(cfg, f"===== {args.step} | prompt='{cfg_get(cfg,'sam3.text_prompt')}' mode={mode} =====", project_dir)

    if args.step == "infer":
        step_infer(cfg, dirs, project_dir)
    elif args.step == "infer_batch":
        step_infer_batch(cfg, dirs, project_dir)
    elif args.step == "merge":
        step_merge(cfg, dirs, project_dir)
    elif args.step == "coords":
        step_coords(cfg, dirs, project_dir)
    elif args.step == "aggregate":
        step_aggregate(cfg, dirs, project_dir)
    elif args.step == "status":
        step_status(cfg, dirs, project_dir)
    elif args.step == "all":
        if mode == "batch":
            step_infer_batch(cfg, dirs, project_dir)
            if STOP.is_set():
                log(cfg, ">>> 推理被中断，续跑完成后再单独执行 merge / coords / aggregate", project_dir)
            else:
                step_merge(cfg, dirs, project_dir)
                step_coords(cfg, dirs, project_dir)
                step_aggregate(cfg, dirs, project_dir)
        else:
            step_infer(cfg, dirs, project_dir)
            if STOP.is_set():
                log(cfg, ">>> 推理被中断，跳过坐标和汇总。续跑完成后再单独执行 coords / aggregate", project_dir)
            else:
                step_coords(cfg, dirs, project_dir)
                step_aggregate(cfg, dirs, project_dir)


if __name__ == "__main__":
    main()
