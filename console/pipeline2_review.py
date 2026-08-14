#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 2 — target 索引 + 人工筛选支撑

    python -m console.pipeline2_review index    构建 target 索引（每个目标一条）
    python -m console.pipeline2_review overlay   生成掩膜叠加预览缓存（--force 重建）
    python -m console.pipeline2_review status    查看索引/已筛选数量

打标本身由控制台 app.py 调用 save_label() 完成，结果写 state.json / jsonl / csv。
"""

import os
import csv
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from console.core import (
    load_config, cfg_get, run_dirs, image_dir, ensure_dir,
    safe_load_json, safe_write_json, target_id_for, overlay_filename,
    emit_progress, open_jsonl_writer, write_jsonl_record, read_jsonl,
)

LABELS = ["accept", "reject", "hard_positive", "hard_negative", "needs_review", "bad_mask", "empty_ok"]
POSITIVE_LABELS = {"accept", "hard_positive"}
NEGATIVE_LABELS = {"hard_negative", "empty_ok"}
SKIP_LABELS = {"reject", "bad_mask", "needs_review"}


# ----------------------------------------------------------------------------
# 索引
# ----------------------------------------------------------------------------

def list_sam_jsons(dirs):
    sam3_dir = dirs.get("merged") or dirs.get("sam3")
    if not sam3_dir or not os.path.isdir(sam3_dir):
        return []
    return sorted(f for f in os.listdir(sam3_dir) if f.endswith(".json"))


def build_index(project_dir, cfg, progress=None):
    dirs = run_dirs(project_dir, cfg)
    files = list_sam_jsons(dirs)
    total = len(files)
    count = 0
    skipped_empty = missing_image = missing_mask = 0
    idir = image_dir(project_dir, cfg)
    sam3_dir = dirs.get("merged") or dirs.get("sam3")
    prompt_mode = dirs.get("prompt_mode", cfg_get(cfg, "sam3.prompt_mode", "joined"))

    ensure_dir(dirs["cache"])
    with open_jsonl_writer(dirs["index"]) as w:
        for n, fname in enumerate(files, 1):
            data = safe_load_json(os.path.join(sam3_dir, fname), None)
            if not data:
                continue
            image = data.get("source_file")
            targets = data.get("targets") or []
            if not image or not targets:
                skipped_empty += 1
                continue
            if not os.path.exists(os.path.join(idir, image)):
                missing_image += 1
                continue
            for idx, t in enumerate(targets):
                tid = t.get("source_target_id") or target_id_for(image, idx)
                mask_file = t.get("mask_file")
                if mask_file and not os.path.exists(os.path.join(dirs["mask"], mask_file)):
                    missing_mask += 1
                rec = {
                    "target_id": tid,
                    "image": image,
                    "target_index": idx,
                    "class_name": t.get("class_name"),
                    "confidence": t.get("confidence"),
                    "bbox": t.get("bbox"),
                    "mask_file": mask_file,
                    "overlay_file": overlay_filename(tid),
                }
                sp = t.get("source_prompts")
                if sp:
                    rec["source_prompts"] = sp
                    rec["prompt_hits"] = t.get("prompt_hits", len(sp))
                stids = t.get("source_target_ids")
                if stids:
                    rec["source_target_ids"] = stids
                file_pmode = data.get("prompt_mode") or prompt_mode
                if file_pmode and file_pmode != "joined":
                    rec["prompt_mode"] = file_pmode
                # 单个 source_prompt（CLIP 等单源场景）
                spt = t.get("source_prompt")
                if spt and not rec.get("source_prompts"):
                    rec["source_prompts"] = [spt]
                    rec["prompt_hits"] = 1
                write_jsonl_record(w, rec)
                count += 1
            if progress and (n % 50 == 0 or n == total):
                progress(n, total, f"targets={count} empty={skipped_empty} no_img={missing_image}")

    try:
        if os.path.exists(dirs["index_legacy"]):
            os.remove(dirs["index_legacy"])
    except Exception:
        pass

    if progress:
        progress(total, total, f"done targets={count}")
    return count


def load_state(project_dir, cfg):
    """读取打标状态：state.json(快照) + jsonl(权威增量日志)回放合并。

    关键：网页打标走内存 StateStore，state.json 每 100 条才落盘；
    jsonl 才是每条都即时追加的权威记录。任何子进程(导出/训练/CSV)读状态
    都必须用本函数，只读 state.json 会看到旧的甚至空的状态。
    """
    dirs = run_dirs(project_dir, cfg)
    state = safe_load_json(dirs["state"], {})
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
                            state[tid] = {k: v for k, v in r.items() if k != "target_id"}
                    except Exception:
                        continue
        except Exception:
            pass
    return state


def load_index(project_dir, cfg):
    """读 target 索引。优先新版 JSONL；若不存在则回退读旧版 .json（兼容老数据）。"""
    dirs = run_dirs(project_dir, cfg)
    if os.path.exists(dirs["index"]):
        return read_jsonl(dirs["index"], [])
    return safe_load_json(dirs["index_legacy"], [])


def count_index(project_dir, cfg):
    """只数 target 数量，不把整个索引读进内存。92 万条时给概览页用，避免每次全量加载。"""
    dirs = run_dirs(project_dir, cfg)
    if os.path.exists(dirs["index"]):
        n = 0
        try:
            with open(dirs["index"], "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
        except Exception:
            return 0
        return n
    legacy = safe_load_json(dirs["index_legacy"], [])
    return len(legacy)




# ----------------------------------------------------------------------------
# 掩膜叠加（纯 OpenCV，瓶颈在 IO/编解码，用 libjpeg-turbo 路径更快）
# ----------------------------------------------------------------------------

def mask_binary(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 2:
        return m > 0
    if m.shape[2] == 4:
        alpha = m[:, :, 3]
        if np.max(alpha) > 0:
            return alpha > 0
        return np.any(m[:, :, :3] > 0, axis=2)
    return np.any(m > 0, axis=2)


# 预览图 JPEG 编码质量。保持高质量（你要放大看细节），不缩小尺寸。
_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 90]

# 叠加颜色（BGR，OpenCV 顺序）
_MASK_BGR = (255, 161, 78)     # 半透明蓝
_MASK_ALPHA = 0.45             # 掩膜不透明度
_AABB_BGR = (70, 70, 255)      # 正框：红
_OBB_BGR = (60, 220, 60)       # 旋转框：绿
_CONTOUR_BGR = (255, 200, 0)   # 轮廓：青蓝

DEFAULT_SHOW = {"mask": True, "aabb": True, "obb": True, "contour": False}


def _show_suffix(show):
    tags = [k[0] for k in ("mask", "aabb", "obb", "contour") if show.get(k)]
    return "".join(tags) or "none"


def overlay_path_for(dirs, item, show):
    stem, ext = os.path.splitext(item["overlay_file"])
    return os.path.join(dirs["overlay"], f"{stem}__{_show_suffix(show)}{ext}")


def _largest_contour(bmask):
    binary = (bmask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# ============================================================
#  新方案：预计算"掩膜彩色层 PNG" + "几何坐标 JSON"
#  - 掩膜层：二值掩膜转成蓝色+alpha 的 PNG，前端当半透明 img 直接叠加（0 逐像素）
#  - 坐标：OBB 4 角点 + 轮廓多边形，存小 JSON，前端按坐标用 canvas/SVG 画
#  二者都与"显示哪些图层"无关，每个目标只算一次，无组合爆炸、无合成 JPEG。
# ============================================================

_LAYER_RGBA = (78, 161, 255, 178)   # 掩膜层颜色 RGBA（前端 PNG，用 RGB 顺序）


def target_layer_path(dirs, item):
    """该目标的掩膜彩色层 PNG 路径。"""
    stem = os.path.splitext(item["overlay_file"])[0]
    return os.path.join(dirs["overlay"], stem + "_layer.png")


def target_geom_path(dirs, item):
    """该目标的几何坐标 JSON 路径（OBB + 轮廓）。"""
    stem = os.path.splitext(item["overlay_file"])[0]
    return os.path.join(dirs["overlay"], stem + "_geom.json")


def _read_target_mask(dirs, item):
    """读该目标掩膜为布尔数组，并返回(布尔图, (h,w))。无掩膜返回(None,None)。"""
    mf = item.get("mask_file")
    if not mf:
        return None, None
    mp = os.path.join(dirs["mask"], mf)
    if not os.path.exists(mp):
        return None, None
    b = mask_binary(mp)
    if b is None:
        return None, None
    return b, b.shape[:2]


def prepare_target(project_dir, cfg, item, force=False):
    """为单个目标生成掩膜层 PNG + 几何 JSON（若不存在）。返回 (layer_path, geom_path)。

    这是新方案的核心，取代旧的"合成整张 overlay JPEG"。
    """
    dirs = run_dirs(project_dir, cfg)
    ensure_dir(dirs["overlay"])
    layer_path = target_layer_path(dirs, item)
    geom_path = target_geom_path(dirs, item)
    if os.path.exists(layer_path) and os.path.exists(geom_path) and not force:
        return layer_path, geom_path

    b, hw = _read_target_mask(dirs, item)

    # 1) 掩膜彩色层 PNG（带 alpha）。无掩膜则生成 1x1 全透明占位，避免前端 404。
    if b is not None:
        h, w = hw
        rgba = np.zeros((h, w, 4), np.uint8)
        r, g, bl, a = _LAYER_RGBA
        rgba[b] = [bl, g, r, a]   # OpenCV BGRA 顺序
        ok, buf = cv2.imencode(".png", rgba)
        if ok:
            _atomic_write(layer_path, buf.tobytes())
    else:
        rgba = np.zeros((1, 1, 4), np.uint8)
        ok, buf = cv2.imencode(".png", rgba)
        if ok:
            _atomic_write(layer_path, buf.tobytes())

    # 2) 几何坐标 JSON：OBB 4 角点 + 轮廓多边形（像素坐标，前端按图缩放换算）
    geom = {"w": hw[1] if hw else None, "h": hw[0] if hw else None,
            "bbox": item.get("bbox"), "obb": None, "contour": None}
    if b is not None:
        c = _largest_contour(b)
        if c is not None and cv2.contourArea(c) >= 1:
            geom["obb"] = cv2.boxPoints(cv2.minAreaRect(c)).astype(float).tolist()
            eps = 0.002 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(approx) > 300:
                approx = approx[:: len(approx) // 300 + 1]
            geom["contour"] = approx.astype(float).tolist()
    _atomic_write(geom_path, json.dumps(geom, ensure_ascii=False).encode("utf-8"))
    return layer_path, geom_path


def _atomic_write(path, data_bytes):
    tmp = path + "." + str(os.getpid()) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data_bytes)
    try:
        os.replace(tmp, path)
    except Exception:
        with open(path, "wb") as f:
            f.write(data_bytes)
        try:
            os.remove(tmp)
        except Exception:
            pass


def draw_overlay(project_dir, cfg, item, force=False, show=None):
    dirs = run_dirs(project_dir, cfg)
    ensure_dir(dirs["overlay"])
    if show is None:
        show = DEFAULT_SHOW
    out_path = overlay_path_for(dirs, item, show)
    if os.path.exists(out_path) and not force:
        return out_path

    src = os.path.join(image_dir(project_dir, cfg), item["image"])
    base = cv2.imread(src, cv2.IMREAD_COLOR)
    if base is None:
        raise RuntimeError(f"无法读取图片: {src}")
    h, w = base.shape[:2]

    bmask = None
    mf = item.get("mask_file")
    if mf:
        mp = os.path.join(dirs["mask"], mf)
        if os.path.exists(mp):
            bmask = mask_binary(mp)
            if bmask is not None and bmask.shape[:2] != (h, w):
                bmask = cv2.resize(bmask.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

    if show.get("mask") and bmask is not None:
        color = np.empty_like(base)
        color[:] = _MASK_BGR
        base[bmask] = (base[bmask].astype(np.float32) * (1 - _MASK_ALPHA)
                       + color[bmask].astype(np.float32) * _MASK_ALPHA).astype(np.uint8)

    if show.get("aabb"):
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(base, (x1, y1), (x2, y2), _AABB_BGR, 3)

    if show.get("obb") and bmask is not None:
        c = _largest_contour(bmask)
        if c is not None and cv2.contourArea(c) >= 1:
            box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
            cv2.polylines(base, [box], True, _OBB_BGR, 3, cv2.LINE_AA)

    if show.get("contour") and bmask is not None:
        c = _largest_contour(bmask)
        if c is not None:
            cv2.drawContours(base, [c], -1, _CONTOUR_BGR, 2, cv2.LINE_AA)

    text = f"#{item.get('target_index')} conf={item.get('confidence')}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(base, (6, 6), (6 + tw + 14, 6 + th + bl + 12), (0, 0, 0), -1)
    cv2.putText(base, text, (13, 6 + th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    tmp = out_path + "." + str(os.getpid()) + ".tmp.jpg"
    ok, buf = cv2.imencode(".jpg", base, _JPEG_PARAMS)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    try:
        os.replace(tmp, out_path)
    except Exception:
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        try:
            os.remove(tmp)
        except Exception:
            pass
    return out_path


def bake_one_mp(args):
    """多进程子任务：为单个目标预计算掩膜层+几何。args=(project_dir, cfg_dict, item[, _])。

    放在本模块（import 无副作用），spawn 方式起子进程时安全。
    """
    if len(args) >= 3:
        project_dir, cfg_dict, item = args[0], args[1], args[2]
    else:
        return False
    try:
        prepare_target(project_dir, cfg_dict, item, force=False)
        return True
    except Exception:
        return False


def build_overlay_cache(project_dir, cfg, force=False, progress=None):
    index = load_index(project_dir, cfg)
    if not index:
        build_index(project_dir, cfg)
        index = load_index(project_dir, cfg)
    workers = int(cfg_get(cfg, "web.workers", os.cpu_count() or 4))
    total = len(index)
    ok = fail = done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(prepare_target, project_dir, cfg, item, force): item["target_id"] for item in index}
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
                ok += 1
            except Exception:
                fail += 1
            if progress and (done % 50 == 0 or done == total):
                progress(done, total, f"ok={ok} fail={fail}")
    if progress:
        progress(total, total, f"done ok={ok} fail={fail}")


# ----------------------------------------------------------------------------
# 打标存储
# ----------------------------------------------------------------------------

def export_state_csv(project_dir, cfg):
    dirs = run_dirs(project_dir, cfg)
    state = load_state(project_dir, cfg)
    index = {x["target_id"]: x for x in load_index(project_dir, cfg)}
    ensure_dir(dirs["review"])
    with open(dirs["csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "target_id", "image", "target_index", "label", "class_name",
            "confidence", "bbox", "mask_file", "note", "updated_at",
            "source_prompts", "prompt_hits", "source_target_ids", "prompt_mode",
            "semantic_object_type", "semantic_issue_type", "semantic_verifiable",
            "semantic_confidence", "semantic_condition",
        ])
        w.writeheader()
        for tid, rec in sorted(state.items()):
            item = index.get(tid, {})
            sem = rec.get("semantic", {})
            w.writerow({
                "target_id": tid,
                "image": item.get("image", rec.get("image", "")),
                "target_index": item.get("target_index", rec.get("target_index", "")),
                "label": rec.get("label", ""),
                "class_name": item.get("class_name", ""),
                "confidence": item.get("confidence", ""),
                "bbox": json.dumps(item.get("bbox", ""), ensure_ascii=False),
                "mask_file": item.get("mask_file", ""),
                "note": rec.get("note", ""),
                "updated_at": rec.get("updated_at", ""),
                "source_prompts": json.dumps(item.get("source_prompts", rec.get("source_prompts", "")), ensure_ascii=False),
                "prompt_hits": str(item.get("prompt_hits", rec.get("prompt_hits", "") or "")),
                "source_target_ids": json.dumps(item.get("source_target_ids", rec.get("source_target_ids", "")), ensure_ascii=False),
                "prompt_mode": item.get("prompt_mode", rec.get("prompt_mode", "")),
                "semantic_object_type": sem.get("object_type", ""),
                "semantic_issue_type": sem.get("issue_type", ""),
                "semantic_verifiable": str(sem.get("verifiable", "") if sem.get("verifiable") is not None else ""),
                "semantic_confidence": str(sem.get("confidence", "") if sem.get("confidence") is not None else ""),
                "semantic_condition": sem.get("condition", ""),
            })


def save_label(project_dir, cfg, target_id, label, note="", item=None):
    """打标。

    性能关键：不再每次全量 load_index 和导出 CSV（30 万条时那会让每次打标卡 10s）。
    - target 的 image/target_index 由调用方通过 item 传入（app 有进程内缓存）；
      传不进来也没关系，state 里本就保存这两个字段。
    - jsonl 用追加（增量，便宜）。
    - CSV 不在这里导出，改为按需（见 export_state_csv / YOLO 导出时）。
    """
    if label not in LABELS:
        raise ValueError(f"非法标签: {label}")
    dirs = run_dirs(project_dir, cfg)

    image = (item or {}).get("image", "")
    target_index = (item or {}).get("target_index", "")
    # 兜底：万一没传 item 又是新 target，才回退查一次（罕见）
    if not image:
        # 从 target_id 解析（格式 "image::idx"），避免全量读索引
        if "::" in target_id:
            image, _, ti = target_id.rpartition("::")
            try:
                target_index = int(ti)
            except Exception:
                target_index = ""

    state = load_state(project_dir, cfg)
    rec = {
        "label": label,
        "note": note,
        "image": image,
        "target_index": target_index,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    state[target_id] = rec
    ensure_dir(dirs["review"])
    safe_write_json(dirs["state"], state)
    with open(dirs["jsonl"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"target_id": target_id, **rec}, ensure_ascii=False) + "\n")
    return rec


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "overlay", "status"])
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    project_dir = os.path.abspath(args.project_dir)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config)
    cfg = load_config(cfg_path)
    if args.prompt:
        cfg.setdefault("sam3", {})["text_prompt"] = args.prompt

    if args.step == "index":
        build_index(project_dir, cfg, progress=emit_progress)
    elif args.step == "overlay":
        build_overlay_cache(project_dir, cfg, force=args.force, progress=emit_progress)
    elif args.step == "status":
        index = load_index(project_dir, cfg)
        state = safe_load_json(run_dirs(project_dir, cfg)["state"], {})
        print("targets:", len(index))
        print("reviewed:", len(state))


if __name__ == "__main__":
    main()
