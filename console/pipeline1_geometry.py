#!/usr/bin/env python3
"""
Pipeline 1.5 — 几何线框过滤（混合策略）
1. 送 bbox 到 8004 geometry_only API，获取 has_lines
2. 混合过滤：有线=保留；无线但邻近有线bbox=保留（同一排车位）；孤立无线=剔除
"""

import os, sys, json, time
import requests

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
from console.core import load_config, run_dirs, image_dir, log, safe_write_json

API_URL = "http://127.0.0.1:8004/api/segment"
MODE = "geometry_only"
IOU_EXPAND_RATIO = 0.5   # bbox 膨胀比例（用来判断相邻车位）


def expand_bbox(bbox, ratio):
    """膨胀 bbox 用于相邻检测。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    dw, dh = w * ratio, h * ratio
    return [x1 - dw, y1 - dh, x2 + dw, y2 + dh]


def bbox_overlap(a, b):
    """两个 bbox 是否重叠。"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def run_geometry_api(cfg, dirs, idir):
    """Step 1: 调用几何 API，写出 target_geometry.jsonl。"""
    index_path = dirs["index"]
    if not os.path.exists(index_path):
        log(cfg, "[几何] 找不到 index", project_dir)
        return False

    out_path = os.path.join(dirs["review"], "target_geometry.jsonl")
    done_path = out_path + ".done_images.txt"

    done_images = set()
    if os.path.exists(done_path):
        with open(done_path, "r") as f:
            done_images = set(line.strip() for line in f)

    by_image = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line.strip())
            by_image.setdefault(t["image"], []).append(t)

    total = len(by_image)
    log(cfg, f"[几何] {total} 张图，调用 {API_URL}", project_dir)

    done = 0
    for image, targets in sorted(by_image.items()):
        done += 1
        if image in done_images:
            if done % 500 == 0:
                print(f"  [{done}/{total}] skipped", flush=True)
            continue

        img_path = os.path.join(idir, image)
        if not os.path.exists(img_path):
            print(f"  [{done}/{total}] MISSING: {image}", flush=True)
            done_images.add(image)
            with open(done_path, "a") as df:
                df.write(image + "\n")
            continue

        bboxes = [t["bbox"] for t in targets]
        try:
            with open(img_path, "rb") as f:
                r = requests.post(API_URL,
                    files={"image": f},
                    data={"bboxes": json.dumps(bboxes), "mode": MODE},
                    timeout=120)
            if r.status_code != 200:
                print(f"  [{done}/{total}] HTTP {r.status_code}: {image}", flush=True)
                continue

            result = r.json()
            geo_map = {gf["bbox_idx"]: gf for gf in result.get("geometry_features", [])}

        except Exception as e:
            print(f"  [{done}/{total}] EXCEPTION: {image} - {e}", flush=True)
            continue

        with open(out_path, "a", encoding="utf-8") as out:
            for i, t in enumerate(targets):
                gf = geo_map.get(i, {})
                rec = {
                    "target_id": t["target_id"],
                    "image": image,
                    "has_lines": bool(gf.get("has_lines", False)),
                    "orientation": gf.get("orientation", ""),
                    "completeness": gf.get("completeness", 0.0),
                    "lines_count": len(gf.get("lines", [])),
                    "corners_count": len(gf.get("corners", [])),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        done_images.add(image)
        with open(done_path, "a") as df:
            df.write(image + "\n")

        if done % 100 == 0:
            print(f"  [{done}/{total}] {done*100//total}%", flush=True)

    log(cfg, f"[几何] 完成 {total} 图 → {out_path}", project_dir)
    return True


def apply_hybrid_filter(cfg, dirs):
    """Step 2: 混合过滤，写出 accept/reject 标签。"""
    index_path = dirs["index"]
    geo_path = os.path.join(dirs["review"], "target_geometry.jsonl")
    jsonl_path = dirs["jsonl"]

    if not os.path.exists(geo_path):
        log(cfg, "[过滤] 没有几何结果，先跑几何 API", project_dir)
        return

    # 加载几何结果
    geo_map = {}
    with open(geo_path, "r") as f:
        for line in f:
            g = json.loads(line.strip())
            geo_map[g["target_id"]] = g

    # 加载 index，按图分组
    by_image = {}
    with open(index_path, "r") as f:
        for line in f:
            t = json.loads(line.strip())
            by_image.setdefault(t["image"], []).append(t)

    total_targets = sum(len(v) for v in by_image.values())
    log(cfg, f"[过滤] {len(by_image)} 张图, {total_targets} targets, 开始混合过滤...", project_dir)

    accept_count = 0
    reject_count = 0
    adj_count = 0  # 通过邻近判定保留的

    with open(jsonl_path, "a", encoding="utf-8") as out:
        for image, targets in sorted(by_image.items()):
            # 收集该图所有 has_lines=True 的 target 的 bbox（膨胀后）
            line_bboxes = []
            for t in targets:
                g = geo_map.get(t["target_id"], {})
                if g.get("has_lines"):
                    line_bboxes.append(expand_bbox(t["bbox"], IOU_EXPAND_RATIO))

            for t in targets:
                g = geo_map.get(t["target_id"], {})
                label = None

                if g.get("has_lines"):
                    label = "accept"
                    accept_count += 1
                elif line_bboxes:
                    # 检查无线 target 的膨胀 bbox 是否与任何有线 bbox 重叠
                    ebbox = expand_bbox(t["bbox"], IOU_EXPAND_RATIO)
                    if any(bbox_overlap(ebbox, lb) for lb in line_bboxes):
                        label = "accept"
                        accept_count += 1
                        adj_count += 1
                    else:
                        label = "reject"
                        reject_count += 1
                else:
                    # 该图没有任何有线 target，全拒
                    label = "reject"
                    reject_count += 1

                rec = {
                    "target_id": t["target_id"],
                    "label": label,
                    "source": "geometry_filter",
                    "ts": "2026-06-30T09:12:00",
                    "is_auto": True,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log(cfg, f"[过滤] accept={accept_count} (adjacent={adj_count}) reject={reject_count}", project_dir)


if __name__ == "__main__":
    cfg = load_config(os.path.join(project_dir, "config.yaml"))
    dirs = run_dirs(project_dir, cfg)
    idir = image_dir(project_dir, cfg)

    # 先检查之前是否有部分几何结果
    geo_path = os.path.join(dirs["review"], "target_geometry.jsonl")
    geom_done = 0
    if os.path.exists(geo_path):
        with open(geo_path) as f:
            geom_done = sum(1 for _ in f)
    total_index = 0
    if os.path.exists(dirs["index"]):
        with open(dirs["index"]) as f:
            total_index = sum(1 for _ in f)
    print(f"[几何] 已有 {geom_done}/{total_index} 条几何结果")

    # 如果几何结果不全，先补跑 API
    if geom_done < total_index:
        ok = run_geometry_api(cfg, dirs, idir)
        if not ok:
            sys.exit(1)

    # Step 2: 混合过滤
    apply_hybrid_filter(cfg, dirs)
    print("Done.")
