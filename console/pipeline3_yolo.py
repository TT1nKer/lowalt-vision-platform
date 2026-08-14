#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 3 — YOLO-OBB 数据导出与训练

    python -m console.pipeline3_yolo export   把人工筛选的正样本掩膜导出为 YOLO-OBB 数据集
    python -m console.pipeline3_yolo train    调用 ultralytics 训练（需已 export）
    python -m console.pipeline3_yolo status

正样本（accept/hard_positive）的掩膜会转成最小外接旋转框写入标签；
负样本（hard_negative/empty_ok）所在图片作为纯背景（空标签）也会被导出，
有助于降低误检。其余标签忽略。
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from console.core import (
    load_config, cfg_get, run_dirs, image_dir, ensure_dir,
    safe_load_json, safe_write_json, emit_progress, get_nested,
)
from console.dataset_quality import audit_yolo_dataset, build_sanitized_candidate, require_quality_pass
from console.golden_set import build_golden_candidate
from console.golden_review import install_golden_package, validate_golden_package


def _yolo_executable():
    candidate = os.path.join(os.path.dirname(sys.executable), "yolo.exe")
    return candidate if os.path.exists(candidate) else "yolo"


def _load_export_dependencies():
    """延迟加载导出依赖，避免 train 路径提前初始化 OpenCV 的 OpenMP runtime。"""
    global cv2, np, Image
    global load_index, load_state, mask_binary, POSITIVE_LABELS, NEGATIVE_LABELS, SKIP_LABELS

    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _Image
    from console.pipeline2_review import (
        load_index as _load_index,
        load_state as _load_state,
        mask_binary as _mask_binary,
        POSITIVE_LABELS as _positive_labels,
        NEGATIVE_LABELS as _negative_labels,
        SKIP_LABELS as _skip_labels,
    )

    cv2 = _cv2
    np = _np
    Image = _Image
    load_index = _load_index
    load_state = _load_state
    mask_binary = _mask_binary
    POSITIVE_LABELS = _positive_labels
    NEGATIVE_LABELS = _negative_labels
    SKIP_LABELS = _skip_labels


# ----------------------------------------------------------------------------
# 数据集划分（按文件名哈希稳定划分，重跑结果一致）
# ----------------------------------------------------------------------------

def stable_ratio(name):
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_for(cfg, name, project_dir="."):
    sp = cfg_get(cfg, "yolo_obb.split", {"mode": "stable_hash", "train": 0.8, "val": 0.1, "test": 0.1})
    mode = sp.get("mode", "stable_hash")
    train = float(sp.get("train", 0.8))
    val = float(sp.get("val", 0.1))
    test = float(sp.get("test", 0.1))
    total = max(1e-9, train + val + test)
    train /= total
    val /= total

    if mode == "fixed_test_list":
        test_list_path = sp.get("fixed_test_list", "") or ""
        if test_list_path and not os.path.isabs(test_list_path):
            # 相对路径优先相对于 project_dir 解析
            test_list_path = os.path.join(project_dir, test_list_path)
        if test_list_path and os.path.exists(test_list_path):
            try:
                with open(test_list_path, "r", encoding="utf-8") as f:
                    test_images = set(line.strip() for line in f if line.strip())
                if name in test_images:
                    return "test"
            except Exception:
                pass
        # stable_hash for train/val
        total_train_val = max(1e-9, train + val)
        train_ratio = train / total_train_val
        r = stable_ratio(name)
        if r < train_ratio:
            return "train"
        return "val"

    r = stable_ratio(name)
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


# ----------------------------------------------------------------------------
# 掩膜 -> 旋转框
# ----------------------------------------------------------------------------

def load_mask_seg(mask_path, img_w, img_h, min_area=10, max_points=200):
    """从掩膜提最大外轮廓，简化后返回归一化多边形点 [x1,y1,...]（YOLO-seg 格式）。"""
    b = mask_binary(mask_path)
    if b is None:
        return None
    h, w = b.shape[:2]
    binary = b.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    # 按周长比例简化点数，避免标签过长
    eps = 0.001 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    if len(approx) > max_points:
        step = len(approx) // max_points + 1
        approx = approx[::step]
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None
    if w != img_w or h != img_h:
        pts[:, 0] *= img_w / float(w)
        pts[:, 1] *= img_h / float(h)
    out = []
    for x, y in pts:
        out.append(min(max(float(x) / float(img_w), 0.0), 1.0))
        out.append(min(max(float(y) / float(img_h), 0.0), 1.0))
    return out


def seg_line(class_id, poly):
    return str(class_id) + " " + " ".join(f"{v:.6f}" for v in poly)


def load_mask_obb(mask_path, img_w, img_h, min_area=10):
    b = mask_binary(mask_path)
    if b is None:
        return None
    h, w = b.shape[:2]
    binary = b.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    if w != img_w or h != img_h:
        box[:, 0] *= img_w / float(w)
        box[:, 1] *= img_h / float(h)
    box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
    # 按 左上/右上/右下/左下 排序，保证顶点顺序一致
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    return np.array([
        box[np.argmin(s)], box[np.argmin(diff)],
        box[np.argmax(s)], box[np.argmax(diff)],
    ], dtype=np.float32)


def obb_line(class_id, box, img_w, img_h):
    vals = []
    for x, y in box:
        vals.append(min(max(float(x) / float(img_w), 0.0), 1.0))
        vals.append(min(max(float(y) / float(img_h), 0.0), 1.0))
    return str(class_id) + " " + " ".join(f"{v:.6f}" for v in vals)


def discover_classes(cfg, index, state):
    names = cfg_get(cfg, "yolo_obb.class_names", []) or []
    if names:
        return list(names)
    class_src = cfg_get(cfg, "yolo_obb.class_source", "") or ""
    exclude = set(str(e) for e in (cfg_get(cfg, "yolo_obb.exclude_classes", []) or []))
    found = set()
    for item in index:
        tid = item.get("target_id", "")
        rec = state.get(tid, {})
        if rec.get("label") in POSITIVE_LABELS:
            if class_src and class_src.startswith("semantic."):
                sem = rec.get("semantic", {})
                cls = get_nested(sem, class_src[len("semantic."):])
                if cls and str(cls) not in exclude:
                    found.add(str(cls))
            elif item.get("class_name") and str(item["class_name"]) not in exclude:
                found.add(str(item["class_name"]))
    if not found and not class_src:
        for item in index:
            rec = state.get(item.get("target_id"), {})
            if rec.get("label") in POSITIVE_LABELS and item.get("class_name"):
                cls = str(item["class_name"])
                if cls not in exclude:
                    found.add(cls)
    return sorted(found) or [cfg_get(cfg, "sam3.text_prompt", "target")]


def export_one(project_dir, cfg, image, items, state, class_to_id, min_area, fmt="obb", out_dirs=None):
    dirs = out_dirs or run_dirs(project_dir, cfg)
    split = split_for(cfg, image, project_dir)
    ensure_dir(os.path.join(dirs["yolo_images"], split))
    ensure_dir(os.path.join(dirs["yolo_labels"], split))
    src = os.path.join(image_dir(project_dir, cfg), image)
    dst_img = os.path.join(dirs["yolo_images"], split, image)
    dst_label = os.path.join(dirs["yolo_labels"], split, os.path.splitext(image)[0] + ".txt")

    try:
        with Image.open(src) as im:
            img_w, img_h = im.size
    except Exception:
        return None

    class_src = cfg_get(cfg, "yolo_obb.class_source", "") or ""
    require = cfg_get(cfg, "yolo_obb.require", {}) or {}

    lines = []
    has_negative = False
    has_unreviewed = False
    has_skipped = False
    for item in items:
        tid = item.get("target_id", "")
        lab = state.get(tid, {}).get("label", "")
        if not lab:
            has_unreviewed = True
            continue
        if lab in SKIP_LABELS:
            has_skipped = True
            continue
        if lab in NEGATIVE_LABELS:
            has_negative = True
            continue
        if lab not in POSITIVE_LABELS:
            continue

        # 确定类别
        cls = None
        if class_src and class_src.startswith("semantic."):
            sem = state.get(tid, {}).get("semantic", {})
            sem_key = class_src[len("semantic."):]
            cls = get_nested(sem, sem_key)
            # 检查 require 条件（仅支持 semantic.* 路径）
            if require:
                skip = False
                for rk, rv in require.items():
                    if rk.startswith("semantic."):
                        actual = get_nested(sem, rk[len("semantic."):])
                        if actual != rv:
                            skip = True
                            break
                    else:
                        # 非 semantic 路径：从 state/item 取值
                        actual = get_nested(state.get(tid, {}), rk) or get_nested(item, rk)
                        if actual != rv:
                            skip = True
                            break
                if skip:
                    continue
        if not cls:
            cls = item.get("class_name") or next(iter(class_to_id), None)
        if not cls:
            continue

        # exclude_classes 过滤
        exclude = cfg_get(cfg, "yolo_obb.exclude_classes", []) or []
        if exclude and str(cls) in [str(e) for e in exclude]:
            continue

        mf = item.get("mask_file")
        if not mf:
            continue
        mp = os.path.join(dirs["mask"], mf)
        if not os.path.exists(mp):
            continue

        if str(cls) not in class_to_id:
            raise RuntimeError(f"发现未注册类别 {cls}，拒绝静默映射到类别 0")
        cid = class_to_id[str(cls)]
        if fmt == "seg":
            poly = load_mask_seg(mp, img_w, img_h, min_area=min_area)
            if poly is None:
                continue
            lines.append(seg_line(cid, poly))
        else:
            box = load_mask_obb(mp, img_w, img_h, min_area=min_area)
            if box is None:
                continue
            lines.append(obb_line(cid, box, img_w, img_h))

    if has_unreviewed or has_skipped:
        return None
    if not lines and not has_negative:
        return None

    shutil.copy2(src, dst_img)
    with open(dst_label, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return {"image": image, "split": split, "objects": len(lines)}


def export_yolo_obb(project_dir, cfg, progress=None, fmt="obb"):
    _load_export_dependencies()
    dirs = run_dirs(project_dir, cfg)
    index = load_index(project_dir, cfg)
    if not index:
        raise RuntimeError("target 索引为空，请先运行 pipeline2_review.py index")
    state = load_state(project_dir, cfg)
    class_names = discover_classes(cfg, index, state)
    class_to_id = {c: i for i, c in enumerate(class_names)}

    staging = dirs["yolo"] + f".staging.{os.getpid()}"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    work_dirs = dict(dirs)
    work_dirs.update({
        "yolo": staging,
        "yolo_images": os.path.join(staging, "images"),
        "yolo_labels": os.path.join(staging, "labels"),
        "yolo_yaml": os.path.join(staging, "data.yaml"),
        "yolo_meta": os.path.join(staging, "export_meta.json"),
    })

    groups = {}
    for item in index:
        groups.setdefault(item["image"], []).append(item)

    min_area = float(cfg_get(cfg, "yolo_obb.min_mask_area", 10))
    workers = int(cfg_get(cfg, "web.workers", os.cpu_count() or 4))
    total = len(groups)
    stats = {s: {"images": 0, "objects": 0} for s in ("train", "val", "test")}
    exported = done = 0

    if progress:
        progress(0, total, f"starting fmt={fmt}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(export_one, project_dir, cfg, img, items, state, class_to_id, min_area, fmt, work_dirs): img
                for img, items in groups.items()}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r:
                exported += 1
                stats[r["split"]]["images"] += 1
                stats[r["split"]]["objects"] += r["objects"]
            if progress and (done % 50 == 0 or done == total):
                progress(done, total, f"exported={exported}")

    if exported == 0:
        if os.path.exists(staging):
            shutil.rmtree(staging)
        raise RuntimeError("没有可安全导出的完整审核图片，旧数据集保持不变")

    ensure_dir(work_dirs["yolo"])
    with open(work_dirs["yolo_yaml"], "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": dirs["yolo"],
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: n for i, n in enumerate(class_names)},
        }, f, allow_unicode=True, sort_keys=False)

    sp = cfg_get(cfg, "yolo_obb.split", {})
    split_mode = sp.get("mode", "stable_hash")
    meta = {
        "exported_at": datetime.now().isoformat(),
        "format": fmt,
        "class_to_id": class_to_id,
        "stats": stats,
        "exported_images": exported,
        "split_mode": split_mode,
    }
    reviewed_records = [record for record in state.values() if record.get("label")]
    auto_reviewed = sum(1 for record in reviewed_records
                        if record.get("source") == "gemma" or record.get("is_auto"))
    meta["review_provenance"] = {
        "reviewed": len(reviewed_records),
        "auto": auto_reviewed,
        "human": len(reviewed_records) - auto_reviewed,
    }
    if split_mode == "fixed_test_list":
        meta["fixed_test_list"] = sp.get("fixed_test_list", "")
        # count fixed test images
        ftl = sp.get("fixed_test_list", "") or ""
        if ftl and not os.path.isabs(ftl):
            ftl = os.path.join(project_dir, ftl)
        if ftl and os.path.exists(ftl):
            try:
                with open(ftl, "r", encoding="utf-8") as f:
                    test_imgs = [l.strip() for l in f if l.strip()]
                meta["fixed_test_count"] = len(test_imgs)
                meta["missing_test_images"] = len(test_imgs) - stats.get("test", {}).get("images", 0)
            except Exception:
                pass
    with open(work_dirs["yolo_meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    quality = audit_yolo_dataset(staging, cfg, project_dir=project_dir, write_report=True)
    meta["quality_status"] = quality["status"]
    with open(work_dirs["yolo_meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if quality["status"] != "passed" and quality["enforced"]:
        stable_report = dirs["yolo"] + ".quality_report.json"
        safe_write_json(stable_report, quality)
        shutil.rmtree(staging)
        details = "; ".join(item["code"] for item in quality["errors"])
        raise RuntimeError(f"导出数据未通过质量门禁，旧数据集保持不变: {details}\n报告: {stable_report}")

    previous = dirs["yolo"] + ".previous"
    try:
        if os.path.exists(previous):
            shutil.rmtree(previous)
        if os.path.exists(dirs["yolo"]):
            os.replace(dirs["yolo"], previous)
        os.replace(staging, dirs["yolo"])
        if os.path.exists(previous):
            shutil.rmtree(previous)
    except Exception:
        if not os.path.exists(dirs["yolo"]) and os.path.exists(previous):
            os.replace(previous, dirs["yolo"])
        if os.path.exists(staging):
            shutil.rmtree(staging)
        raise

    if progress:
        progress(total, total, f"done fmt={fmt} exported={exported} classes={len(class_names)}")


def ensure_model(model_name, project_dir, cfg):
    """检查模型权重是否存在。没有则给出清晰的指引(不再尝试在线下载,因为多数公司环境拉不到)。

    搜索顺序：
      1) project_dir/<model_name>（用户自己放的）
      2) project_dir/models/<model_name>
      3) ~/.cache/ultralytics/<model_name>（ultralytics 默认）
      4) cfg.train.model_mirrors 配置的 URL 列表（只在用户明确配了时才试）
    """
    candidates = [
        os.path.join(project_dir, model_name),
        os.path.join(project_dir, "models", model_name),
        os.path.expanduser(os.path.join("~", ".cache", "ultralytics", model_name)),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 100 * 1024:
            print(f"[模型] 命中本地: {p}", flush=True)
            return p

    # 用户在 config 里配了镜像才试 - 内置的那些大概率拉不到,不浪费时间
    mirrors = cfg_get(cfg, "train.model_mirrors", []) or []
    if mirrors:
        import urllib.request
        dst = os.path.join(project_dir, "models", model_name)
        ensure_dir(os.path.dirname(dst))
        last_err = None
        for tpl in mirrors:
            url = tpl.replace("{name}", model_name)
            print(f"[模型] 尝试下载: {url}", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 100 * 1024:
                    raise RuntimeError(f"返回体过小({len(data)}字节)，疑似失败页")
                with open(dst + ".part", "wb") as f:
                    f.write(data)
                os.replace(dst + ".part", dst)
                print(f"[模型] 下载成功: {dst} ({len(data)//1024} KB)", flush=True)
                return dst
            except Exception as e:
                last_err = e
                print(f"[模型] 失败: {e}", flush=True)
                try:
                    if os.path.exists(dst + ".part"):
                        os.remove(dst + ".part")
                except Exception:
                    pass
        msg_tail = f"（已试 {len(mirrors)} 个镜像，最后错误: {last_err}）"
    else:
        msg_tail = ""

    # 走到这里就是真没有
    raise FileNotFoundError(
        f"找不到 YOLO 权重 {model_name}{msg_tail}\n"
        f"请把 {model_name} 放到以下任一位置：\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n"
        f"\n"
        f"获取方式建议:\n"
        f"  1. 从能上网的电脑下载: https://github.com/ultralytics/assets/releases\n"
        f"  2. 拷到目标机器对应路径\n"
        f"  3. 或在 config.train.model_mirrors 配置你能访问的镜像 URL（{{name}} 是占位符）"
    )


def _preflight_check(project_dir, cfg, data_yaml, model_path, task, resume=False):
    """训练前的预飞行检查。任何一项失败都抛带清晰描述的错误。"""
    issues = []
    info = {}

    if not os.path.exists(data_yaml):
        issues.append(f"找不到 data.yaml: {data_yaml}（先运行导出？）")
        raise RuntimeError("预飞行失败:\n - " + "\n - ".join(issues))
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            dy = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"data.yaml 解析失败: {e}")
    info["data_yaml"] = data_yaml

    quality = require_quality_pass(os.path.dirname(data_yaml), cfg, project_dir=project_dir)
    info["quality"] = {
        "status": quality["status"],
        "fingerprint": quality["dataset_fingerprint"],
        "warnings": len(quality["warnings"]),
    }

    # 读 export_meta.json 看上次导出的统计(若有),帮助诊断
    meta_path = os.path.join(os.path.dirname(data_yaml), "export_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                em = json.load(f)
            info["export_stats"] = em.get("stats", {})
            info["export_total"] = em.get("exported_images", 0)
            info["export_format"] = em.get("format", "?")
        except Exception:
            pass

    names = dy.get("names") or {}
    if not names:
        issues.append("data.yaml 里 names 为空 - 还没有任何类别")
    info["classes"] = names

    root = dy.get("path") or os.path.dirname(data_yaml)
    splits = {"train": dy.get("train"), "val": dy.get("val")}
    counts = {}
    for split, rel in splits.items():
        if not rel:
            issues.append(f"data.yaml 没有 {split} 字段")
            continue
        img_dir = os.path.join(root, rel) if not os.path.isabs(rel) else rel
        # labels 目录推导:分隔符无关(Windows 下 data.yaml 用 / 而 root 用 \,会出现混合)
        _parts = img_dir.replace("\\", "/").split("/")
        for _i in range(len(_parts) - 1, -1, -1):
            if _parts[_i] == "images":
                _parts[_i] = "labels"
                break
        lbl_dir = "/".join(_parts)
        if not os.path.isdir(img_dir):
            issues.append(f"{split} 图片目录不存在: {img_dir}")
            continue
        imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        lbls_all = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")] if os.path.isdir(lbl_dir) else []
        # 统计非空 vs 空标签文件(空 = 负样本/背景图)
        non_empty = 0
        if os.path.isdir(lbl_dir):
            for f in lbls_all:
                fp = os.path.join(lbl_dir, f)
                try:
                    if os.path.getsize(fp) > 0:
                        non_empty += 1
                except Exception:
                    pass
        empty = len(lbls_all) - non_empty
        counts[split] = {
            "images": len(imgs),
            "label_files": len(lbls_all),
            "positives": non_empty,        # 含至少一个目标
            "backgrounds": empty,          # 空文件 = 负样本图
            "label_dir": lbl_dir,
        }
        if not imgs:
            issues.append(f"{split}: 图片目录为空 ({img_dir})")
            continue
        # 关键诊断:区分"没标签文件"和"标签文件全是空的"
        if len(lbls_all) == 0:
            issues.append(
                f"{split}: 共 {len(imgs)} 张图但 0 个标签文件 ({lbl_dir})\n"
                f"   可能原因: 1) 还没导出 2) 导出后被删 3) labels 目录路径不对"
            )
        elif non_empty == 0:
            issues.append(
                f"{split}: 共 {len(lbls_all)} 个标签文件但**全是空的 (0 个正样本)**\n"
                f"   意味着你打标全是 reject/hard_negative/empty_ok 等负样本类别,\n"
                f"   YOLO 无法只用负样本训练。请至少标几个 accept 或 hard_positive。"
            )
    info["counts"] = counts

    info["model"] = {"path": model_path, "size_mb": 0}
    if model_path.endswith(".pt"):
        if not os.path.exists(model_path):
            issues.append(f"预训练权重不存在: {model_path}")
        elif os.path.getsize(model_path) < 100 * 1024:
            issues.append(f"权重文件似乎无效(< 100KB): {model_path}")
        else:
            info["model"]["size_mb"] = round(os.path.getsize(model_path) / 1024 / 1024, 2)
    elif model_path.endswith(".yaml"):
        info["model"]["mode"] = "from-scratch (架构 yaml,不需要预下载)"

    name = os.path.basename(model_path).lower()
    if not resume:  # resume 时 skip 此检查，因为 last.pt 不含 obb/seg 前缀
        if task == "obb" and "obb" not in name:
            issues.append(f"task=obb 但模型名不含 'obb': {name}")
        if task == "segment" and "seg" not in name:
            issues.append(f"task=segment 但模型名不含 'seg': {name}")

    try:
        import ultralytics  # noqa: F401
        info["ultralytics"] = getattr(__import__("ultralytics"), "__version__", "unknown")
    except ImportError as e:
        import sys as _sys
        import traceback as _tb
        info["python_exe"] = _sys.executable
        info["python_version"] = _sys.version
        info["import_error"] = str(e)
        info["import_error_type"] = type(e).__name__
        info["import_traceback"] = _tb.format_exc().strip()
        # 检查 pip 里有没有 ultralytics
        try:
            import importlib.metadata as _imd
            _imd.version("ultralytics")
            info["pip_has_ultralytics"] = "YES (但 import 失败)"
        except Exception:
            info["pip_has_ultralytics"] = "NO"
        issues.append("Python 包 ultralytics 没装。pip install ultralytics")

    import shutil as _sh
    if not _sh.which("yolo"):
        issues.append("yolo 命令不在 PATH (装了 ultralytics 后通常会自动有)")

    # GPU / CUDA 可用性
    # PyTorch 版本检测（用 subprocess 避免在父进程初始化 CUDA）
    try:
        import sys as _sys
        result = subprocess.run(
            [_sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=15
        )
        info["pytorch_version"] = result.stdout.strip() if result.returncode == 0 else "NOT INSTALLED"
        info["cuda_available"] = "见训练日志"
    except Exception:
        info["pytorch_version"] = "检测失败"
        info["pytorch_version"] = "NOT INSTALLED"
        info["cuda_available"] = "N/A (torch not found)"

    if issues:
        # 把诊断信息也打出来,即使失败
        diag = [f"  {k}: {v}" for k, v in info.items()]
        raise RuntimeError(
            "预飞行检查未通过:\n - " + "\n - ".join(issues) +
            "\n\n[诊断信息]\n" + "\n".join(diag)
        )
    return info


def train_yolo(project_dir, cfg, model=None, imgsz=None, epochs=None, device=None, resume=False):
    dirs = run_dirs(project_dir, cfg)
    data = dirs["yolo_yaml"]
    if not os.path.exists(data):
        raise FileNotFoundError("找不到 data.yaml，请先 export")
    meta = {}
    if os.path.exists(dirs["yolo_meta"]):
        try:
            meta = json.load(open(dirs["yolo_meta"], encoding="utf-8"))
        except Exception:
            meta = {}
    fmt = meta.get("format", "obb")
    task = "segment" if fmt == "seg" else "obb"
    # 固定输出路径
    train_dir = os.path.join(dirs["yolo"], "train")
    
    default_model = "yolo26n-seg.pt" if fmt == "seg" else "yolo26n-obb.pt"
    model = model or cfg_get(cfg, "train.model", default_model)
    
    # resume 模式：跳过模型下载，直接用 last.pt
    if resume:
        last_pt = os.path.join(train_dir, "weights", "last.pt")
        if os.path.exists(last_pt):
            model = last_pt
            print(f"[train] 从 {last_pt} 续训", flush=True)
        else:
            raise FileNotFoundError(f"找不到续训文件: {last_pt}")
    
    # 处理模型:
    # - 如果是文件路径(已存在)直接用
    # - 如果是 .pt 模型名,试本地;找不到就退到对应的 .yaml 架构(从0训练)
    if not resume:
        if os.path.exists(model):
            pass
        elif model.endswith(".pt") and (os.sep not in model and "/" not in model):
            try:
                model = ensure_model(model, project_dir, cfg)
                print(f"[train] 使用预训练权重: {model}", flush=True)
            except Exception as e:
                yaml_alt = model.replace(".pt", ".yaml")
                print(f"[train] 找不到预训练权重 ({e})", flush=True)
                print(f"[train] 自动退回从架构 {yaml_alt} 从 0 训练(慢一些但不依赖下载)", flush=True)
                model = yaml_alt

    # 预飞行检查
    print("=" * 60, flush=True)
    print("[预飞行检查]", flush=True)
    info = _preflight_check(project_dir, cfg, data, model, task, resume=resume)
    print(f"  类别: {info['classes']}", flush=True)
    print(f"  数据计数: {info['counts']}", flush=True)
    print(f"  模型: {info['model']['path']} ({info['model']['size_mb']} MB)", flush=True)
    if "ultralytics" in info:
        print(f"  ultralytics 版本: {info['ultralytics']}", flush=True)
    print("[预飞行检查通过]", flush=True)
    print("=" * 60, flush=True)

    imgsz = str(imgsz or cfg_get(cfg, "train.imgsz", 1024))
    epochs = str(epochs or cfg_get(cfg, "train.epochs", 100))
    batch = str(cfg_get(cfg, "train.batch", None) or "")

    # 内存预检：系统可用内存不足时直接拒绝，比子进程 MemoryError 报错更清晰
    import psutil as _psutil
    _mem = _psutil.virtual_memory()
    _avail_gb = _mem.available / (1024**3)
    if _avail_gb < 2:
        raise RuntimeError(
            f"系统可用内存不足 ({_avail_gb:.1f} GB)，训练至少需要约 2 GB。\n"
            f"建议：关闭其他程序、降低 config.train.imgsz (当前 {imgsz})、\n"
            f"      或设置 config.train.workers=0 config.train.batch=4 减少峰值。")
    print(f"[train] 可用内存: {_avail_gb:.1f} GB, imgsz={imgsz}, workers={cfg_get(cfg, 'train.workers', 0)}", flush=True)

    # 设置环境变量防止 OpenMP 冲突 (子进程中必须重新设置)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # 滚动备份旧模型
    ensure_dir(os.path.join(train_dir, "weights"))
    best_pt = os.path.join(train_dir, "weights", "best.pt")
    results_csv = os.path.join(train_dir, "results.csv")
    keep_backups = 3
    if os.path.exists(best_pt):
        for i in range(keep_backups, 0, -1):
            old = os.path.join(train_dir, "weights", f"best.{i}.pt")
            if os.path.exists(old):
                if i == keep_backups:
                    os.remove(old)
                else:
                    os.replace(old, os.path.join(train_dir, "weights", f"best.{i+1}.pt"))
        os.replace(best_pt, os.path.join(train_dir, "weights", "best.1.pt"))
        print(f"[train] 旧模型已备份到 best.1.pt ~ best.{keep_backups}.pt", flush=True)
    # 冷备份 results.csv — YOLO resume 只保留 resumed epoch 之后的数据
    if not resume and os.path.exists(results_csv):
        i = 1
        while os.path.exists(os.path.join(train_dir, f"results.{i}.csv")):
            i += 1
        if i > keep_backups:
            os.remove(os.path.join(train_dir, f"results.1.csv"))
            for j in range(1, keep_backups):
                src = os.path.join(train_dir, f"results.{j+1}.csv")
                dst = os.path.join(train_dir, f"results.{j}.csv")
                if os.path.exists(src):
                    os.replace(src, dst)
            i = keep_backups
        os.replace(results_csv, os.path.join(train_dir, f"results.{i}.csv"))
        print(f"[train] 旧训练曲线已备份到 results.{i}.csv", flush=True)

    # 使用 sys.executable 所在目录的 yolo，避免 PATH 问题
    yolo_exe = os.path.join(os.path.dirname(sys.executable), "yolo.exe")
    if not os.path.exists(yolo_exe):
        yolo_exe = "yolo"  # fallback
    cmd = [yolo_exe, f"task={task}", "mode=train", f"model={model}", f"data={data}",
           f"imgsz={imgsz}", f"epochs={epochs}",
           f"project={dirs['yolo']}", "name=train", "exist_ok=True"]
    if resume:
        # resume 模式：用 last.pt 续跑，不需要重新指定 model
        last_pt = os.path.join(train_dir, "weights", "last.pt")
        if os.path.exists(last_pt):
            cmd.append(f"resume=True")
        else:
            print(f"[train] resume=True 但找不到 {last_pt}，从头训练", flush=True)
    # 学习率参数：从 config 读取，未配置则使用 YOLO 默认值
    for key, flag in [
        ("lr0", "lr0"), ("lrf", "lrf"), ("momentum", "momentum"),
        ("weight_decay", "weight_decay"), ("warmup_epochs", "warmup_epochs"),
        ("warmup_momentum", "warmup_momentum"), ("warmup_bias_lr", "warmup_bias_lr"),
        ("workers", "workers"),
    ]:
        v = cfg_get(cfg, f"train.{key}", None)
        if v is not None:
            cmd.append(f"{flag}={v}")
    if cfg_get(cfg, "train.cos_lr", False):
        cmd.append("cos_lr=True")
    if batch:
        cmd.append(f"batch={batch}")
    if device is not None:
        cmd.append(f"device={device}")
    elif cfg_get(cfg, "train.device", None) is not None:
        cmd.append(f"device={cfg_get(cfg, 'train.device')}")
    print("运行:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # 关键: 把 stderr 合并到 stdout 一起拿,避免错误信息丢
    p = subprocess.Popen(cmd, cwd=project_dir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env, bufsize=1)
    # 保留最后 80 行,失败时一并报出来
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        print(line, flush=True)
        tail.append(line)
        if len(tail) > 80:
            tail = tail[-80:]
    rc = p.wait()
    if rc != 0:
        snippet = "\n".join(tail)
        raise RuntimeError(
            f"YOLO 训练失败 rc={rc}\n"
            f"--- 子进程最后 80 行 ---\n{snippet}\n"
            f"--- end ---\n"
            f"常见原因:\n"
            f"  1. PyTorch 没装或 CUDA 版本不匹配 (看上面 'No module' 或 'CUDA' 字样)\n"
            f"  2. 显存不足 (看 'CUDA out of memory') — 降低 imgsz 或 batch\n"
            f"  3. 数据集格式错 (看 'Dataset' 或 'labels' 报错)\n"
            f"  4. 模型 task 不匹配 (看 'task' 报错)")
    print(f"[train] 训练完成，模型已保存: {best_pt}", flush=True)
    print(f"[train] 滚动备份: best.1.pt ~ best.{keep_backups}.pt", flush=True)
    print(f"[train] 下次 eval/compare 将自动使用此模型", flush=True)


def predict_yolo(project_dir, cfg, model_path, source, conf=0.25, imgsz=None, device=None):
    """用训练好的模型对单图或文件夹做预测，结果写到 yolo run 目录的 predict/ 子文件夹。"""
    dirs = run_dirs(project_dir, cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not os.path.exists(source):
        raise FileNotFoundError(f"输入不存在: {source}")

    meta = {}
    if os.path.exists(dirs["yolo_meta"]):
        try:
            meta = json.load(open(dirs["yolo_meta"], encoding="utf-8"))
        except Exception:
            meta = {}
    fmt = meta.get("format", "obb")
    task = "segment" if fmt == "seg" else "obb"
    imgsz = str(imgsz or cfg_get(cfg, "train.imgsz", 1024))
    out_dir = os.path.join(dirs["yolo"], "predict")
    ensure_dir(out_dir)
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(out_dir, run_name)
    cmd = [_yolo_executable(), f"task={task}", "mode=predict",
           f"model={model_path}", f"source={source}",
           f"imgsz={imgsz}", f"conf={conf}",
           f"project={out_dir}", f"name={run_name}", "exist_ok=False", "save=True"]
    if device is not None:
        cmd.append(f"device={device}")
    print("运行:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(cmd, cwd=project_dir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"YOLO 预测失败 rc={rc}")
    safe_write_json(os.path.join(out_dir, "latest.json"), {
        "dir": run_dir,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": os.path.abspath(model_path),
        "source": os.path.abspath(source),
    })
    return run_dir


# ============================================================
#  eval / compare — 训练前后 YOLO 表现对比
# ============================================================

def _get_task_from_meta(project_dir, cfg):
    dirs = run_dirs(project_dir, cfg)
    fmt = "obb"
    if os.path.exists(dirs["yolo_meta"]):
        try:
            meta = json.load(open(dirs["yolo_meta"], encoding="utf-8"))
            fmt = meta.get("format", "obb")
        except Exception:
            pass
    return "segment" if fmt == "seg" else "obb", fmt


def _parse_yolo_metrics(lines):
    """Parse both named metric logs and the standard Ultralytics `all` summary row."""
    import re as _re

    metrics = {"precision": None, "recall": None, "mAP50": None, "mAP50-95": None}
    for line in lines:
        m = _re.search(r'mAP50[:\s]*([\d.]+)', line)
        if m:
            metrics["mAP50"] = float(m.group(1))
        m = _re.search(r'mAP50-95[:\s]*([\d.]+)', line)
        if m:
            metrics["mAP50-95"] = float(m.group(1))
        m = _re.search(r'P[:\s]*([\d.]+).*R[:\s]*([\d.]+)', line)
        if m:
            metrics["precision"] = float(m.group(1))
            metrics["recall"] = float(m.group(2))
        m = _re.search(
            r'^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            line,
        )
        if m:
            metrics.update({
                "precision": float(m.group(1)),
                "recall": float(m.group(2)),
                "mAP50": float(m.group(3)),
                "mAP50-95": float(m.group(4)),
            })
    return metrics


def _run_yolo_val(project_dir, cfg, model_path, split="test", imgsz=None, device=None):
    dirs = run_dirs(project_dir, cfg)
    data = dirs["yolo_yaml"]
    if not os.path.exists(data):
        raise FileNotFoundError("找不到 data.yaml，请先 export")
    if cfg_get(cfg, "evaluation.require_quality_pass", True):
        require_quality_pass(dirs["yolo"], cfg, project_dir=project_dir)
    task, fmt = _get_task_from_meta(project_dir, cfg)
    imgsz = str(imgsz or cfg_get(cfg, "train.imgsz", 1024))
    out_dir = os.path.join(dirs["yolo"], "eval")
    ensure_dir(out_dir)

    cmd = [_yolo_executable(), f"task={task}", "mode=val",
           f"model={model_path}", f"data={data}",
           f"imgsz={imgsz}", f"split={split}",
           f"project={out_dir}", "name=run", "exist_ok=True"]
    if device is not None:
        cmd.append(f"device={device}")
    print("运行:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    tail = []
    p = subprocess.Popen(cmd, cwd=project_dir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env, bufsize=1)
    for line in p.stdout:
        line = line.rstrip()
        print(line, flush=True)
        tail.append(line)
        if len(tail) > 80:
            tail = tail[-80:]
    rc = p.wait()
    if rc != 0:
        snippet = "\n".join(tail)
        raise RuntimeError(f"YOLO val 失败 rc={rc}\n{snippet}")

    return _parse_yolo_metrics(tail), out_dir


def do_eval(project_dir, cfg, model_path, split="test", imgsz=None, device=None):
    metrics, out_dir = _run_yolo_val(project_dir, cfg, model_path, split=split,
                                      imgsz=imgsz, device=device)
    dirs = run_dirs(project_dir, cfg)
    eval_dir = os.path.join(dirs["yolo"], "eval")
    ensure_dir(eval_dir)
    out_path = os.path.join(eval_dir, f"{split}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_path,
            "split": split,
            "format": _get_task_from_meta(project_dir, cfg)[1],
            "metrics": metrics,
            "eval_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    print(f"[eval] 指标已保存: {out_path}")
    return metrics


def _ordered_names(value):
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda item: int(item))]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _validate_comparison_models(data_yaml, baseline_model, trained_model, required=True):
    """Require both evaluated models to use the dataset's exact class ontology."""
    with open(data_yaml, "r", encoding="utf-8") as handle:
        dataset_names = _ordered_names((yaml.safe_load(handle) or {}).get("names"))
    try:
        from ultralytics import YOLO
        baseline_names = _ordered_names(YOLO(baseline_model).names)
        trained_names = _ordered_names(YOLO(trained_model).names)
    except Exception as exc:
        raise RuntimeError(f"无法读取模型类别元数据: {exc}") from exc
    compatibility = {
        "status": "passed" if baseline_names == dataset_names and trained_names == dataset_names else "failed",
        "dataset_names": dataset_names,
        "baseline_names": baseline_names,
        "trained_names": trained_names,
    }
    if compatibility["status"] != "passed" and required:
        raise RuntimeError(
            "模型类别体系不兼容，禁止生成误导性验收对比。\n"
            f"数据集类别: {dataset_names}\n"
            f"基线模型类别: {baseline_names}\n"
            f"候选模型类别: {trained_names}\n"
            "请选择使用相同业务类别训练并已批准的上一版本作为基线。"
        )
    return compatibility


def do_compare(project_dir, cfg, baseline_model, trained_model, split="test", imgsz=None, device=None):
    dirs = run_dirs(project_dir, cfg)
    eval_dir = os.path.join(dirs["yolo"], "eval")
    ensure_dir(eval_dir)

    quality = require_quality_pass(dirs["yolo"], cfg, project_dir=project_dir)
    compatibility = _validate_comparison_models(
        dirs["yolo_yaml"], baseline_model, trained_model,
        required=bool(cfg_get(cfg, "evaluation.require_class_compatibility", True)),
    )

    print(f"--- baseline: {baseline_model} ---")
    b_metrics, _ = _run_yolo_val(project_dir, cfg, baseline_model, split=split,
                                  imgsz=imgsz, device=device)
    print(f"--- trained: {trained_model} ---")
    t_metrics, _ = _run_yolo_val(project_dir, cfg, trained_model, split=split,
                                  imgsz=imgsz, device=device)

    fmt = _get_task_from_meta(project_dir, cfg)[1]
    delta = {}
    for k in b_metrics:
        if b_metrics.get(k) is not None and t_metrics.get(k) is not None:
            delta[k] = round(t_metrics[k] - b_metrics[k], 4)
        else:
            delta[k] = None

    report = {
        "format": fmt,
        "split": split,
        "baseline_model": baseline_model,
        "trained_model": trained_model,
        "baseline": b_metrics,
        "trained": t_metrics,
        "delta": delta,
        "quality": {"status": quality["status"], "dataset_fingerprint": quality["dataset_fingerprint"]},
        "compatibility": compatibility,
        "compared_at": datetime.now().isoformat(),
    }
    thresholds = {
        "precision": float(cfg_get(cfg, "evaluation.min_precision", 0.75)),
        "recall": float(cfg_get(cfg, "evaluation.min_recall", 0.75)),
        "mAP50-95": float(cfg_get(cfg, "evaluation.min_map50_95", 0.60)),
    }
    max_regression = float(cfg_get(cfg, "evaluation.max_regression", 0.02))
    failed_checks = [key for key, limit in thresholds.items()
                     if t_metrics.get(key) is None or t_metrics[key] < limit]
    failed_checks.extend(f"regression:{key}" for key, value in delta.items()
                         if value is not None and value < -max_regression)
    report["acceptance"] = {
        "status": "passed" if not failed_checks else "failed",
        "thresholds": thresholds,
        "max_regression": max_regression,
        "failed_checks": failed_checks,
    }
    report_path = os.path.join(eval_dir, "compare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = os.path.join(eval_dir, "compare_report.csv")
    import csv as _csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["metric", "baseline", "trained", "delta"])
        for k in b_metrics:
            w.writerow([k, b_metrics.get(k, ""), t_metrics.get(k, ""), delta.get(k, "")])

    print(f"[compare] report -> {report_path}")
    print(f"[compare] csv    -> {csv_path}")

    # 测试集图片可视化对比：同一张图 baseline vs trained 左右看
    try:
        import random as _random
        # 从 test 标注目录随机抽 6 张图
        test_label_dir = os.path.join(dirs["yolo_labels"], split)
        test_img_dir = os.path.join(dirs["yolo_images"], split)
        if os.path.isdir(test_img_dir):
            candidates = sorted(os.listdir(test_img_dir))
            _random.seed(42)
            _random.shuffle(candidates)
            samples = candidates[:20]  # 预测对比样本数
            compare_dir = os.path.join(eval_dir, "compare_images")
            ensure_dir(compare_dir)
            source_dir = os.path.join(compare_dir, "source")
            for generated in (source_dir, os.path.join(compare_dir, "baseline"), os.path.join(compare_dir, "trained")):
                if os.path.isdir(generated):
                    shutil.rmtree(generated)
            ensure_dir(source_dir)
            samples_json = []
            for img_name in samples:
                src = os.path.join(test_img_dir, img_name)
                dst = os.path.join(source_dir, img_name)
                shutil.copy2(src, dst)
                samples_json.append(img_name)
            # 用两个模型分别 predict，输出到 compare_dir
            task, _ = _get_task_from_meta(project_dir, cfg)
            imgsz_str = str(imgsz or cfg_get(cfg, "train.imgsz", 1024))
            for model_path, tag in [(baseline_model, "baseline"), (trained_model, "trained")]:
                out_sub = os.path.join(compare_dir, tag)
                ensure_dir(out_sub)
                cmd = [_yolo_executable(), f"task={task}", "mode=predict",
                       f"model={model_path}", f"source={source_dir}",
                       f"imgsz={imgsz_str}", f"project={out_sub}",
                       "name=run", "exist_ok=True", "save_txt=False", "save_conf=False",
                       "line_width=2"]
                if device is not None:
                    cmd.append(f"device={device}")
                elif cfg_get(cfg, "train.device", None) is not None:
                    cmd.append(f"device={cfg_get(cfg, 'train.device')}")
                print(f"[compare/predict] {tag}: {' '.join(cmd)}")
                env = os.environ.copy()
                p = subprocess.Popen(cmd, cwd=project_dir, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", env=env, bufsize=1)
                for line in p.stdout:
                    print(f"  {line.rstrip()}", flush=True)
                rc = p.wait()
                if rc != 0:
                    print(f"  [compare/predict] {tag} 预测失败 rc={rc}")
            # 记录样本文件列表
            with open(os.path.join(compare_dir, "samples.json"), "w", encoding="utf-8") as f:
                json.dump({"format": fmt, "images": samples_json,
                           "baseline_dir": os.path.join(compare_dir, "baseline", "run"),
                           "trained_dir": os.path.join(compare_dir, "trained", "run")}, f)
            print(f"[compare] 可视化对比图 -> {compare_dir}")
    except Exception as e:
        print(f"[compare] 可视化生成失败: {e}")

    return report


# ============================================================
#  Pipeline 2.5: YOLO predict → SAM3 target JSON → 送审
# ============================================================

def predict_to_targets(project_dir, cfg, model_path, conf=0.25, imgsz=None, device=None):
    """YOLO 在全量图上跑 predict，把检出转成 SAM3 兼容 JSON，直接写入 merged 目录。
    build_index 会自动读取，无需额外步骤。
    """
    import numpy as np
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("predict_to_targets 需要 pip install ultralytics")

    dirs = run_dirs(project_dir, cfg)
    idir = image_dir(project_dir, cfg)
    out_dir = dirs.get("merged") or dirs["sam3"]
    ensure_dir(out_dir)

    files = sorted(f for f in os.listdir(idir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    total = len(files)
    if not total:
        print("图片目录无文件", flush=True)
        return

    imgsz_val = int(imgsz or cfg_get(cfg, "train.imgsz", 1024))
    conf_val = float(conf)
    model = YOLO(model_path)
    if device is not None:
        model.to(device)

    print(f"[predict_targets] {total} 张图 -> {out_dir}", flush=True)
    count = 0
    for n, fname in enumerate(files):
        out_json = os.path.join(out_dir, f"yolo_{os.path.splitext(fname)[0]}.json")
        if os.path.exists(out_json):
            continue

        img_path = os.path.join(idir, fname)
        results = model.predict(img_path, imgsz=imgsz_val, conf=conf_val,
                                device=device, verbose=False)
        targets = []
        for r in results:
            if r.obb is not None:
                for box, cls, conf_val2 in zip(
                        r.obb.xyxyxyxy.cpu().numpy() if r.obb.xyxyxyxy is not None else [],
                        r.obb.cls.cpu().numpy() if r.obb.cls is not None else [],
                        r.obb.conf.cpu().numpy() if r.obb.conf is not None else []):
                    corners = box.tolist()[:4]
                    xs = [p[0] for p in corners]
                    ys = [p[1] for p in corners]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    cls_name = r.names.get(int(cls), f"class_{int(cls)}")
                    targets.append({
                        "bbox": bbox,
                        "confidence": float(conf_val2),
                        "class_name": cls_name,
                        "source_prompt": "YOLO predict",
                        "source_prompt_slug": "yolo_pred",
                        "source_target_id": f"{fname}::yolo::{cls_name}::{len(targets)}",
                    })
        if targets:
            slim = {
                "source_file": fname,
                "text_prompt": "__YOLO_PREDICT__",
                "prompt_mode": "yolo",
                "targets": targets,
            }
            safe_write_json(out_json, slim)
            count += len(targets)

        if (n + 1) % 50 == 0 or n == total - 1:
            emit_progress(n + 1, total, f"targets={count}")

    print(f"[predict_targets] 完成 {total} 张图，{count} 个新 target -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "audit", "sanitize", "golden", "golden_validate", "golden_install", "train", "predict", "eval", "compare", "predict_targets", "status"])
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--fmt", default="obb", choices=["obb", "seg"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--baseline", default=None, help="compare 时 baseline 模型路径")
    ap.add_argument("--trained", default=None, help="compare 时 trained 模型路径")
    ap.add_argument("--imgsz", default=None)
    ap.add_argument("--epochs", default=None)
    ap.add_argument("--resume", action="store_true", default=False, help="从 last.pt 续训")
    ap.add_argument("--source", default=None, help="predict 时输入图片或目录")
    ap.add_argument("--split", default="test", help="eval/compare 的数据集划分")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="训练/预测设备: 0,1,2... (GPU序号) 或 cpu")
    ap.add_argument("--no-fail", action="store_true", help="audit 仅生成报告，不以非零状态退出（Web 使用）")
    args = ap.parse_args()
    project_dir = os.path.abspath(args.project_dir)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config)
    cfg = load_config(cfg_path)
    if args.prompt:
        cfg.setdefault("sam3", {})["text_prompt"] = args.prompt

    if args.step == "export":
        export_yolo_obb(project_dir, cfg, progress=emit_progress, fmt=args.fmt)
    elif args.step == "audit":
        report = audit_yolo_dataset(run_dirs(project_dir, cfg)["yolo"], cfg,
                                    project_dir=project_dir, write_report=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "passed" and report["enforced"] and not args.no_fail:
            raise SystemExit(2)
    elif args.step == "sanitize":
        source = run_dirs(project_dir, cfg)["yolo"]
        candidate = source + "_clean_candidate"
        result = build_sanitized_candidate(source, cfg, candidate)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.step == "golden":
        source = run_dirs(project_dir, cfg)["yolo"] + "_clean_candidate"
        output = os.path.join(project_dir, "quality", "golden_review_v1")
        print(json.dumps(build_golden_candidate(source, output, count=300), ensure_ascii=False, indent=2))
    elif args.step == "golden_validate":
        package = os.path.join(project_dir, "quality", "golden_review_v1")
        report = validate_golden_package(package)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "passed":
            raise SystemExit(2)
    elif args.step == "golden_install":
        package = os.path.join(project_dir, "quality", "golden_review_v1")
        dataset = run_dirs(project_dir, cfg)["yolo"] + "_clean_candidate"
        print(json.dumps(install_golden_package(package, project_dir, dataset), ensure_ascii=False, indent=2))
    elif args.step == "train":
        train_yolo(project_dir, cfg, model=args.model, imgsz=args.imgsz, epochs=args.epochs, device=args.device, resume=args.resume)
    elif args.step == "predict":
        if not args.model or not args.source:
            raise SystemExit("predict 需要 --model 和 --source")
        out = predict_yolo(project_dir, cfg, args.model, args.source, conf=args.conf, imgsz=args.imgsz, device=args.device)
        print(f"预测结果已保存到: {out}")
    elif args.step == "status":
        print("yolo:", run_dirs(project_dir, cfg)["yolo"])
    elif args.step == "eval":
        if not args.model:
            raise SystemExit("eval 需要 --model")
        do_eval(project_dir, cfg, args.model, split=args.split,
                imgsz=args.imgsz, device=args.device)
    elif args.step == "compare":
        if not args.baseline or not args.trained:
            raise SystemExit("compare 需要 --baseline 和 --trained")
        do_compare(project_dir, cfg, args.baseline, args.trained,
                   split=args.split, imgsz=args.imgsz, device=args.device)
    elif args.step == "predict_targets":
        if not args.model:
            raise SystemExit("predict_targets 需要 --model（训练好的权重路径）")
        predict_to_targets(project_dir, cfg, args.model,
                           conf=args.conf, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
