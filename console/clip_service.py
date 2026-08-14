#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 + CLIP 视觉参考匹配服务 (端口 8004)。

流程:
1. 用户在图片上画 bbox → register_reference() 注册特征
2. scan_images() 遍历全图调 auto_segment 找相似目标
3. convert_to_targets() 把 CLIP 结果转成 SAM3 兼容 JSON
4. 写入 dirs["merged"] 目录，自动进入 review 流程
"""
import os
import json
import urllib.request
import urllib.error

from console.core import run_dirs, image_dir, cfg_get, ensure_dir, safe_write_json


def _clip_url(cfg, path):
    base = cfg_get(cfg, "clip.api_url", "http://127.0.0.1:8004").rstrip("/")
    return f"{base}{path}"


def register_reference(project_dir, cfg, image_name, bbox):
    """在指定图片的 bbox 区域注册参考特征。返回 {ok, feature_id, ...}。"""
    idir = image_dir(project_dir, cfg)
    img_path = os.path.join(idir, image_name)
    if not os.path.exists(img_path):
        return {"ok": False, "error": f"图片不存在: {image_name}"}

    import io
    boundary = "----FormBoundary" + os.urandom(16).hex()
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="image"; filename="{image_name}"\r\n'.encode())
    body.write(b"Content-Type: image/png\r\n\r\n")
    with open(img_path, "rb") as f:
        body.write(f.read())
    body.write(f"\r\n--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="bbox"\r\n\r\n')
    body.write(json.dumps(bbox).encode())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    timeout = int(cfg_get(cfg, "clip.timeout", 120))
    req = urllib.request.Request(
        _clip_url(cfg, "/api/register_reference"),
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        if resp.get("success"):
            return {
                "ok": True,
                "feature_id": resp.get("feature_id"),
                "features": resp.get("features"),
                "reference_image_base64": resp.get("reference_image_base64"),
                "message": resp.get("message"),
            }
        return {"ok": False, "error": resp.get("error", "CLIP 注册失败")}
    except Exception as e:
        return {"ok": False, "error": f"CLIP 注册请求失败: {e}"}


def auto_segment(project_dir, cfg, image_name, feature_id):
    """对单张图调用 auto_segment。返回 {ok, found, masks, ...}。"""
    idir = image_dir(project_dir, cfg)
    img_path = os.path.join(idir, image_name)
    if not os.path.exists(img_path):
        return {"ok": False, "error": f"图片不存在: {image_name}"}

    threshold = cfg_get(cfg, "clip.similarity_threshold", 0.7)

    import io
    boundary = "----FormBoundary" + os.urandom(16).hex()
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="image"; filename="{image_name}"\r\n'.encode())
    body.write(b"Content-Type: image/png\r\n\r\n")
    with open(img_path, "rb") as f:
        body.write(f.read())
    body.write(f"\r\n--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="feature_id"\r\n\r\n')
    body.write(feature_id.encode())
    body.write(f"\r\n--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="similarity_threshold"\r\n\r\n')
    body.write(str(threshold).encode())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    timeout = int(cfg_get(cfg, "clip.timeout", 120))
    req = urllib.request.Request(
        _clip_url(cfg, "/api/auto_segment"),
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        if resp.get("success") and resp.get("found"):
            return {
                "ok": True,
                "found": True,
                "similarity": resp.get("similarity"),
                "masks": resp.get("masks", []),
                "num_objects": resp.get("num_objects", 0),
            }
        if resp.get("success"):
            return {"ok": True, "found": False, "masks": [], "num_objects": 0}
        return {"ok": False, "error": resp.get("error", "CLIP 匹配失败")}
    except Exception as e:
        return {"ok": False, "error": f"CLIP 匹配请求失败: {e}"}


def list_features(project_dir, cfg):
    """列出所有已注册的 CLIP 参考特征。"""
    timeout = int(cfg_get(cfg, "clip.timeout", 120))
    req = urllib.request.Request(_clip_url(cfg, "/api/list_features"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        if resp.get("success"):
            return {"ok": True, "features": resp.get("features", []), "count": resp.get("count", 0)}
        return {"ok": False, "error": resp.get("error", "获取列表失败")}
    except Exception as e:
        return {"ok": False, "error": f"CLIP 列表请求失败: {e}"}


def delete_feature(project_dir, cfg, feature_id):
    """删除指定 CLIP 参考特征。"""
    timeout = int(cfg_get(cfg, "clip.timeout", 120))
    req = urllib.request.Request(
        _clip_url(cfg, f"/api/delete_feature/{feature_id}"),
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        if resp.get("success"):
            return {"ok": True, "message": resp.get("message")}
        return {"ok": False, "error": resp.get("error", "删除失败")}
    except Exception as e:
        return {"ok": False, "error": f"CLIP 删除请求失败: {e}"}


def convert_to_targets(fname, clips, feature_id):
    """将 auto_segment 返回的 masks 转成 SAM3 兼容 target 列表。"""
    targets = []
    for i, m in enumerate(clips):
        bbox = m.get("bbox", [])
        sim = m.get("similarity", 0)
        targets.append({
            "bbox": bbox,
            "confidence": round(float(sim), 4),
            "class_name": "clip_match",
            "source_prompt": f"CLIP:{feature_id}",
            "source_prompt_slug": f"clip_{feature_id[:12]}",
            "source_target_id": f"{fname}::clip::{feature_id}::{i}",
            "mask_file": None,
        })
    return targets


def scan_images(project_dir, cfg, feature_id, progress=None):
    """遍历所有图片调用 auto_segment，结果写入 merged 目录。"""
    dirs = run_dirs(project_dir, cfg)
    idir = image_dir(project_dir, cfg)
    out_dir = dirs.get("merged") or dirs["sam3"]
    ensure_dir(out_dir)

    files = sorted(f for f in os.listdir(idir) if f.endswith(".png"))
    total = len(files)
    if not total:
        return {"ok": False, "error": "图片目录无 PNG 文件"}

    total_targets = 0
    matched_images = 0
    errors = []

    for n, fname in enumerate(files):
        try:
            r = auto_segment(project_dir, cfg, fname, feature_id)
            if not r["ok"]:
                errors.append(f"{fname}: {r.get('error')}")
                continue
            if not r["found"]:
                continue

            matched_images += 1
            targets = convert_to_targets(fname, r["masks"], feature_id)
            if targets:
                slim = {
                    "source_file": fname,
                    "text_prompt": f"CLIP_REF:{feature_id}",
                    "prompt_mode": "clip",
                    "targets": targets,
                }
                out_name = fname.replace(".png", f"_clip_{feature_id[:12]}.json")
                safe_write_json(os.path.join(out_dir, out_name), slim)
                total_targets += len(targets)

        except Exception as e:
            errors.append(f"{fname}: {e}")

        if progress and ((n + 1) % 50 == 0 or n == total - 1):
            progress(n + 1, total, f"matched={matched_images} targets={total_targets}")

    return {
        "ok": True,
        "total_images": total,
        "matched_images": matched_images,
        "total_targets": total_targets,
        "feature_id": feature_id,
        "errors": (errors[:5] + ["..."] + errors[-5:]) if len(errors) > 10 else errors,
    }
