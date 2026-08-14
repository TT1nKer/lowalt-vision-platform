
import os
import re
import json
import yaml
import hashlib

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("infer", {})
    cfg["infer"]["save_mask"] = True
    cfg.setdefault("web", {})
    cfg.setdefault("yolo_obb", {})
    return cfg

def prompt_slug(prompt):
    s = re.sub(r"[^\w]+", "_", prompt.strip()).strip("_").lower()
    if len(s) > 60:
        h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:6]
        s = s[:50] + "_" + h
    return s or "default"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def project_path(project_dir, *parts):
    return os.path.abspath(os.path.join(project_dir, *parts))

def run_dirs(project_dir, cfg):
    root = project_path(project_dir, cfg["paths"]["results_root"])
    slug = prompt_slug(cfg["sam3"]["text_prompt"])
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
        "index": os.path.join(base, "review_cache", "target_index.json"),
        "overlay": os.path.join(base, "review_cache", "overlay"),
        "state": os.path.join(base, "review", "target_state.json"),
        "jsonl": os.path.join(base, "review", "target_reviews.jsonl"),
        "csv": os.path.join(base, "review", "target_export.csv"),
        "yolo": os.path.join(base, "yolo_obb"),
        "yolo_images": os.path.join(base, "yolo_obb", "images"),
        "yolo_labels": os.path.join(base, "yolo_obb", "labels"),
        "yolo_yaml": os.path.join(base, "yolo_obb", "data.yaml"),
        "yolo_meta": os.path.join(base, "yolo_obb", "export_meta.json"),
    }

def image_dir(project_dir, cfg):
    p = cfg["paths"]["merged_dir"]
    return p if os.path.isabs(p) else project_path(project_dir, p)

def safe_load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def target_id_for(image, idx):
    return f"{image}::{idx}"

def overlay_filename(target_id):
    return hashlib.md5(target_id.encode("utf-8")).hexdigest() + ".jpg"
