#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma(或其它多模态 LLM) 预审：给单个 target 一个建议标签。

精度优先方案（本地部署、不计较 token）：
- 用 OBB 斜框替代正框（从掩膜算的 4 角点），精准贴合目标朝向
- 默认 crop + 全图双图喂：crop 让模型看清细节，全图给上下文判断
- crop 用 OBB 外接矩形 + padding 扩边（默认 70%）
- 不缩小图、PNG 无损编码
- prompt 写清概念名、SAM3 置信度、每个标签含义
- 多图失败自动回退到单图(crop)，不让一次失败弄崩任务
"""
import os
import re
import json
import base64
import urllib.request
import urllib.error

import cv2
import numpy as np

from console.core import run_dirs, image_dir, cfg_get
from console.pipeline2_review import mask_binary

DEFAULT_LABEL_GLOSSARY = {
    "accept": "the target is correctly detected; the mask precisely covers an instance of the target concept",
    "reject": "the highlighted region is NOT an instance of the target concept (wrong object)",
    "hard_positive": "this IS the target concept but it is a difficult/atypical instance (partial, occluded, unusual angle)",
    "hard_negative": "this is NOT the target concept but visually resembles it (a confusing wrong example)",
    "needs_review": "you are unsure; the image is ambiguous",
    "bad_mask": "the concept is correct but the mask is clearly wrong (too small/large/wrong shape)",
    "empty_ok": "the image contains no instance of the target concept and this detection should not exist",
}

# 内置默认提示词，针对无人机俯视场景，含决策表和常见混淆点。
# 用户可在网页或文件里覆盖。{concept}/{labels} 是模板占位符。
BUILTIN_PROMPT = """\
You are reviewing detections on a TOP-DOWN aerial/satellite-style image captured by a drone. Your job: judge whether the highlighted region is a correct detection of the concept "{concept}".

WHAT YOU SEE
- A close-up CROP centered on the candidate region (best for checking visual details of the target).
- The FULL original tile (best for understanding context: what is around the region).
- Overlays on both: a red rotated polygon = the oriented bounding box of the SAM mask; a thin blue outline = the mask boundary itself. The actual pixels are NOT covered by the mask, so you can see the surface clearly.

VISUAL EVIDENCE
Decide based on what you can actually SEE in the image. For a top-down view, look for the characteristic shape, texture, and surroundings of "{concept}". Pay special attention to:
- Whether the region truly contains an instance of "{concept}" (vs a visually similar but different surface/object)
- Whether the SAM mask is on the right object (vs accidentally on a neighboring object or empty ground)

DECISION RULES (apply in order)
1. The region clearly contains "{concept}" and the mask is well-placed → accept
2. It contains "{concept}" but is partial/occluded/atypical/edge case → hard_positive
3. It is something that LOOKS like "{concept}" but isn't → hard_negative
4. It is clearly something else (wrong category entirely) → reject
5. The concept is correct but the mask shape is grossly wrong (way too small/large/leaking) → bad_mask
6. There is no instance of "{concept}" in this tile at all → empty_ok
7. You genuinely cannot tell → needs_review

LABEL MEANINGS
{labels_desc}

OUTPUT FORMAT
First write 1–2 short lines stating what you see (e.g. "Paved surface, no visible markings.").
Then on the FINAL LINE, output ONLY the chosen label (one word, lowercase), nothing else.

Allowed labels: {labels}
"""

_MASK_COLOR_BGR = (255, 161, 78)
_OBB_COLOR_BGR = (60, 60, 230)
_MASK_OUTLINE_BGR = (255, 200, 120)   # 轮廓用更亮的色
_MASK_ALPHA = 0.40


def _read_mask(dirs, item, h, w):
    mf = item.get("mask_file")
    if not mf:
        return None
    mp = os.path.join(dirs["mask"], mf)
    if not os.path.exists(mp):
        return None
    b = mask_binary(mp)
    if b is None:
        return None
    if b.shape[:2] != (h, w):
        b = cv2.resize(b.astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return b


def _largest_contour(bmask):
    binary = (bmask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _apply_mask_overlay(img, bmask, style="outline"):
    """在 img 上画掩膜。style: filled(半透明填充) / outline(只画轮廓线) / none(不画)。
    outline 是默认，不遮挡像素，对停车场标线、纹理细节更友好。
    """
    if style == "none" or bmask is None or not bmask.any():
        return img
    if style == "filled":
        color = np.empty_like(img)
        color[:] = _MASK_COLOR_BGR
        img[bmask] = (img[bmask].astype(np.float32) * (1 - _MASK_ALPHA)
                      + color[bmask].astype(np.float32) * _MASK_ALPHA).astype(np.uint8)
    else:   # outline
        binary = (bmask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cv2.drawContours(img, cnts, -1, _MASK_OUTLINE_BGR, 2, cv2.LINE_AA)
    return img


def _draw_obb_or_aabb(img, bmask, bbox, color=_OBB_COLOR_BGR, thickness=4):
    if bmask is not None:
        c = _largest_contour(bmask)
        if c is not None and cv2.contourArea(c) >= 1:
            box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
            cv2.polylines(img, [box], True, color, thickness, cv2.LINE_AA)
            return "obb"
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return "aabb"
    return None


def _encode_png_b64(img):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _get_crop_rect(base_img, bmask, bbox, padding=0.7):
    """计算裁剪区域，返回 (x1,y1,x2,y2)。没有 bbox 时返回 None。"""
    h, w = base_img.shape[:2]
    src_x1 = src_y1 = src_x2 = src_y2 = None
    if bmask is not None:
        c = _largest_contour(bmask)
        if c is not None and cv2.contourArea(c) >= 1:
            obb_pts = cv2.boxPoints(cv2.minAreaRect(c))
            src_x1 = float(np.min(obb_pts[:, 0])); src_y1 = float(np.min(obb_pts[:, 1]))
            src_x2 = float(np.max(obb_pts[:, 0])); src_y2 = float(np.max(obb_pts[:, 1]))
    if src_x1 is None and bbox and len(bbox) == 4:
        src_x1, src_y1, src_x2, src_y2 = [float(v) for v in bbox]
    if src_x1 is None:
        return None
    bw = src_x2 - src_x1
    bh = src_y2 - src_y1
    cx = (src_x1 + src_x2) / 2
    cy = (src_y1 + src_y2) / 2
    half = max(bw, bh) * (1 + padding) / 2
    x1 = int(max(0, round(cx - half)))
    y1 = int(max(0, round(cy - half)))
    x2 = int(min(w, round(cx + half)))
    y2 = int(min(h, round(cy + half)))
    return x1, y1, x2, y2


def _compose_full(base_img, bmask, bbox, mask_style="outline"):
    img = base_img.copy()
    if bmask is not None:
        _apply_mask_overlay(img, bmask, style=mask_style)
    return img


def _compose_crop(base_img, bmask, bbox, padding=0.7, mask_style="outline"):
    rect = _get_crop_rect(base_img, bmask, bbox, padding=padding)
    if rect is None:
        return _compose_full(base_img, bmask, bbox, mask_style=mask_style)
    x1, y1, x2, y2 = rect
    crop = base_img[y1:y2, x1:x2].copy()
    crop_mask = bmask[y1:y2, x1:x2] if bmask is not None else None
    if crop_mask is not None:
        _apply_mask_overlay(crop, crop_mask, style=mask_style)
    return crop


def compose_views(project_dir, cfg, item, padding=0.7):
    dirs = run_dirs(project_dir, cfg)
    src = os.path.join(image_dir(project_dir, cfg), item["image"])
    base_img = cv2.imread(src, cv2.IMREAD_COLOR)
    if base_img is None:
        raise RuntimeError(f"无法读图: {src}")
    h, w = base_img.shape[:2]
    bmask = _read_mask(dirs, item, h, w)
    bbox = item.get("bbox")
    mode = (cfg_get(cfg, "gemma.image_mode", "both") or "both").lower()
    mask_style = (cfg_get(cfg, "gemma.mask_style", "outline") or "outline").lower()
    structured = bool(cfg_get(cfg, "gemma.structured_output", False))
    views = []
    # 结构化模式始终发原图裁剪（无标注，Gemma 看清像素）
    if structured:
        rect = _get_crop_rect(base_img, bmask, bbox, padding=padding)
        if rect:
            x1, y1, x2, y2 = rect
            raw_crop = base_img[y1:y2, x1:x2].copy()
            views.append(("raw_crop", _encode_png_b64(raw_crop)))
    if mode in ("crop", "both"):
        views.append(("crop", _encode_png_b64(_compose_crop(base_img, bmask, bbox,
                                                            padding=padding, mask_style=mask_style))))
    if mode in ("full", "both"):
        views.append(("full", _encode_png_b64(_compose_full(base_img, bmask, bbox,
                                                            mask_style=mask_style))))
    if not views:
        views.append(("full", _encode_png_b64(_compose_full(base_img, bmask, bbox, mask_style=mask_style))))
    return views


def _render_prompt(template, concept, allowed):
    """渲染提示词模板的占位符 {concept}/{labels}/{labels_desc}。"""
    labels_desc = "\n".join(f"- {lab}: {DEFAULT_LABEL_GLOSSARY.get(lab, '')}" for lab in allowed)
    return (template
            .replace("{concept}", concept)
            .replace("{labels_desc}", labels_desc)
            .replace("{labels}", ", ".join(allowed)))


def prompt_file_path(project_dir, cfg):
    """每个 run 自己的 prompt 覆盖文件路径，网页编辑时存这里。"""
    dirs = run_dirs(project_dir, cfg)
    return os.path.join(dirs["base"], "gemma_prompt.txt")


def get_active_prompt_template(project_dir, cfg):
    """返回当前生效的提示词模板（未渲染）。优先级：
    1. run 目录里的 gemma_prompt.txt（网页编辑保存的）
    2. config.gemma.prompt（用户在 yaml 里写的）
    3. 内置 BUILTIN_PROMPT
    """
    p = prompt_file_path(project_dir, cfg)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                t = f.read().strip()
            if t:
                return t, "file"
        except Exception:
            pass
    custom = cfg_get(cfg, "gemma.prompt") or ""
    if custom.strip():
        return custom, "config"
    return BUILTIN_PROMPT, "builtin"


def _build_prompt(project_dir, cfg, item, allowed, views):
    from console.core import cfg_get as _cg
    structured = _cg(cfg, "gemma.structured_output", False)
    if structured:
        concept = (_cg(cfg, "gemma.target_concept") or _cg(cfg, "sam3.text_prompt") or "the target concept").strip()
        return _build_structured_prompt(cfg, concept)
    concept = (_cg(cfg, "sam3.text_prompt") or "the target concept").strip()
    template, _ = get_active_prompt_template(project_dir, cfg)
    return _render_prompt(template, concept, allowed)


def _content_with_images(prompt, views, schema):
    if schema == "gemini":
        parts = [{"text": prompt}]
        for _, b64 in views:
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        return parts
    content = [{"type": "text", "text": prompt}]
    for _, b64 in views:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return content


def _call_openai(url, model, api_key, prompt, views, timeout=120):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _content_with_images(prompt, views, "openai")}],
        "temperature": 0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8", errors="ignore"))
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp)[:300]


def _call_gemini(url, api_key, prompt, views, timeout=120):
    full_url = url
    if api_key and "key=" not in url:
        sep = "&" if "?" in url else "?"
        full_url = f"{url}{sep}key={api_key}"
    body = {
        "contents": [{"parts": _content_with_images(prompt, views, "gemini")}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
    }
    req = urllib.request.Request(full_url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8", errors="ignore"))
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(resp)[:300]


def _parse_label(text, allowed):
    if not text:
        return None
    s = text.strip()
    last_line = s.splitlines()[-1].strip().lower() if s.splitlines() else ""
    for lab in allowed:
        if last_line == lab.lower():
            return lab
    s_low = s.lower()
    for lab in sorted(allowed, key=lambda x: -len(x)):
        if re.search(rf'(?<![a-z0-9_]){re.escape(lab.lower())}(?![a-z0-9_])', s_low):
            return lab
    return None


def _parse_structured(text, cfg):
    """从 LLM 回复中解析结构化 JSON。失败返回 None。"""
    if not text:
        return None
    text = text.strip()
    # 尝试找 ```json ... ``` 块
    m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
    else:
        # 尝试找第一个 { 到最后 }
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last > first:
            candidate = text[first:last + 1]
        else:
            return None
    try:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def _build_structured_prompt(cfg, concept):
    """构建结构化输出的 prompt，要求 LLM 返回 JSON。"""
    from console.core import cfg_get
    obj_types = cfg_get(cfg, "gemma.object_types", []) or []
    issue_types = cfg_get(cfg, "gemma.issue_types", []) or []
    cond_types = cfg_get(cfg, "gemma.condition_types", []) or []
    pos_rules = cfg_get(cfg, "gemma.positive_rules", []) or []
    neg_rules = cfg_get(cfg, "gemma.negative_rules", []) or []
    unverif = cfg_get(cfg, "gemma.unverifiable_rules", []) or []

    parts = [
        f'You are reviewing a top-down aerial image for the concept: "{concept}".',
        '',
        'Determine the object_type, issue_type, condition, and whether this is verifiable from the image alone.',
        'You will see 3 images: raw_crop (clean target region, no annotations), crop (with mask outline), full (full tile with mask outline). Prioritize raw_crop for pixel details, full for context.',
        '',
    ]
    if obj_types:
        parts.append(f'Allowed object_types: {json.dumps(obj_types)}')
    if issue_types:
        parts.append(f'Allowed issue_types: {json.dumps(issue_types)}')
    if cond_types:
        parts.append(f'Allowed conditions: {json.dumps(cond_types)}')
    if pos_rules:
        parts.append(f'Positive rules (confirm target): {"; ".join(pos_rules)}')
    if neg_rules:
        parts.append(f'Negative rules (reject target): {"; ".join(neg_rules)}')
    if unverif:
        parts.append(f'Cannot be verified from image alone: {"; ".join(unverif)}')

    parts.extend([
        '',
        'CRITICAL: Only accept if you are SURE the target matches. Default to reject or needs_review when uncertain.',
        '',
        'Verifiable from image: object type, parking location, visible damage, missing safety gear, oil leaks.',
        'NOT verifiable from image: certification, inspection status, internal mechanical failure.',
        '',
        'Examples:',
        '- "raw_crop shows a yellow excavator with arm and bucket on emergency lane → accept, object_type=excavator, issue_type=illegal_parking, verifiable=true"',
        '- "raw_crop shows a regular white sedan on highway → reject, verifiable=true"',
        '- "mask covers only part of a loader, rest is cut off → hard_positive, object_type=loader, verifiable=true"',
        '',
        'Output ONLY this JSON:',
        '{',
        '  "label": "accept",',
        '  "semantic": {',
        '    "object_type": "from allowed list",',
        '    "issue_type": "from allowed list",',
        '    "condition": "from allowed list",',
        '    "confidence": 0.91,',
        '    "verifiable": true,',
        '    "evidence": "what you see in raw_crop"',
        '  }',
        '}',
    ])
    return "\n".join(parts)


def suggest_label(project_dir, cfg, item, labels):
    gcfg = cfg.get("gemma", {})
    if not gcfg.get("enabled"):
        return {"ok": False, "error": "Gemma 未启用 (config.gemma.enabled=false)"}
    allowed = gcfg.get("allowed_labels") or labels
    padding = float(cfg_get(cfg, "gemma.crop_padding", 0.7))
    timeout = int(cfg_get(cfg, "gemma.timeout", 120))
    structured = bool(cfg_get(cfg, "gemma.structured_output", False))
    try:
        views = compose_views(project_dir, cfg, item, padding=padding)
    except Exception as e:
        return {"ok": False, "error": f"合成图失败: {e}"}
    prompt = _build_prompt(project_dir, cfg, item, allowed, views)
    schema = (gcfg.get("schema") or "openai").lower()
    url = gcfg.get("url")
    model = gcfg.get("model")
    api_key = gcfg.get("api_key") or ""

    def _do(views_to_send):
        if schema == "gemini":
            return _call_gemini(url, api_key, prompt, views_to_send, timeout=timeout)
        return _call_openai(url, model, api_key, prompt, views_to_send, timeout=timeout)

    try:
        raw = _do(views)
    except urllib.error.HTTPError as e:
        if len(views) > 1:
            crop_only = [v for v in views if v[0] == "crop"] or [views[0]]
            try:
                raw = _do(crop_only)
            except Exception as e2:
                return {"ok": False, "error": f"LLM 失败 (多图 {e}; 单图 {e2})"}
        else:
            return {"ok": False, "error": f"LLM HTTP 错误: {e}"}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"网络/超时: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"LLM 调用失败: {e}"}

    # 结构化输出
    if structured:
        parsed = _parse_structured(raw, cfg)
        if parsed:
            lab = parsed.get("label", "")
            sem = parsed.get("semantic", {})
            if lab in allowed:
                return {
                    "ok": True, "label": lab, "raw": raw,
                    "semantic": sem, "gemma_raw": raw,
                }
            # 结构化解析成功但 label 不在允许列表，尝试旧解析
        # 结构化解析失败或 label 不在列表，回退旧解析
        lab = _parse_label(raw, allowed)
        if lab:
            return {"ok": True, "label": lab, "raw": raw}
        return {"ok": True, "label": None, "raw": raw,
                "error": "结构化输出无法解析为合法标签"}

    # 非结构化：旧逻辑
    lab = _parse_label(raw, allowed)
    if not lab:
        return {"ok": True, "label": None, "raw": raw,
                "error": "无法从回复中解析到合法标签"}
    return {"ok": True, "label": lab, "raw": raw}


def _repair_json(text):
    """修复小模型常犯的 JSON 错误：尾部逗号、缺失引号等。返回修复后的文本和是否修改过。"""
    if not text:
        return text, False
    fixed = text.strip()
    modified = False

    # 去掉首尾的 markdown fence（_parse_structured 里已处理，但再做一次保底）
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', fixed, re.DOTALL)
    if m:
        fixed = m.group(1).strip()
        modified = True

    # 找第一个 { 到最后一个 }
    first = fixed.find('{')
    last = fixed.rfind('}')
    if first != -1 and last > first:
        fixed = fixed[first:last + 1]
        if first > 0 or last < len(fixed) - 1:
            modified = True

    # 去掉数组/对象末尾的尾部逗号: ,]  ,}
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # 去掉键值对末尾的尾部逗号（JSON 不允许）
    # 实际上上面的正则已经处理了对象级别的 ,}
    # 再去掉单行末尾逗号和多余空格
    if re.search(r',\s*[}\]](?!\n)', fixed):
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        modified = True

    # 尝试修复缺失引号的键（小模型常见：{key: "value"} 而不是 {"key": "value"}）
    # 匹配模式: { ,或行首后跟 字母/下划线开头，后跟冒号
    fixed = re.sub(r'(?<=[{,\s\n])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', fixed)

    # 单引号改双引号（仅限值）
    # 太复杂可能引入其他问题，保守处理：只修键
    return fixed, modified


def _parse_json_robust(text, cfg):
    """多层回退解析 JSON，针对小模型优化。"""
    # 第一层：标准解析
    parsed = _parse_structured(text, cfg)
    if parsed:
        return parsed
    # 第二层：修复后重试
    repaired, ok = _repair_json(text)
    if ok:
        parsed = _parse_structured(repaired, cfg)
        if parsed:
            return parsed
    # 第三层：尝试用 ast.literal_eval（如果模型不小心用了 Python 字面量）
    try:
        import ast
        val = ast.literal_eval(repaired)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    return None


CONFIG_GEN_PROMPT = """\
You are setting up an aerial drone detection pipeline. Generate a JSON config based on the user's target domain.

JSON SCHEMA (output ONLY this JSON, no markdown):
{
  "batch_name": "lowercase_slug",
  "target_concept": "one sentence describing the target domain",
  "sam3_prompts": ["short phrase 1", "phrase 2", ...],
  "object_types": ["type_1", "type_2", "other_something", "non_target"],
  "issue_types": ["normal", "unknown", "domain_specific_1", ...],
  "positive_rules": ["rule sentence 1", "rule sentence 2", ...],
  "negative_rules": ["rule sentence 1", ...],
  "unverifiable_rules": ["cannot verify X from aerial image", ...],
  "exclude_classes": ["non_target"]
}

RULES:
- sam3_prompts: 8-10 diverse short English noun phrases (mix general + specific, vary wording)
- object_types: 5-10 specific types + "other_X" catchall + "non_target", all lowercase_underscore
- issue_types: include "normal" and "unknown" + 3-5 domain issues visible from top-down view
- positive_rules: 3-4 things you CAN see in an aerial image that confirm the target
- negative_rules: 3-4 things that REJECT the target (common confusions)
- unverifiable_rules: 3-4 things impossible to verify from drone imagery
- exclude_classes: object_types to skip in training (at least "non_target")

User domain description:"""


def generate_config(project_dir, cfg, domain_description):
    """调用 Gemma 根据用户描述生成 SAM3 prompts + Gemma review 配置。"""
    gcfg = cfg.get("gemma", {})
    if not gcfg.get("enabled"):
        return {"ok": False, "error": "Gemma 未启用 (config.gemma.enabled=false)"}

    prompt = CONFIG_GEN_PROMPT + "\n" + domain_description
    schema = (gcfg.get("schema") or "openai").lower()
    url = gcfg.get("url")
    model = gcfg.get("model")
    api_key = gcfg.get("api_key") or ""
    timeout = int(cfg_get(cfg, "gemma.timeout", 120))

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if schema == "gemini":
        full_url = url
        if api_key and "key=" not in url:
            sep = "&" if "?" in url else "?"
            full_url = f"{url}{sep}key={api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
        }
        req = urllib.request.Request(full_url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    else:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2048,
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        return {"ok": False, "error": f"LLM 调用失败: {e}"}

    if schema == "gemini":
        try:
            raw = resp["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return {"ok": False, "error": f"Gemini 解析失败: {json.dumps(resp)[:300]}"}
    else:
        try:
            raw = resp["choices"][0]["message"]["content"]
        except Exception:
            return {"ok": False, "error": f"OpenAI 解析失败: {json.dumps(resp)[:300]}"}

    # 多层回退解析 JSON
    parsed = _parse_json_robust(raw, cfg)
    if not parsed:
        return {"ok": False, "error": "LLM 返回无法解析为 JSON（已尝试修复尾部逗号/缺引号/引号混用）", "raw": raw[:500]}

    # 构建标准化返回 — 缺失字段给空值，不崩
    result = {
        "ok": True,
        "sam3": {
            "prompt_mode": "batch",
            "batch_name": parsed.get("batch_name", ""),
            "prompts": parsed.get("sam3_prompts", []) or [],
        },
        "gemma": {
            "target_concept": parsed.get("target_concept", ""),
            "object_types": parsed.get("object_types", []) or [],
            "issue_types": parsed.get("issue_types", []) or [],
            "positive_rules": parsed.get("positive_rules", []) or [],
            "negative_rules": parsed.get("negative_rules", []) or [],
            "unverifiable_rules": parsed.get("unverifiable_rules", []) or [],
        },
        "yolo_obb": {
            "exclude_classes": parsed.get("exclude_classes", []) or [],
        },
        "raw": raw,
    }
    return result
