# 低空遥感智能分析平台

[English](README.md) | **简体中文**

公开 CPU 验收场景已收敛为：**多段素材导入 → GPS/时间溯源 → 地图点位 → 完整性校验的证据 ZIP**。使用明确标记的合成素材，无需实习数据、模型权重或远端服务。格式与验证边界见[验证记录](docs/public-import-validation.md)。

面向低空遥感（正射影像、无人机图片与视频）的通用视觉智能分析框架，包含两套可独立运行的 Web 系统：

| 端口 | 系统 | 职责 |
|---|---|---|
| 7860 | 视觉任务控制台 | 通用视觉模型蒸馏流水线：定义目标 → SAM3 发现候选 → Gemma 语义筛选 → 导出分割/旋转框数据 → 训练轻量 YOLO 模型。 |
| 7861 | 低空遥感智能分析平台 | 全域正射影像地图 + 候选图层展示 + DJI 图片/视频导入（EXIF/SRT 定位、视频抽帧）与 SegFormer 停车分析 + 技术工作台。 |

## 能力目录

- 停车设施候选（内置示例能力）：SegFormer-large-parking 推理 → 证据分级（车排支持 / 检测到车辆 / 仅模型候选）→ 地图交互查看。
- DJI 素材导入：图片保留 EXIF GPS/时间/相机；视频按步长抽帧并用同名 SRT 定位；导入资产以点位图层回到地图。
- 分析服务按“独立、可测试的服务 + 能力目录注册”扩展；没有模型或数据时显示“待接入”，不生成演示假数据。

## 目录

```text
app.py                    视觉任务控制台后端入口（7860）
platform_app.py           低空遥感平台入口（7861）
console/                  控制台后端：核心配置、SAM3/Gemma 流水线、数据质量、发布门禁
lowalt_platform/          平台后端：领域模型、服务、API、前端
parking/                  停车设施候选与分析（内置示例能力）
parking_map/              停车地图构建研究代码
training/                 模型训练与实验脚本（SegFormer 微调/推理、YOLO、拓扑分类）
audits/                   一次性数据审计脚本
web/                      控制台前端（原生 HTML/CSS/JS）
tests/                    全量测试（unittest discover）
scripts/                  启动与运维辅助脚本
profiles/                 任务配置示例
config.example.yaml       示例配置（复制为 config.yaml 后按需修改）
```

## 运行

公开轻量路径支持 Python **3.11 / 3.12**：

```bash
git clone https://github.com/TT1nKer/lowalt-vision-platform.git
cd lowalt-vision-platform
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-core.lock
python -m scripts.check_core
python -m scripts.reproduce_import --output runs/public-import
python platform_app.py --project-dir runs/public-import --port 7861
```

打开[地图](http://127.0.0.1:7861/)或[工程控制台](http://127.0.0.1:7861/engineering)。固定样例生成 11 个资产、7 个定位点，另有 4 个资产刻意保持无定位；不生成停车候选，不执行模型推理。无正射影像时地图使用空背景，并缩放到导入的相机点位。工程控制台提供逐任务地图链接与证据包下载。

产物为 `runs/public-import/evidence.zip`、`assets.geojson`、`verification.json`。重复验收时换一个输出目录，脚本不会覆盖旧产物。也可通过 `python platform_app.py --project-dir . --port 7861` 空数据启动；只允许导入指定项目数据目录及其子目录，不再默认放开整个 Windows 盘符。

`requirements-core.txt` 为轻量直接依赖，`requirements-core.lock` 固定 CI 实测解析版本。原 `requirements.txt` 是包含大型依赖的可选模型/研究环境，此样例不需要。需要运行旧任务控制台时，在单独环境安装它，复制 `config.example.yaml` 为 `config.yaml` 并配置数据路径与服务，再运行：

```bash
python app.py --project-dir . --config config.yaml --port 7860
```

Windows 的 `start_platform.cmd` / `start_console.cmd` 仍可用于已配置的模型环境。

## 数据与模型（不在本仓库）

仓库只包含代码、测试与示例配置。运行前需要自行准备：

- 正射影像目录（默认 `imagery/`，可在 `config.yaml` 的 `paths.merged_dir` 与 `lowalt_platform/settings.py` 中修改）；
- 模型权重：SegFormer-large-parking（`models/segformer_large_parking/best_model.ckpt`）、轻量 YOLO 权重；SAM3 与 Gemma 服务地址在 `config.yaml` 中配置。

## 测试

```bash
python -m scripts.check_core
```

公开 CPU 范围运行 189 项测试：**182 通过、7 跳过**（未装 `torch` / `transformers`）。另外 8 个历史测试模块因公开仓库缺少 `uav_data` / `equipment_data` 源码而明确排除，运行时逐项列出；未把它们算作通过或已修复。直接执行 `python -m unittest discover -s tests -t .` 仍会暴露这 8 个导入错误，详见[验证记录](docs/public-import-validation.md)。

[GitHub Actions](https://github.com/TT1nKer/lowalt-vision-platform/actions/workflows/core.yml) 在 Python 3.11 / 3.12 上运行轻量测试和独立验收，并保留合成证据产物。可选模型测试、实际推理和准确率均不在本次验证范围。

## 设计原则

- 所有识别结果为“模型候选”，除非经过人工验收，不冒充真值；
- 质量门禁默认开启：无固定测试集清单、金标准标签与审核来源时拒绝正式训练；
- 数据导入保留时间、GPS/RTK、姿态元数据，未知值明确标注，不猜测填充。
