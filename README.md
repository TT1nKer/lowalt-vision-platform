# 低空遥感智能分析平台

面向低空遥感（正射影像、无人机图片与视频）的通用视觉智能分析框架，包含两套可独立运行的 Web 系统：

| 端口 | 系统 | 职责 |
|---|---|---|
| 7860 | 视觉任务控制台 | 通用视觉模型蒸馏流水线：定义目标 → SAM3 发现候选 → Gemma 语义筛选 → 导出分割/旋转框数据 → 训练轻量 YOLO 模型。 |
| 7861 | 低空遥感智能分析平台 | 全域正射影像地图 + 候选图层展示 + DJI 图片/视频导入（EXIF/SRT 定位、视频抽帧）与 SegFormer 停车分析 + 技术工作台。 |

## 能力目录

- 停车设施候选（内置示例能力）：SegFormer-large-parking 推理 → 证据分级（车排支持 / 检测到车辆 / 仅模型候选）→ 地图交互查看。
- DJI 素材导入：图片保留 EXIF GPS/时间/相机；视频按步长抽帧并用同名 SRT 定位；导入资产以点位图层回到地图。
- 分析服务按"独立、可测试的服务 + 能力目录注册"扩展；没有模型或数据时显示"待接入"，不生成演示假数据。

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
uav_data/                 无人机数据 schema 与适配器
web/                      控制台前端（原生 HTML/CSS/JS）
tests/                    全量测试（unittest discover）
scripts/                  启动与运维辅助脚本
profiles/                 任务配置示例
config.example.yaml       示例配置（复制为 config.yaml 后按需修改）
```

## 运行

```bash
pip install -r requirements.txt
python platform_app.py --project-dir . --port 7861      # 低空遥感平台
python app.py --project-dir . --config config.yaml --port 7860   # 视觉任务控制台
```

Windows 下可直接双击 `start_platform.cmd` / `start_console.cmd`。

## 数据与模型（不在本仓库）

仓库只包含代码、测试与示例配置。运行前需要自行准备：

- 正射影像目录（默认 `imagery/`，可在 `config.yaml` 的 `paths.merged_dir` 与 `lowalt_platform/settings.py` 中修改）；
- 模型权重：SegFormer-large-parking（`models/segformer_large_parking/best_model.ckpt`）、轻量 YOLO 权重；SAM3 与 Gemma 服务地址在 `config.yaml` 中配置。

## 测试

```bash
python -m unittest discover -s tests -t .
```

模型推理相关测试需要可选依赖 `torch` / `transformers`；未安装时这些测试自动跳过（skip），其余全部运行。

## 设计原则

- 所有识别结果为"模型候选"，除非经过人工验收，不冒充真值；
- 质量门禁默认开启：无固定测试集清单与金标签署时拒绝正式训练；
- 数据导入保留时间、GPS/RTK、姿态元数据，未知值明确标注，不猜测填充。
