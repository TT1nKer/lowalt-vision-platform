# 使用说明

两套可独立运行的 Web 系统：**低空遥感智能分析平台**（7861，地图与数据导入）与**视觉任务控制台**（7860，模型蒸馏流水线）。

## 一、环境准备

```bash
pip install -r requirements.txt
# 模型推理相关测试需要额外的可选依赖：
# pip install torch transformers opencv-python
```

数据与模型不在仓库内，运行前准备：

1. 影像目录（正射影像瓦片，默认 `imagery/`，可在 `config.yaml` 的 `paths.merged_dir` 修改）；
2. `config.yaml`：从 `config.example.yaml` 复制，把 `sam3.api_url`、`gemma.url` 换成实际服务地址；
3. 模型权重（`models/` 下，见 README）：
   - `segformer_large_parking/best_model.ckpt` + `nvidia_mit_b5_config/`（停车分割）
   - 轻量 YOLO 预训练权重（训练起点）

## 二、启动

```bash
python platform_app.py --project-dir . --port 7861        # 低空遥感平台
python app.py --project-dir . --config config.yaml --port 7860   # 视觉任务控制台
```

Windows 下可直接双击 `start_platform.cmd` / `start_console.cmd`（自带"已运行则不重复启动"检查）。两个服务都只监听本机（127.0.0.1）。

## 三、7861 平台

### 地图与候选

1. 打开 `http://127.0.0.1:7861`；
2. 左侧按证据等级勾选候选图层（车排支持 / 检测到车辆 / 仅模型候选）；
3. 点击候选查看原图、模型掩膜与二次分析证据，可放大查看；
4. 地图放大到 16 级自动加载原始瓦片。

### DJI 图片/视频导入

1. `http://127.0.0.1:7861/engineering` → 数据导入区；
2. 填源目录（允许范围见页面提示；改 `lowalt_platform/settings.py` 的 `allowed_source_roots` 可扩展）与抽帧间隔；
3. 扫描 → 导入：图片保留 EXIF GPS/时间/相机；视频抽帧并用同名 `.SRT` 定位（精确或插值，均标注 `gps_source`）；产物在 `dji_imports/run-*/`；
4. 任务行点"运行停车分析"：SegFormer（GPU auto）逐张分割，掩膜写入 `analysis/`；
5. 回地图页，DJI 导入资产以点位图层展示，点击可看原图/掩膜。

## 四、7860 控制台

通用视觉模型蒸馏流水线（对任意识别目标通用）：

```
01 定义目标 → 02 SAM3 发现 → 03 Gemma 筛选 → 04 生成训练数据 → 05 训练轻量模型
```

1. **换目标**：任务控制台 → "配置识别目标" → 一句话描述目标 → 生成任务配置（Gemma 自动写 SAM3 提示词与判断规则）→ 应用。
2. **发现**：运行 SAM3（支持断点续跑、坐标换算、多提示词合并）。
3. **筛选**：启动 Gemma 批量判断；审核页人工抽查（数字键打标、←→ 翻页、Z 撤销）。
4. **导出**：生成分割/旋转框数据；只导出审核明确的样本，原子替换旧数据集；可跑数据质量检查。
5. **训练**：选基础模型、尺寸、轮数，开始/续训；模型验证页做基线 vs 当前模型的公平对比（P/R/mAP、对比图）。

### 质量门禁

默认开启：无固定测试集清单与金标签署时拒绝正式训练（这是保护，不是故障）。解锁需要：人工审核 ≥5% → 固定测试集清单 → 金标签署清单。

## 五、数据与产物

| 内容 | 位置 |
|---|---|
| 影像 | `imagery/`（或 config 指定的目录） |
| 任务产物 | `sam3_runs/<任务名>/`（候选、审核、导出数据集） |
| DJI 导入 | `dji_imports/run-*/` |
| 模型权重 | `models/` |
| 停车候选权威输入 | `quality/parking_direct_first_layers/` |
| 日志 | `pipeline.log` |

## 六、测试

```bash
python -m unittest discover -s tests -t .
```

`test_parkseg12k_infer`、`test_parkseg12k_finetune` 需要 `torch`/`transformers`，未安装时这两个模块跳过，其余全部运行。

## 七、口径

所有识别结果为**模型候选**，未经人工金标验收不构成正式结论；数据导入保留时间与 GPS 元数据，未知值明确标注，不猜测填充。
