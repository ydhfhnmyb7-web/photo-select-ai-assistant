# PhotoSelect AI Assistant

Windows 本地摄影审片助手，面向人像写真、婚纱、全家福、亲子、商务会议合影、活动纪实等真实摄影工作流。软件默认只复制导出文件，永远不删除原图、不覆盖原图。

## 功能说明

- 后台导入照片，主界面不阻塞。
- 支持 jpg、jpeg、png、webp。
- 缩略图缓存到所选照片文件夹下的 `cache/thumbnails`。
- 左侧缩略图使用 `QListView + Model/Delegate`，支持大量照片滚动浏览。
- 缩略图支持小 / 中 / 大尺寸，以及 `contain` / `cover` 显示模式。
- 单图预览支持适应窗口、100%、放大、缩小。
- 多图对比支持 2-4 张照片并排或 2x2 对比。
- 手动标记：精选、待定、废片、人工复核、可用待修、可用待裁切。
- SQLite 保存导入结果、手动分类、AI 结果、最终分类、备注、相似组和偏好学习字段。
- 本地 AI 支持三层判断：基础视觉质量、OpenCLIP/YOLO 语义与检测、用户偏好模型。
- 支持模型管理：模型状态、下载推荐模型、测试模型、检测 GPU、清理模型缓存、清理 embedding 缓存。
- 支持相似组查看，并在组内设置推荐保留。
- 支持筛片策略：保守、严格、交付、作品集、学习我的风格。
- 支持记录人工筛片偏好、训练轻量偏好模型、使用偏好模型重新评分。
- 导出前显示分类统计预览，确认后按中文多级分类复制到 `PhotoSelect_Output`。
- 导出 CSV 报告，包含 AI 判断原因、偏好模型原因、最终建议原因、语义分、检测信息和 embedding 缓存状态。

## 当前版本边界

- 不会默认下载大模型。必须用户点击“下载推荐模型”并确认后才会下载。
- 没有下载模型或依赖缺失时，软件会继续使用 OpenCV/fallback 轻量模式。
- OpenCLIP 用于语义分类和 embedding；YOLO11 用于人数、主体位置和主体完整性辅助判断。
- YOLO 检测只作为辅助，不一票否决。
- “年轻女生”“情侣”“母女”等只表示摄影业务归类和视觉外观估计，不代表真实年龄、性别或身份判断。
- “学习我的风格”是基于 OpenCLIP embedding 和本地特征的轻量偏好模型，不是训练或微调大型视觉模型。

## 安装基础版

基础版可完成导入、缩略图、手动审片、SQLite 保存和安全导出。

```powershell
cd photo_select_ai
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果系统中的 `python` 命令不可用，可以尝试：

```powershell
py -m pip install -r requirements.txt
py main.py
```

## 安装 GPU AI 依赖

`requirements.txt` 中的 torch、OpenCLIP、Ultralytics、scikit-learn 等为可选 AI 依赖。基础手动功能不强依赖 torch。

PyTorch CUDA 版请优先到官方安装页选择 Windows / Pip / Python / CUDA：

https://pytorch.org/get-started/locally/

Windows + CUDA 12.8 示例：

```powershell
.\.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install open_clip_torch
python -m pip install -U ultralytics
python -m pip install scikit-learn imagehash joblib scipy
```

测试 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果 `torch.cuda.is_available()` 是 `False`，通常说明当前 PyTorch 不是 CUDA 版，或 NVIDIA 驱动 / CUDA wheel / Python 环境不匹配。

## 模型管理

模型目录：

```text
models/
  openclip/
  yolo/
  cache/
cache/
  embeddings/
```

软件顶部提供：

- AI模型状态
- 下载推荐模型
- 测试模型
- 检测GPU
- 清理模型缓存
- 清理embedding缓存
- 模型档位：fast / balanced / accurate / ultra

点击“下载推荐模型”时会二次确认。下载过程在后台线程执行，不会卡住主界面。下载失败会提示“模型下载失败，请检查网络或代理”，详情写入 `logs/app.log`。

默认配置不会自动下载大模型：

```json
{
  "auto_download_models": false,
  "allow_large_model_download": true
}
```

## 模型档位

- `fast`：OpenCLIP ViT-B/32 + YOLO11n，速度优先，默认 batch_size 32。
- `balanced`：OpenCLIP ViT-L/14 + YOLO11m/配置模型，适合 RTX 4070 Ti，默认 batch_size 16。
- `accurate`：OpenCLIP ViT-H/14 + YOLO11x，精度优先，默认 batch_size 8-12。
- `ultra`：实验档，优先大模型，显存不足时自动降低 batch_size 或回退。

RTX 4070 Ti 推荐：

```json
{
  "use_gpu": true,
  "model_profile": "balanced",
  "batch_size": 16,
  "max_image_size": 1024,
  "use_fp16": true,
  "num_workers": 4,
  "prefetch": true,
  "pin_memory": true
}
```

运行稳定后可以尝试把 `batch_size` 调整到 `32`。如果显存不足，改为 `8` 或 `16`。

## 工作流与模型体检

新版界面按摄影师工作流分成 6 个板块：

- 项目与导入：选择照片文件夹、保存项目、恢复上次项目、查看工作流向导。
- 审片与对比：单图审片、多图对比、相似组查看，以及精选 / 待定 / 废片 / 可用待修 / 可用待裁切。
- AI分析：分析当前照片、批量 AI 分析、重新评分、查看分析统计。
- 我的筛片风格：记录人工偏好、训练轻量偏好模型、使用我的风格重新评分、查看风格报告。
- 模型与性能：模型状态、下载推荐模型、切换档位、检测 GPU、模型自测、自动调优、性能统计。
- 导出与报告：按分类复制、导出 CSV、导出偏好数据。

“工作流向导”会提示当前导入、AI 分析、人工样本、偏好模型训练、相似组复核和导出准备状态。样本不足时会明确提示还差多少张。

“模型自测”会检查 Python / torch / CUDA / GPU / batch_size / fp16、OpenCLIP 和 YOLO 加载状态，并抽取当前项目最多 10 张照片做小样本推理，输出 top3 分类和平均耗时。

“自动调优”会测试不同 batch_size，并根据显存、模型档位和平均耗时给出推荐配置，然后写入 `config.json`。GPU 占用率低不一定是错误，常见原因包括图片读取和解码在 CPU、batch_size 太小、embedding 缓存命中、模型过轻或过重、UI 刷新等待、磁盘读取慢等。优先目标是稳定、不卡、不爆显存和结果可复核，而不是强行占满 GPU。

RTX 4070 Ti 日常建议：

- 大量粗筛：`balanced`
- 正式精筛：优先尝试 `accurate`
- `ultra` 是实验模式，可能 batch_size 被压低，未必比 `accurate` 更适合当前任务

## 审美素材库与自我评估

软件不会自动爬取网络照片，也不会从搜索引擎、社交平台或未知来源批量下载含人脸图片。

外部素材只支持这些安全来源：

- 用户手动添加的本地素材文件夹。
- 用户自己拥有使用权的作品素材。
- 用户自行确认授权的数据集。
- 用户提供的 CSV / JSON 标注文件或 URL 列表，且需要自行确认数据权利。

“审美素材库”用于建立本地参考，不训练大模型。后续会优先用于 embedding 相似检索、偏好报告和模型自测基准集。当前版本支持添加素材文件夹、扫描素材、查看素材库统计；不会自动下载网络数据。

## 筛选原因与建议字段

每张照片会尽量生成以下解释字段，并写入右侧信息面板和 CSV：

- 筛选原因：说明为什么建议保留、可修、可裁切、相似待选、人工复核或废片疑似。
- 风格判断：例如棚拍写真、婚纱正式照、旅拍环境人像、商务形象照、生活感等。
- 修图建议：例如降低高光、提升面部亮度、校正色温、压暗背景杂物、皮肤轻修。
- 裁切建议：例如建议 4:5 竖版、16:9 横版、边缘穿帮可二次裁切。
- 作品/交付建议：说明是否适合客户交付、相似组备选、社交媒体候选或不建议进作品集。

AI 不确定时优先进入“人工复核 / 可用待修 / 可用待裁切 / 相似重复待选 / 废片疑似”，不再默认大量塞入“待定”。“待定”主要保留给人工操作。

## AI 识别逻辑

第一层：基础视觉质量分

- 清晰度
- 曝光
- 色彩 / 对比度
- 主体完整性
- 人脸 / 主体可见性

第二层：大模型语义理解分

- OpenCLIP 批量生成 image embedding。
- 摄影业务分类使用多个 prompt，不只依赖单 prompt。
- YOLO11 检测人物数量、主体框、主体面积、边缘裁切风险和多人合影倾向。

第三层：用户偏好分

- 用户人工标记会记录为偏好样本。
- 样本达到 50 张后，可训练本地轻量偏好模型。
- 优先使用 Logistic Regression，依赖不足时回退轻量 centroid/KNN 风格模型。
- 偏好模型只给建议，不覆盖人工分类。

## embedding 缓存

embedding 缓存位于：

```text
cache/embeddings/
```

缓存 key 包括：

- 文件路径
- 文件大小
- 修改时间
- 模型名称
- 模型预训练权重

同一批照片第二次分析时，如果文件和模型未变化，会直接读取 embedding 缓存。更换模型后会重新生成。缓存损坏会自动重算并写入日志。

## 保存机制

- `Ctrl + S` 或顶部“保存”按钮会保存当前项目。
- 默认每 60 秒自动保存一次。
- 修改手动分类、备注、AI 结果、最终分类、推荐保留、配置项后会进入未保存状态。
- 关闭窗口时，如果有未保存内容，会询问“保存并退出 / 不保存退出 / 取消”。
- 如果正在导入、AI 分析、导出、模型下载或训练偏好模型，关闭窗口会询问“等待完成 / 停止任务并退出 / 取消”。
- 异常退出后，下次启动可提示恢复项目。

## 快捷键

- `1`：标记精选
- `2`：标记待定
- `3`：标记废片
- `A`：对当前照片执行 AI 分析
- `Ctrl + Shift + A`：批量 AI 分析
- `Ctrl + A`：当前列表全选
- `←` / `→`：上一张 / 下一张
- `Space`：回到单图聚焦
- `Ctrl + S`：保存当前结果
- `Ctrl + E`：按分类导出
- `Tab`：对比模式切换聚焦图
- `Enter`：当前聚焦图设为推荐保留
- `Esc`：退出对比模式或取消当前操作

快捷键不会在备注输入框中触发。

## 运行

```powershell
cd photo_select_ai
.\.venv\Scripts\activate
python main.py
```

## 打包 exe

```powershell
cd photo_select_ai
.\.venv\Scripts\activate
pyinstaller --onefile --windowed main.py
```

如果把 PyTorch / OpenCLIP / YOLO 一起打包，体积会很大，建议先确认开发环境运行稳定，再做完整打包。

## 常见报错

### No module named cv2

```powershell
python -m pip install opencv-python
```

缺少 `cv2` 时，主界面仍应打开，AI 自动分析会提示安装依赖。

### No module named torch

手动审片不需要 torch。需要 GPU AI 时安装 CUDA 版 PyTorch：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### No module named open_clip

```powershell
python -m pip install open_clip_torch
```

### No module named ultralytics

```powershell
python -m pip install -U ultralytics
```

### CUDA 不可用

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

CUDA 不可用时软件会切换到 CPU / fallback 模式。

### PySide6 未安装

```powershell
python -m pip install PySide6
```

### python 不是内部或外部命令

安装 Python 时勾选 `Add python.exe to PATH`，或使用 Python Launcher：

```powershell
py main.py
```

## 注意事项

- 原图不会被删除。
- AI 分类只是辅助，人工判断优先。
- 低置信度照片会进入人工复核。
- 可修复问题不等于废片，背景穿帮、轻微曝光偏差、可裁切问题会优先归入可用待修或可用待裁切。
- 只有严重失焦、主体严重糊、严重曝光不可恢复、文件损坏等情况才会进入明确废片。
- 相似照片只分组、排名和建议，不会自动删除。
- 偏好模型不会覆盖人工分类，只提供建议。
- 不建议默认下载或训练超大模型；先用预训练模型批量推理和 embedding 缓存提升识别质量。
