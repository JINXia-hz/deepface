# deepface

基于 PyTorch 的灰度人脸检测项目。默认使用轻量级 ResUNet 生成逐像素分数图，同时保留 36×36 CNN 分类器及其全卷积热图检测流程，便于训练和对比。

项目以图像金字塔覆盖不同尺度的人脸，再经过峰值提取、非极大值抑制（NMS）、尺度细化和框形校准得到最终结果。模型无需额外检测框架，CPU 即可运行，训练脚本也支持 CUDA。

## 目录结构

```text
├── deepface/              # 核心包
│   ├── models.py          # Net、ResUNet 和模型加载
│   ├── detect.py          # 图像金字塔、候选提取、NMS 与检测入口
│   ├── data.py            # 36×36 人脸/非人脸数据加载
│   ├── wider.py           # WIDER FACE 标注解析与训练采样
│   └── evaluate.py        # 分类指标、召回率与 WIDER AP
├── tools/                 # 从项目根目录运行的命令行脚本
├── models/                # 已训练权重
├── crops.tar.gz           # 裁剪数据集，使用 Git LFS 管理
├── colab_train.ipynb      # Colab 训练笔记本
├── data/                  # 本地场景数据，不纳入 Git
├── train_images/          # 本地训练裁剪图，不纳入 Git
├── test_images/           # 本地测试裁剪图，不纳入 Git
└── outputs/               # 检测结果与评估缓存，不纳入 Git
```

核心逻辑位于 `deepface/`，可执行脚本位于 `tools/`。以下命令均应在项目根目录执行；Windows 使用 `py`，其他环境可将命令中的 `py` 换成 `python`。

## 安装

建议使用 Python 3.12，并在虚拟环境中安装依赖：

```bash
py -m venv .venv
.venv/Scripts/activate
py -m pip install -r requirements.txt
```

Linux/macOS 的虚拟环境激活命令为 `source .venv/bin/activate`。

## 快速检测

```bash
py -m tools.detect <图片路径> [模型路径] [置信度阈值]
```

例如：

```bash
py -m tools.detect path/to/image.jpg
py -m tools.detect path/to/image.jpg models/model_deep.pth 0.85
py -m tools.detect path/to/image.jpg models/model_fcn2.pth 0.995
```

不传模型时默认使用 `models/model_deep.pth`，不传阈值时默认使用 `0.85`；CNN 热图后端的默认阈值为 `0.995`。脚本会根据权重自动识别模型结构，无需额外指定后端。

结果保存为 `outputs/<图片名>_<人脸数>faces.png`，输出目录会自动创建。

## 数据准备

### 裁剪数据集

CNN 训练、分类测试以及 ResUNet 的裁剪图辅助训练需要 `train_images/` 和 `test_images/`。它们已打包在 `crops.tar.gz` 中：

```bash
git lfs pull
tar -xzf crops.tar.gz
```

解压后的目录按 ImageFolder 格式组织：`0/` 为非人脸，`1/` 为人脸。如果归档只是很小的文本文件，通常是尚未安装或拉取 Git LFS 内容。

### 场景数据集

场景级训练和评估还需要自行准备以下数据，这些目录不会提交到 Git：

```text
data/
├── widerface/
│   ├── WIDER_train/images/
│   ├── WIDER_val/images/
│   ├── wider_face_split/
│   │   ├── wider_face_train_bbx_gt.txt
│   │   └── wider_face_val_bbx_gt.txt
│   └── eval_tools/ground_truth/  # 官方 WIDER 验证集评估标注
├── no_person/                    # 无人脸负样本场景
└── holdout/                      # 综合评估使用的无人脸留出集
```

`samples/` 是本地示例图片目录，同样不纳入 Git。只有运行对应的场景训练或评估脚本时才需要这些数据；单张图片检测和裁剪数据集训练不需要它们。

## 模型权重

| 文件 | 用途 |
| --- | --- |
| `models/model_deep.pth` | 默认 ResUNet 检测模型 |
| `models/model_deep_v1.pth` | 较早的 ResUNet 基线 |
| `models/model_fcn.pth`、`models/model_fcn2.pth` | 在场景数据上微调的 CNN 热图模型 |
| `models/model_noeq.pth` | CNN 基线，也是 CNN 训练脚本的默认输出 |
| `models/model.pth` | 旧版 CNN 权重，仅用于兼容和对比 |

`deepface.models.load_model()` 会读取 state dict 并返回 `(model, backend)`，其中 `backend` 为 `deep` 或 `heatmap`。

## 训练

### 训练 CNN 分类器

准备好裁剪数据集后运行：

```bash
py -m tools.train_cnn 10
```

参数为训练轮数。训练完成后会覆盖保存 `models/model_noeq.pth`。

### 训练 ResUNet

准备好裁剪数据集、WIDER FACE 和无人脸场景后运行：

```bash
py -m tools.train_deep 2000 1.0 models/model_deep_new.pth 0
```

完整参数顺序如下：

```text
py -m tools.train_deep [steps] [lambda] [out] [seed] [init] [tiles] [lr] [device] [batch]
```

- `steps`：训练步数，默认 `2000`。
- `lambda`：场景损失权重，默认 `1.0`。
- `out`：输出权重路径，默认 `models/model_deep.pth`。
- `seed`：随机种子，默认 `0`。
- `init`：可选的 `.pth` 权重或 `.ckpt` 检查点，用于热启动或恢复训练。
- `tiles`：场景图块尺寸，默认 `192`；可用逗号指定多个尺寸。
- `lr`：学习率，默认 `1e-3`。
- `device`：`auto`、`cuda` 或 `cpu`，默认 `auto`。
- `batch`：每步的场景图块数和负样本图块数，默认 `10,4`。

训练每 500 步保存一次 `<out>.ckpt`，结束时保存模型权重和最终检查点。长时间 GPU 训练也可以参考 `colab_train.ipynb`。

### 微调 CNN 热图模型

```bash
py -m tools.finetune_fcn [steps] [lambda] [out] [seed]
```

该脚本从 `models/model_noeq.pth` 开始，以全卷积方式在 WIDER FACE 场景上微调，默认输出 `models/model_fcn2.pth`。

## 评估

```bash
py -m tools.test_cnn                         # CNN 裁剪图测试集指标
py -m tools.export_mistakes                  # 导出 CNN 误判样本到 outputs/
py -m tools.eval_fcn models/model_deep.pth   # 裁剪图、留出集、示例图与 WIDER 召回率
```

`eval_fcn` 会遍历完整场景评估数据，运行前需准备 `data/holdout/`、WIDER FACE 和本地 `samples/`。

官方 WIDER FACE 验证集 AP 分为候选扫描和指标计算两步：

```bash
py -m tools.eval_wider scan --model models/model_deep.pth --workers 7
py -m tools.eval_wider ap
```

扫描结果缓存到 `outputs/cache/`，中断后可继续，后续调整 NMS 或阈值时无需重新运行模型。可通过 `--sample 4` 做按场景分层的小规模评估；完整评估会处理全部 3226 张验证图片。

## 开发检查

```bash
py -m pip install -r requirements-dev.txt
py -m ruff check deepface tools
py -m mypy deepface tools
```

GitHub Actions 会对推送和 Pull Request 执行相同的 Ruff 与 mypy 检查。
