# TinyTL 小样本：复现与压力测试

JHU 课程项目 *Reproducing and Stress-Testing TinyTL*（Chen Chen、Tao Wang、Yujia Zhang）。

在 TinyTL（ProxylessNAS-Mobile，lite residual + bias，冻结权重 8-bit 量化）上做小样本迁移：每类只有 k 张训练图时，反传要到多深、batch 取多大、分类头怎么初始化，才能在给定内存下得到最好的精度。数据集是 Flowers102 和 FGVC-Aircraft。

## 从哪里看起

| 文件 | 内容 |
|---|---|
| [`实验设计/实验大纲.md`](实验设计/实验大纲.md) | 全部实验的大纲：目的、内容、结果、配置建议、下一步 |
| [`midterm/`](midterm/) | 期中报告的 LaTeX、参考文献，图 1、图 4 及其画图脚本 `make_figures.py`。图 2、图 3 的 PDF 不在仓库里 |
| [`实验设计/最终实验设计.md`](实验设计/最终实验设计.md) | k=5 主实验（截断深度 × 数据集）的预先登记设计 |
| [`实验设计/补充实验设计.md`](实验设计/补充实验设计.md) | 补充实验：k=2/10、imprinting、学习率稳健性 |
| [`实验设计/运行说明.md`](实验设计/运行说明.md) | 每个脚本的作用、截断方式（arm）的定义、运行命令 |
| [`实验设计/分析输出_全部.txt`](实验设计/分析输出_全部.txt) | k=5 主实验的分析结果 |
| [`实验设计/分析输出_补充.txt`](实验设计/分析输出_补充.txt) | 补充实验的分析结果 |
| `runs/<运行名>/result.json` | 每次运行的结果：test/val/train 精度、损失曲线、显存、各种哈希 |
| [`README_开始复现.md`](README_开始复现.md)、[`复现结果-主方法.md`](复现结果-主方法.md) | 最早的全量 Flowers102 复现（2026-09） |
| [`实验报告-分辨率分配.md`](实验报告-分辨率分配.md) | 之前的分辨率分配实验 |

## 环境

Windows + Git Bash，RTX 3060 12GB，Python 3.12，PyTorch 2.5.1+cu124。依赖固定在 `requirements-lock.txt`。

1. 官方代码不在本仓库里，克隆到 `upstream/` 并打上兼容补丁：

   ```bash
   git clone https://github.com/mit-han-lab/tinyml.git upstream
   git -C upstream checkout 0e58f314dfe12967d6165f45238b1ea77b0f7a14
   git -C upstream submodule update --init once-for-all
   git -C upstream apply --ignore-whitespace ../tinytl-compat.patch
   git -C upstream/once-for-all apply --ignore-whitespace ../../ofa-compat.patch
   ```

   补丁只做兼容修改（新版 scikit-learn、torchvision、Windows），详见 `README_开始复现.md`。

2. 运行 `install-environment.cmd`，在 `.venv/` 里装好依赖。
3. 按下一节下载两个数据集和预训练权重。

## 数据集

两个数据集都不在仓库里，要从官方地址下载，使用时请遵守官网的条款。下载和处理完共占约 8 GB 硬盘空间。命令都在仓库根目录的 Git Bash 里运行。

每次运行需要的 k-shot 子集（`fewshot/`、`fewshot_aircraft/`）会在运行时自动生成，不用手动准备。

### Flowers102

`prepare.py` 会下载数据集、检查 MD5、整理目录，再下载预训练权重：

```bash
.venv/Scripts/python.exe prepare.py
```

自动下载失败时，手动下载下面三个文件放到 `data/flowers102_raw/`，再运行一次 `prepare.py`。已经存在的文件会跳过。

| 文件 | 字节数 |
|---|---|
| [`102flowers.tgz`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) | 344,862,509 |
| [`imagelabels.mat`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) | 502 |
| [`setid.mat`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat) | 14,989 |

处理完 `data/flowers102/train/` 有 2,040 张图，是官方的 train 加 val；`data/flowers102/val/` 有 6,149 张图，是官方的 test。

### FGVC-Aircraft

这个数据集没有自动下载，用下面的命令：

```bash
mkdir -p data/aircraft_raw
curl -L -C - -o data/aircraft_raw/fgvc-aircraft-2013b.tar.gz https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
sha256sum data/aircraft_raw/fgvc-aircraft-2013b.tar.gz
.venv/Scripts/python.exe prepare_aircraft.py
```

压缩包有 2,753,340,328 字节，SHA-256 应该是 `e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834`。下载中断时重新运行 `curl` 那一行，会接着下载。

`prepare_aircraft.py` 会解压、裁掉每张图底部 20 像素的版权条，再把短边缩放到 512。处理完 `data/aircraft_prep/` 有 10,000 张图，划分记录在 `splits.json` 里。

## 模型文件

模型文件放在 [Release weights-v1](https://github.com/DoubleV6317/tinytl-fewshot/releases/tag/weights-v1) 里，不在 Git 历史里。每个压缩包都在仓库根目录解压。

| 文件 | 内容 | 什么时候需要 |
|---|---|---|
| `pretrained-cache.tar.gz` | 预训练权重、网络配置和 8-bit 量化缓存，解压到 `cache/` | 官方地址下载不了，或者想省去第一次量化的时间 |
| `trained-flowers102.tar.gz`、`trained-aircraft.tar.gz` | 每次运行最后一轮的可训练参数，解压到 `runs/<运行名>/final_trainable.pth` | 想看训练好的参数，不重新训练 |
| `trained-earlier.tar.gz` | 2026-09 的全量复现、分辨率分配实验和协议桥用到的完整 checkpoint，解压到 `runs/` | 运行 `eval_final_ckpt.py`，或查看早期结果 |

`prepare.py` 默认从 MIT Han Lab 的官方地址下载预训练权重。想用这里的镜像，就在运行 `prepare.py` 之前解压，文件已经存在时它不会再下载：

```bash
curl -L -O https://github.com/DoubleV6317/tinytl-fewshot/releases/download/weights-v1/pretrained-cache.tar.gz
tar -xzf pretrained-cache.tar.gz
```

重新训练不需要 `trained-*` 这几个包，分析脚本也不需要，它们只读 `result.json`。`final_trainable.pth` 只保存可训练参数，要还原完整网络还需要 `pretrained-cache.tar.gz`。

## 运行

```bash
bash run-flowers.sh    # Flowers102，k=5 主实验
bash run-aircraft.sh   # Aircraft，k=5 主实验
bash run-supp.sh       # 补充实验
```

已完成的运行会跳过，中断后重新执行同一条命令即可接着跑。分析：

```bash
.venv/Scripts/python.exe analyze_depth.py
.venv/Scripts/python.exe analyze_supp.py
```

## 不在仓库里

数据集、每次运行的控制台日志（`logs/`、`runs/*/logs/`）、`.venv/`。预训练权重和训练得到的权重（`*.pth`）在上面的 Release 里，不在 Git 历史里。每次运行只保留 `result.json`，分析脚本只读这些文件。
