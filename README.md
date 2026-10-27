# TinyTL 小样本：复现与压力测试

JHU 课程项目 *Reproducing and Stress-Testing TinyTL*（Chen Chen、Tao Wang、Yujia Zhang）。

在 TinyTL（ProxylessNAS-Mobile，lite residual + bias，冻结权重 8-bit 量化）上做小样本迁移：每类只有 k 张训练图时，反传要到多深、batch 取多大、分类头怎么初始化，才能在给定内存下得到最好的精度。数据集是 Flowers102 和 FGVC-Aircraft。

## 从哪里看起

| 文件 | 内容 |
|---|---|
| [`实验设计/最终实验设计.md`](实验设计/最终实验设计.md) | k=5 主实验（截断深度 × 数据集）的预先登记设计 |
| [`实验设计/补充实验设计.md`](实验设计/补充实验设计.md) | 补充实验：k=2/10、imprinting、学习率稳健性 |
| [`实验设计/运行说明.md`](实验设计/运行说明.md) | 每个脚本的作用、截断方式（arm）的定义、运行命令 |
| [`实验设计/分析输出_全部.txt`](实验设计/分析输出_全部.txt) | k=5 主实验的分析结果 |
| `实验设计/分析输出_补充.txt` | 补充实验的分析结果（补充实验跑完后由 `run-supp.sh` 生成） |
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
3. 运行 `.venv/Scripts/python.exe prepare.py`，下载 Flowers102 和预训练权重。
4. Aircraft：把 `fgvc-aircraft-2013b.tar.gz` 放到 `data/aircraft_raw/`，再运行 `.venv/Scripts/python.exe prepare_aircraft.py`。

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

数据集、预训练权重、训练得到的权重（`*.pth`）、每次运行的控制台日志（`logs/`、`runs/*/logs/`）、`.venv/`。每次运行只保留 `result.json`，分析脚本只读这些文件。
