# TinyTL 本地复现环境

位置：本文件所在的 TinyTL 文件夹。显卡已确认是 RTX 3060 12GB。

**当前计划更新（2026-09-10）：用户已改为直接训练 200 轮。`run.py` 默认训练轮数与 `train-flowers.cmd` 均已改为 200。当前后台任务见 `direct-200.json`，实时日志为 `direct-200.stdout.log` 和 `direct-200.stderr.log`。旧的“50 轮后再启动 200 轮”脚本已禁用；当前任务运行时不要再次双击训练脚本。下文提到的 50 轮是原始官方配置参考。**

部署验证已通过：2026-09-10，真实 Flowers102 小样本完成了 8-bit 聚类量化、2 个训练 batch 的反向传播与优化、评估和 checkpoint 保存。该验证全流程的 CUDA 峰值分配约 703 MiB；完整训练的评估 batch 更大，不应直接用此数字预测完整运行显存。详见 `verification.json` 和 `smoke-test.log`。

## 开始使用

1. 双击 `smoke-test.cmd`，用 16 张训练图和 16 张测试图验证训练、评估和保存流程。它不产生论文复现准确率。
2. 在没有训练任务运行时，双击 `train-flowers.cmd`，启动完整 Flowers102 的 200 轮 TinyTL 实验。每次运行创建独立目录，结果保存在 `runs/`。
3. 关闭训练窗口或按 Ctrl+C 会中断训练。`train-flowers.cmd` 每次都从预训练模型重新开始。
4. 训练被中断后（崩溃、睡眠、重启），双击 `resume-latest.cmd` 续训：它找到 `runs/` 下最近更新且带 checkpoint 的运行，从该 checkpoint 继续，不会新建目录。
   续训时上游会用时钟重新设种子并跳过重新量化，因此续训得到的曲线与不中断的运行不是逐位一致的，写报告时需注明中断与续训。

## 长时间训练前必须先关掉睡眠

2026-09-10 的 200 轮任务在第 47 轮（19:38:00）中断，原因不是代码错误：系统事件日志显示机器进入睡眠，随后在 19:40 重启。完整 200 轮约需 3.5 小时，开始前先执行：

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 15
```

训练结束后可按需改回（例如 `standby-timeout-ac 30`）。

也可在此文件夹打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe run.py --mode smoke
.\.venv\Scripts\python.exe run.py --mode train --epochs 50
.\.venv\Scripts\python.exe run.py --mode train --epochs 200 --resume-from runs\<运行目录名>
```

本机无需激活环境，也无需单独安装 CUDA Toolkit；启动脚本使用项目内的 GPU 版 PyTorch。

## 配置与复现边界

- 官方 TinyML 仓库 commit：`0e58f314dfe12967d6165f45238b1ea77b0f7a14`。
- 官方 once-for-all 依赖固定为原仓库指定 commit：`cfa0722d57e3a2391eb36b8cf613dd17ff7a32ae`。
- Python 3.12.10；PyTorch 2.5.1+cu124；torchvision 0.20.1+cu124。全部依赖见 `requirements-lock.txt`。
- 默认方法：TinyTL lite residual + bias；ProxylessNAS Mobile；224×224；batch=8；Adam；学习率 0.0002；label smoothing=0.7；冻结参数 8-bit 聚类量化；50 epochs。训练配置依据官方 `exp_scripts/batch8/flowers.sh`。
- 测试 batch 从官方 100 改为 64；Windows 默认 workers=0；关闭官方把整个测试集缓存在 RAM 的优化，改为逐批读取。
- Flowers102 沿用官方脚本：原 train+validation 共 2,040 张用于训练；原 test 共 6,149 张作为磁盘 `val/`。这是测试集，不应根据它反复调参。做新方法比较时应另外预先划分验证集。
- 官方默认每轮用测试集评估并保存最佳模型；写报告时明确其选模协议，扩展实验避免用测试集选参数。
- 这是现代 Windows 兼容环境，不是 2020 年原始软件栈。数值完全一致不能预先保证。

## 已做兼容修改

- 固定分辨率路径不再导入过时的自定义 DataLoader，保留标准 PyTorch DataLoader。
- 移除新版 scikit-learn 已删除的 KMeans `n_jobs` 参数；保留原算法参数。
- 更新 torchvision MobileNetV2 模块导入路径。
- 权重和配置下载链接指向 MIT Han Lab 旧站 `hanlab18.mit.edu`；源地址及 SHA256 见 `assets.json`。
- 数据和模型缓存重定向到本项目，下载失败抛出明确异常。
- 关闭整套评估图像的 RAM 缓存。
- 完整改动见 `tinytl-compat.patch` 和 `ofa-compat.patch`，官方源代码仍在 `upstream/`。

## 查看结果

在每次运行的 `runs/<运行名>/` 下：

- `args.txt`：实际参数。
- `logs/train_console.txt`、`logs/valid_console.txt`：训练、评估结果。
- `checkpoint/`：权重和训练状态。
- `net_info.txt`：网络信息和论文式内存估算。
- `verification.json`：成功完成后记录 GPU、运行模式及 CUDA 峰值分配内存。

论文的内存估算与 PyTorch 实际显存是不同口径：本代码中的 8-bit 聚类量化用浮点张量承载，不等于 GPU 上真正按 int8 存储。报告中不要把它们混为一个指标。

## 对照实验入口

```powershell
.\.venv\Scripts\python.exe run.py --mode train --method last --seed 0
.\.venv\Scripts\python.exe run.py --mode train --method full --seed 0
.\.venv\Scripts\python.exe run.py --mode train --method tinytl-bias --seed 0
```

这些入口先沿用相同训练超参数，便于检查对照流程；它们不代表已分别调优的论文基线结果。首次只运行一个任务，避免多个训练进程抢占显存。

## 资料

- [TinyTL 官方代码](https://github.com/mit-han-lab/tinyml/tree/master/tinytl)
- [论文](https://arxiv.org/abs/2007.11622)
- [Flowers102 官方数据](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)

重新准备数据：`.\.venv\Scripts\python.exe prepare.py`。数据预处理保留原始文件，不破坏下载材料。
