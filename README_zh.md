# TrialCEBRA


[![PyPI](https://img.shields.io/pypi/v/TrialCEBRA?color=blue)](https://pypi.org/project/TrialCEBRA/)
[![Tests](https://github.com/colehank/TrialCEBRA/actions/workflows/tests.yml/badge.svg)](https://github.com/colehank/TrialCEBRA/actions)  
[English](README.md) | 中文  

**为 CEBRA 提供 trial 感知对比学习** —— 在不修改 CEBRA 源代码的前提下，为其添加三种面向试次结构的采样模式。

适用于神经科学实验中以重复试次（trial）为单位组织的神经记录数据。核心思想：将正样本对的选取从"时间点级"提升到"试次级"——先按刺激相似度或均匀随机选择目标 trial，再在目标 trial 内采样正样本时间点。

---

## 背景
<div style="display: flex; gap: 2em; align-items: stretch;">

<div style="flex: 1;">

`CEBRA` 原生的三种 conditional（`time`、`delta`、`time_delta`）均在扁平时间序列上操作，面对试次结构数据存在两个问题：

1. **跨试次边界伪影** —— 1D CNN 卷积跨越 trial 边缘，混淆刺激前后的神经活动。
2. **无法利用 trial 层级结构** —— `delta` 在刺激嵌入空间中寻找最近邻时间点；当 trial 内所有时间点共享相同刺激嵌入时，退化为帧内采样，丢失跨 trial 对比信号。

`trial_cebra` 通过将正样本选取提升至 trial 层级解决上述问题。
- **time conditional**：正样本来自随机选取的目标 trial，时间位置对齐到 anchor 的相对位置附近。
- **delta conditional**：正样本来自基于 trial 辅助变量的 Gaussian 相似度选定的 trial 内，时间位置均匀随机。
- **time_delta conditional**：正样本来自基于 trial 辅助变量的 Gaussian 相似度选定的 trial 内，时间位置对齐到 anchor 的相对位置附近。

</div>

<div style="flex: 1; display: flex; align-items: stretch;">
  <img src="resources/sampling_schema.png" 
       style="width: 100%; height: 100%; object-fit: contain;" />
</div>
</div>
---

## 安装

**第一步 — 安装 PyTorch**，请前往 [pytorch.org](https://pytorch.org/get-started/locally/) 选择适合你硬件的版本（CUDA 版本或 CPU）。

**第二步 — 安装 TrialCEBRA：**

```bash
pip install TrialCEBRA
```

---

## 快速开始

```python
import numpy as np
from trial_cebra import TrialCEBRA

# Epoch 格式神经数据：(ntrial, ntime, nneuro)
X = np.random.randn(40, 50, 64).astype(np.float32)

# Trial 级刺激嵌入：(ntrial, stim_dim)
y = np.random.randn(40, 16).astype(np.float32)

model = TrialCEBRA(
    model_architecture     = "offset10-model",
    conditional            = "delta",   # 基于 trial 相似度的采样
    time_offsets           = 5,
    delta                  = 0.3,
    sample_fix_trial       = False,
    sample_exclude_intrial = True,
    output_dimension       = 3,
    max_iterations         = 1000,
    batch_size             = 512,
)

model.fit(X, y)                        # X 为 3D，自动推断 trial 边界
embeddings = model.transform_epochs(X) # (ntrial, ntime, 3)
```

**各 conditional 对应的标签形状：**

| `conditional` | y 形状 | 含义 |
|---|---|---|
| `"time"` | 不需要 | 随机 trial + ±`time_offsets` 时间窗 |
| `"delta"` | `(ntrial, nd)` 或 `(ntrial, ntime, nd)` | trial 级 或 逐时间点标签（3-D 形式配合 `y_discrete` 可启用类条件 trial 选择）|
| `"time_delta"` | `(ntrial, ntime, nd)` | 时间点级标签 |

---

## Conditional 体系

三种 trial-aware conditional，与 CEBRA 原生 conditional 命名对齐，提升至 trial 层级：

| `conditional` | Trial 选择 | Trial 内采样 | y 要求 | `sample_fix_trial` | `sample_exclude_intrial` |
|---|---|---|---|---|---|
| `"time"` | 均匀随机 | ±`time_offsets` 时间窗 | 不需要 | 无效 | ✓ |
| `"delta"` | 基于 y 的 Gaussian 相似度（提供 `y_discrete` + 3-D `y` 时按类条件进行）| 均匀（自由） | `(ntrial, nd)` 或 `(ntrial, ntime, nd)` | ✓ | ✓ |
| `"time_delta"` | 跨 trial 候选的联合 argmin | ±`time_offsets` 时间窗 | `(ntrial, ntime, nd)` | ✓ | ✓ |

`sample_fix_trial`（默认 `False`）控制 trial→trial 映射的计算方式：`True` 在 init 时预计算并固定，`False` 则每训练步独立重采样。对 `"time"` 无效。

`sample_exclude_intrial`（默认 `True`）控制是否将 anchor 所在 trial 排除在正样本采样之外。`False` 时正例可来自任意 trial（含自身）。

传入扁平 2D 数据时，原生 CEBRA conditional 直接透传，行为不变。

---

## 采样机制详解

### `"time"` —— 随机 trial + 时间窗

用 Gumbel-max trick 均匀随机选取目标 trial（≠ 自身），再在目标 trial 内 ±`time_offsets` 范围内采样正样本时间点。

### `"delta"` —— Gaussian 相似度 + trial 内均匀

将 CEBRA 的 `DeltaNormalDistribution` 提升至 trial 层级：

```
query        = y[anchor_trial] + N(0, δ²I) / √d
target_trial = argmin_j  dist(query, y[j]),  j ≠ anchor
```

`y` 接受 `(ntrial, nd)`（per-trial）或 `(ntrial, ntime, nd)`（per-timepoint）两种形状。`δ` 控制探索半径。正样本在选定 trial 内**均匀**采样。

**Discrete-first 类条件 trial 选择**（提供 `y_discrete` 时）：对齐 CEBRA 官方 `ConditionalIndex` 设计，trial 选择基底改为按 anchor 自身 class 切换：

* **Mode A** —— `y_discrete` 是 per-trial（trial 内不变）：候选限于与 anchor 同 class 的 trial。
* **Mode B** —— `y_discrete` 是 per-timepoint **且** `y` 是 3-D：`trial_emb_per_class[c][trial] = mean(y[trial, t] for t where class(trial,t) == c)`，anchor 用自己 class 对应的基底查询。
* **Mode C** —— `y_discrete` 是 per-timepoint 但 `y` 仅为 2-D：2-D `y` 会自动沿时间轴 broadcast 为 `(ntrial, ntime, nd)`（各时间点 embedding 相同），再走 Mode B 聚合。常量 tensor 的 per-class 均值等于原始值，因此 trial 选择等效于 class-agnostic，但走的是 class-conditional 代码路径，无 warning。

所有模式都会在 `argmin` 之前给 `dists` 加极小 Gumbel 扰动以随机化打破并列（例如所有 trial 的 class-c 嵌入相同时——pre-stim 灰屏标签的典型情况）。

### `"time_delta"` —— 跨 trial 候选联合 argmin

对于位于 `(trial_i, rel_i)` 的 anchor，候选池为所有其他 trial 中相对位置落在 ±`time_offsets` 范围内的时间点：

```
候选池  = {(trial_j, t) : trial_j ≠ trial_i，|t − rel_i| ≤ time_offsets}
query   = y[trial_i, rel_i] + N(0, δ²I) / √d
正样本  = argmin_{(trial_j, t) ∈ 候选池}  dist(y[trial_j, t], query)
```

`y`（形状 `(ntrial, ntime, nd)`）直接用作逐时间点标签，无聚合。正样本同时满足三个约束：**跨 trial**、**时间对齐**（±`time_offsets` 内）、**label 相似**。

对于静态刺激（trial 内 y 恒定），argmin 自然退化为 delta 式 trial 选择 + 时间窗均匀采样，无需特殊处理。

**`fix_trial=True`**：目标 trial 在 init 时锁定，使用与 `"delta"` 相同的 Gaussian 相似度查询（基于 trial onset 嵌入 `y[:, 0, :]`）。每步在锁定 trial 的 ±`time_offsets` 窗口内取 y 距离最小的时间点。

### `sample_fix_trial`

| | `sample_fix_trial=False`（默认） | `sample_fix_trial=True` |
|---|---|---|
| 目标 trial | 每训练步独立重采样 | init 时预计算，全程固定 |
| 梯度信号 | 多样：anchor 每步见到不同相似 trial | 一致：同一 trial 对反复比较 |
| 适用场景 | 试次较多、刺激内容丰富 | 试次较少、需要稳定训练 |

---

## 采样行为可视化

以下图片在真实 MEG + ImageNet 刺激数据上生成。每个面板展示 **R**（参考锚点）、**+**（正样本）、**−**（负样本）。

### Trial 采样：R / + / −

![Trial 采样可视化](resources/fig_trial_sampling.png)

- **`time`** — 正样本来自均匀随机的目标 trial，时间位置对齐到 anchor 的相对位置附近。
- **`delta`** — 正样本来自基于 trial 嵌入 Gaussian 相似度选定的 trial（`fix_trial=False` 时每步变化）。提供 `y_discrete` 时选择按类条件进行（discrete-first 原则）。
- **`time_delta`** — 同样基于速度相似度选 trial，额外叠加 ±`time_offsets` 时间窗约束。

### 采样时间线

![采样时间线](resources/fig_sampling.png)

每个采样帧在完整 trial 时间轴上展示。绿色带标记 anchor 相对位置附近的 ±`time_offsets` 窗口。

---

## 学习到的嵌入

在同一 MEG 数据集上训练全部 6 种 conditional（3 个原生 CEBRA + 3 个 trial-aware）。点按 **trial 内时间**着色。

### 按时间着色的 3D 嵌入

![3D 嵌入](resources/fig_3d_embeddings.png)

**原生 CEBRA（上排）**：`time` — 均匀球面，无时间结构。`delta` — 刺激内容主导，trial 内扁平结构。`time_delta` — 弱时间梯度。

**Trial-aware TrialCEBRA（下排）**：`time` — 跨 trial 对齐形成时间环。`delta` — 按刺激相似度干净聚类。`time_delta` — 最清晰的逐时延结构。

### 训练损失

![损失曲线](resources/fig_loss.png)

所有 conditional 均平滑收敛。Trial-aware conditional 起始损失更高（对比任务更丰富），最终收敛到与原生 conditional 相近水平。

---

## 标签广播规则（Epoch 格式）

当 `X` 为 3D `(ntrial, ntime, nneuro)` 时，标签自动广播为扁平格式：

| 标签形状 | 解释 | 扁平输出形状 |
|---|---|---|
| `(ntrial,)` | per-trial 离散 | `(ntrial*ntime,)` |
| `(ntrial, d)`，`d ≠ ntime` | per-trial 连续 | `(ntrial*ntime, d)` |
| `(ntrial, ntime)` | per-timepoint | `(ntrial*ntime,)` |
| `(ntrial, ntime, d)` | per-timepoint | `(ntrial*ntime, d)` |

---

## 多会话训练

TrialCEBRA 在 trial-aware 采样之上对齐 CEBRA 原生的多会话哲学。把 `X` 传成 **list of epoch-format 数组**（每 session 一个），辅助标签同样按 list 给：

```python
# 2 个 session，每 session 形状可不同 (ntrial, ntime, nneuro)
X = [
    np.random.randn(30, 100, 64).astype(np.float32),   # session 0
    np.random.randn(25,  80, 48).astype(np.float32),   # session 1（异构）
]
y_cont = [np.random.randn(30, 100, 16).astype(np.float32),
          np.random.randn(25,  80, 16).astype(np.float32)]
y_disc = [np.zeros((30, 100), dtype=np.int64), np.zeros((25, 80), dtype=np.int64)]
# ... 设置 pre/post 类别 ...

model = TrialCEBRA(conditional="delta", max_iterations=1000, output_dimension=3, ...)
model.fit(X, y_disc, y_cont)   # 自动检测 list-of-arrays 进入 multisession
```

### 保留 CEBRA 哲学

对齐力来自**跨 session query shuffle**（参考 `cebra.distributions.multisession.MultisessionSampler`）：每 session 在自己 y 空间算 query，所有 query 跨 session 重新分配，每个 positive 一定来自 ≠ anchor 所在 session 的 session，迫使各 session 的 encoder 把语义等价的状态映射到相近的 embedding。`mix` / `index_reversed` 在编码后把 ref ↔ pos 重新对齐，再送入对比损失。

### 支持范围

| Conditional | Multi 支持 | 行为 |
|---|---|---|
| `"delta"` | ✓ 完整 | Mode A / Mode B 类条件 trial 选择按 session 独立；跨 session shuffle；同类约束跨 session 生效 |
| `"time_delta"` | ✓ | y 空间 joint argmin；**±`time_offsets` 时间窗被丢弃**（ntime 异构时相对位置不可跨 session 对齐） |
| `"time"` | ✗ `NotImplementedError` | 与 CEBRA 原生一致——`_init_loader` 在无行为索引时拒绝 multisession |

### 校验规则（init 时）

- **≥ 2 个 session**；允许 `(ntrial_s, ntime_s, nneuro_s)` 异构
- 所有 session 共享相同的连续 y 特征维度 `nd`
- 提供 `y_discrete` 时所有 session 必须共享**完全相同的 sorted unique class 集合**
- **Mode C**（per-timepoint 离散 + 2-D 连续）：2-D y 自动 broadcast 为 3-D，multisession 下透明支持，无限制。
- 严格跨 session：每个 positive 都来自 ≠ anchor 所在 session 的 session（按 batch 位置对 session 轴做 derangement）

### `sample_exclude_intrial` 在 multisession 下

sampler 层已严格保证跨 session，单 session 的 `sample_exclude_intrial` 在此被覆盖。每 session 的 `TrialAwareDistribution` 内部用 `sample_exclude_intrial=False` 避免双重屏蔽。

---

## 输入格式与行为

### 标签类型识别

TrialCEBRA 根据 `dtype` 自动分类输入标签：

| dtype | 分类 | 用途 |
|---|---|---|
| `float32` / `float64` | 连续变量 | `delta` / `time_delta` 的 trial embedding |
| `int32` / `int64` / `uint` | 离散变量 | 类别标签，用于 balanced prior + 同类约束 |

**规则：**
- 最多传入一个连续标签和一个离散标签
- 多个连续或多个离散标签会报错
- `fit(*y)` 中标签的顺序无关 — 分类自动进行

### 连续 `y` 的形状解析

提供多种形状时，`TrialCEBRA` 为每个 conditional 选择合适的：

| conditional | 优先形状 | 备选 | 行为 |
|---|---|---|---|
| `"delta"` | `(ntrial, ntime, nd)` 3-D | `(ntrial, nd)` 2-D | 3-D 启用 Mode B 类条件 trial 选择；`y_discrete` 为 per-timepoint 时 2-D 自动 broadcast |
| `"time_delta"` | `(ntrial, ntime, nd)` 3-D | — | 必须 3-D |
| `"time"` | 不使用 | — | 忽略所有连续 y |

### 离散标签 Prior

提供 `y_discrete` 时，anchor 采样分布由 `sample_prior` 控制：

| `sample_prior` | 行为 |
|---|---|
| `"balanced"`（默认） | 先均匀选类别，再在类内均匀采样 → 少数类过采样 `1 / class_freq` 倍 |
| `"uniform"` | 在所有时间点均匀采样 → anchor 类别分布与实际频率一致 |

严重不平衡数据集应使用 `"uniform"`，避免过采样扭曲 prior。

### 2-D vs 3-D 输入

| 输入形状 | 行为 |
|---|---|
| `X` 为 2-D `(N, nneuro)` | 原生 CEBRA 行为（除非手动提供 `trial_starts` / `trial_ends`，否则跳过 trial-aware 路径）|
| `X` 为 3-D `(ntrial, ntime, nneuro)` | 激活 trial-aware：flatten 为 2-D，附加 trial 元数据，替换为 `TrialAwareDistribution` |

**Conditional 名称解析**：`"time"`、`"delta"`、`"time_delta"` 在 CEBRA 原生和 TrialCEBRA trial-aware 之间共享。区分依据是 dataset 上是否存在 trial 元数据（`trial_starts` / `trial_ends`）— 仅当元数据存在时才激活 trial-aware 行为。

---

## API 参考

### `TrialCEBRA`

继承 `cebra.CEBRA` 全部参数，新增：

```python
TrialCEBRA(
    conditional: str,                    # "time"、"delta"、"time_delta" 或任意原生 CEBRA conditional
    time_offsets: int,                   # trial 内时间窗半宽
    delta: float,                        # trial 相似度匹配的 Gaussian 噪声标准差
    sample_fix_trial: bool = False,      # 在 init 时预计算 trial→trial 映射
    sample_exclude_intrial: bool = True, # 排除 anchor 所在 trial 进行正样本采样
    sample_prior: str = "balanced",      # "balanced" 或 "uniform"；提供 y_discrete 时控制 anchor 采样
    **cebra_kwargs,
)

# Epoch 格式 —— trial 边界自动推导
model.fit(X, *y)           # X: (ntrial, ntime, nneuro)
model.fit_epochs(X, *y)    # convenience alias

model.transform(X)         # → np.ndarray (N, output_dimension)
model.transform_epochs(X)  # → np.ndarray (ntrial, ntime, output_dimension)
model.distribution_        # 训练后可访问的 TrialAwareDistribution 实例
```

### `TrialAwareDistribution`

采样分布类，可独立使用于诊断分析：

```python
from trial_cebra import TrialAwareDistribution
import torch

dist = TrialAwareDistribution(
    ntrial                 = 40,
    ntime                  = 50,
    conditional            = "delta",
    y                      = torch.randn(40, 16),   # (ntrial, nd)
    y_discrete             = None,                  # 可选离散标签 (ntrial*ntime,)
    sample_fix_trial       = False,
    sample_exclude_intrial = True,
    sample_prior           = "balanced",            # "balanced" 或 "uniform"
    time_offsets           = 10,
    delta                  = 0.3,
    device                 = "cpu",
    seed                   = 42,
)

ref = dist.sample_prior(num_samples=64)
pos = dist.sample_conditional(ref)
```

### `flatten_epochs`

将 epoch 格式数组转换为扁平格式并附带 trial 元数据：

```python
from trial_cebra import flatten_epochs

X_flat, y_flat, trial_starts, trial_ends = flatten_epochs(X_ep, y_ep)
# X_ep: (ntrial, ntime, nneuro) → X_flat: (ntrial*ntime, nneuro)
```

### `TrialTensorDataset`

带 trial 元数据的 PyTorch 数据集，供 sklearn 接口之外使用：

```python
from trial_cebra import TrialTensorDataset

dataset = TrialTensorDataset(
    neural       = neural_tensor,
    continuous   = stim_tensor,
    trial_starts = starts_tensor,
    trial_ends   = ends_tensor,
    device       = "cpu",
)
```

---

## 实现原理

**Post-replace distribution** —— `TrialCEBRA` 不修改 CEBRA 源码。它临时将 `conditional = "time_delta"` 以通过 CEBRA 内部校验，调用 `super()._prepare_loader(...)` 获取标准 Loader，再将 `loader.distribution` 原地替换为 `TrialAwareDistribution`。两种 Loader 在 `get_indices` 中均只调用 `distribution.sample_prior` 和 `distribution.sample_conditional`，因此替换对训练循环完全透明。

conditional 命名冲突（`"time"` 和 `"time_delta"` 同时是 CEBRA 原生名称和 TrialCEBRA conditional 名称）通过检测 dataset 上的 trial 元数据来解决：仅当 `trial_starts`/`trial_ends` 存在于 dataset 时才激活 trial-aware 路径，确保传入扁平 2D 数据时完全沿用原生 CEBRA 行为。

---

## 项目结构

```
src/trial_cebra/
  __init__.py       公开 API：TrialCEBRA, TrialTensorDataset, TrialAwareDistribution, flatten_epochs
  cebra.py          TrialCEBRA sklearn 估计器
  dataset.py        TrialTensorDataset（PyTorch 数据集）
  distribution.py   TrialAwareDistribution（三种 trial-aware conditional）
  epochs.py         flatten_epochs 工具函数

tests/
  test_cebra.py
  test_dataset.py
  test_distribution.py
  test_epochs.py
```

---

## 参与贡献

**配置环境**（克隆仓库后运行一次）：

```bash
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

**CI 检查**（push 到 main 时自动运行）：

| 检查 | 命令 |
|---|---|
| Lint + 格式 | `ruff check . && ruff format --check .` |
| 测试 | `pytest tests/ -v` |

**发布新版本** —— 版本号从 git tag 自动读取：

```bash
git tag vx.x.x
git push origin vx.x.x   # 触发构建并发布到 PyPI
```
