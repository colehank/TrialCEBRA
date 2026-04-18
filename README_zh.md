# TrialCEBRA
*[English](README.md)*  
**为 CEBRA 提供 trial 感知对比学习** —— 在不修改 CEBRA 源代码的前提下，为其添加五种面向试次结构的采样模式。

适用于神经科学实验中以重复试次（trial）为单位组织的神经记录数据（刺激呈现、条件实验、epoch 数据）。核心思想是将正样本对的选取从"时间点级"提升到"试次级"：先按刺激相似度或均匀随机选择目标 trial，再在目标 trial 内采样正样本时间点。

---

## 背景

CEBRA 原生的三种 conditional（`time`、`delta`、`time_delta`）均在扁平时间序列上操作，面对试次结构数据存在两个问题：

1. **跨试次边界伪影** —— 1D CNN 卷积跨越 trial 边缘，混淆刺激前后的神经活动。
2. **无法利用 trial 层级结构** —— `delta` 在刺激嵌入空间中寻找最近邻时间点；当 trial 内所有时间点共享相同刺激嵌入时，退化为帧内采样，丢失跨 trial 对比信号。

`trial_cebra` 通过将正样本选取提升至 trial 层级解决上述问题。

---

## 安装

```bash
pip install trial_cebra
```

依赖 `cebra >= 0.4` 和 `torch`。

---

## 快速开始

```python
import numpy as np
from trial_cebra import TrialCEBRA

# 神经数据: (N_timepoints, neural_dim)
X = np.random.randn(2000, 64).astype(np.float32)

# 连续辅助变量（如刺激嵌入）: (N_timepoints, stim_dim)
y_cont = np.random.randn(2000, 16).astype(np.float32)

# Trial 边界：40 个 trial，每个 50 帧
trial_starts = np.arange(0,   2000, 50)
trial_ends   = np.arange(50,  2001, 50)

model = TrialCEBRA(
    model_architecture = "offset10-model",
    conditional        = "trial_delta",   # 每步独立重采样的 delta-style trial 选取
    time_offsets       = 5,
    delta              = 0.3,
    output_dimension   = 3,
    max_iterations     = 1000,
    batch_size         = 512,
)

model.fit(X, y_cont, trial_starts=trial_starts, trial_ends=trial_ends)
embeddings = model.transform(X)   # (N_timepoints, 3)
```

### Epoch 格式（ntrial × ntime × nneuro）

若数据已按 trial 组织为三维数组，可直接传入——trial 边界自动推导，无需手动指定：

```python
# Epoch 数据: (ntrial, ntime, nneuro)
X_ep = np.random.randn(40, 50, 64).astype(np.float32)

# 标签可以是 per-trial 或 per-timepoint：
y_pertrial     = np.random.randn(40, 16).astype(np.float32)        # (ntrial, stim_dim)
y_pertimepoint = np.random.randn(40, 50, 16).astype(np.float32)    # (ntrial, ntime, stim_dim)

model.fit(X_ep, y_pertrial)          # 自动识别 3D，无需 trial_starts/ends
emb = model.transform_epochs(X_ep)   # (ntrial, ntime, output_dimension)
```

3D 输入时标签广播规则：

| 标签形状 | 解释 | 输出形状 |
|---|---|---|
| `(ntrial,)` | per-trial 离散 | `(ntrial*ntime,)` |
| `(ntrial, d)`，`d ≠ ntime` | per-trial 连续 | `(ntrial*ntime, d)` |
| `(ntrial, ntime)` | per-timepoint | `(ntrial*ntime,)` |
| `(ntrial, ntime, d)` | per-timepoint | `(ntrial*ntime, d)` |

---

## Conditional 体系

五种 trial-aware conditional，沿三个正交轴设计：

| 轴 | 选项 |
|---|---|
| **Trial 选择方式** | Random（均匀随机）· Gaussian delta-style · Gaussian time_delta-style |
| **时间约束** | Time（目标 trial 内 ±`time_offset` 相对位置）· Free（trial 内均匀，无约束） |
| **锁定方式** | Locked（init 时预计算，全程固定）· Re-sampled（每训练步独立重采样） |

### Conditional 对比表

| `conditional` | Trial 选择 | 时间约束 | 锁定 | Gap 策略 |
|---|---|---|---|---|
| `"trialTime"` | Random | ±`time_offset` | — | 全局 ±`time_offset`（有离散标签时全类均匀） |
| `"trialDelta"` | delta-style | Free（均匀） | **Locked** | 时间点级 delta-style |
| `"trial_delta"` | delta-style | Free（均匀） | Re-sampled | 时间点级 delta-style |
| `"trialTime_delta"` | delta-style | ±`time_offset` | Re-sampled | 时间点级 delta-style |
| `"trialTime_trialDelta"` | time_delta-style | ±`time_offset` | **Locked** | 时间点级 time_delta-style |

原生 CEBRA conditional（`"time"`、`"delta"`、`"time_delta"` 等）直接透传，不受影响。

### 命名规律

```
trialDelta          → 大写 D，无下划线 → Locked + delta-style Gaussian
trial_delta         → 下划线 + 小写 d → Re-sampled + delta-style Gaussian
trialTime           → Random trial + 时间约束
trialTime_delta     → 时间约束 + Re-sampled delta-style
trialTime_trialDelta → 时间约束 + Locked delta-style（time_delta 机制）
```

`_delta`（下划线小写）始终表示 Re-sampled；`trialDelta`（大写 D）始终表示 Locked，与 CEBRA 原生 `delta` 为 re-sampled 的约定保持一致。

---

## 采样机制详解

### Trial 选择：delta-style

用于 `trialDelta`、`trial_delta`、`trialTime_delta`。将 CEBRA 的 `DeltaNormalDistribution` 提升至 trial 层级：

```
query = trial_mean[anchor_trial] + N(0, δ²I)
target_trial = argmin_j  dist(query, trial_mean[j])
```

每个 trial 以其所有时间点连续辅助变量的**均值**作为代表向量。`δ` 控制探索半径：小 `δ` 选取最相似 trial，大 `δ` 广泛探索。由于噪声每步重新采样，同一 anchor 在不同训练步可与不同 trial 配对。

### Trial 选择：time_delta-style

仅用于 `trialTime_trialDelta`。将 CEBRA 的 `TimedeltaDistribution` 提升至 trial 层级：

```
Δstim[k] = continuous[k] - continuous[k − time_offset]   （预计算）
query    = trial_mean[anchor_trial] + Δstim[random_k]
target_trial = argmin_j  dist(query, trial_mean[j])
```

使用实测刺激速度向量作为扰动，数据驱动而非各向同性。

### Locked vs Re-sampled

| | Locked（`trialDelta`、`trialTime_trialDelta`） | Re-sampled（`trial_delta`、`trialTime_delta`） |
|---|---|---|
| 目标 trial | init 预计算，全程固定 | 每训练步独立重采样 |
| 梯度信号 | 一致：同一 trial 对反复比较 | 多样：anchor 每步见到不同相似 trial |
| 泛化性 | 较弱（可能学到 trial 对特有特征） | 较强（学到对所有相似 trial 成立的特征） |
| 适用场景 | 试次较少、需要稳定训练 | 试次较多、刺激内容丰富的视觉任务 |

---

## 采样行为可视化

以下图片由 `example/viz_trial_sampling.py` 和 `example/draft.py` 在真实 MEG + ImageNet 刺激数据上生成。每个面板展示 **R**（参考锚点）、**+**（正样本）、**−**（负样本）。图像边框颜色表示该帧在 trial 内的时间位置（colorbar 见右侧；黑色边框 = gap 时间点）。

### Trial 采样：R / + / −

![Trial 采样可视化](resources/fig_trial_sampling.png)

各 conditional 的正样本分布特征一目了然：

- **`trialTime`**（左上）—— 正样本来自均匀随机的目标 trial，时间位置对齐到 anchor 的相对位置附近。图像网格多样，无刺激相似度偏好。
- **`trialDelta`**（中上）—— 正样本集中在**固定的单个**目标 trial（由刺激相似度在 init 时锁定）。所有正样本图像相同（牛头梗），印证了 `ref_trial → target_trial` 的固定映射。
- **`trial_delta`**（右上）—— 目标 trial 每步重采样。正样本跨越多个相似刺激，内容一致性高于随机，多样性高于 `trialDelta`。
- **`trialTime_delta`**（左下）—— trial 选取多样性与 `trial_delta` 相同，额外叠加 ±`time_offset` 时间窗约束，从 colorbar 可见正样本时间分布更紧凑。
- **`trialTime_trialDelta`**（中下）—— 固定目标 trial（同 `trialDelta`）加时间窗，正样本集中在特定刺激图像和特定 post-stimulus 潜伏期。

### 采样时间线

![采样时间线](resources/fig_sampling.png)

每个采样帧按 trial 内绝对时间标注在时间轴上。绿色高亮区域为 anchor 相对位置 ±`time_offset` 的时间窗。可以观察到：

- 带 `Time` 约束的 conditional（`trialTime`、`trialTime_delta`、`trialTime_trialDelta`）的正样本落在绿色窗内。
- `trialDelta`、`trial_delta` 的正样本均匀分布于整个 trial 长度，无时间约束。
- anchor 在自身 trial 内的绝对时间固定；目标 trial 可以不同，但相对位置对齐。

---

## 学习到的嵌入

以下为在相同 MEG 数据集上以 8 种 conditional（3 种原生 CEBRA + 5 种 trial-aware）各训练 10 000 步的结果。点的颜色按**trial 内时间**编码（黑色 = 刺激前 / gap；黄绿色 = 刺激后晚期）。

### 3D 嵌入（按时间着色）

![3D 嵌入](resources/fig_3d_embeddings.png)

**上排 —— 原生 CEBRA：**

- **`time`** —— 时间点均匀分布于球面，模型未学到任何时间结构（正样本仅来自平坦时间窗，无刺激信息）。
- **`time_delta`** —— 布局相似但稍有组织，刺激速度扰动产生弱时间梯度。
- **`delta`** —— 刺激内容主导。刺激前 gap 帧（黑色）塌缩为球底一个暗色团块，trial 帧广泛分散，无时间序列结构。

**下排 —— Trial-aware TrialCEBRA：**

- **`trialTime_delta`** —— 结构最清晰的时间环：点按 trial 内时间绕球旋转，gap 帧（黑色）分离成独立簇，**同时学到时间结构和 gap/trial 区分**。
- **`trialTime`** —— 类似的时间环，时间窗较宽，梯度更平滑，trial-gap 分离略松。
- **`trialDelta`** —— gap 帧清晰分离，但锁定映射 + 无时间约束产生较分散的 trial 帧分布。
- **`trial_delta`** —— 重采样的 trial 多样性使 trial 帧嵌入更均匀，gap 帧保持独立。
- **`trialTime_trialDelta`** —— 固定 trial + 时间窗，时间环最紧凑，每个潜伏期的点集聚程度最高。

### 训练损失曲线

![训练损失](resources/fig_loss.png)

所有 conditional 均平稳收敛。Trial-aware conditional 初始损失较高（对比任务更难），但最终收敛至与原生 CEBRA 相当的水平，表明模型成功学习了层级结构。

---

## Gap（试次间）时间点

Trial 边界之间的时间点作为**合法 anchor** 参与训练，各 conditional 有各自的 gap 正样本策略：

| `conditional` | Gap 策略 |
|---|---|
| `trialTime` | 全局 ±`time_offset` 窗口；有离散标签时 → 全类均匀（Gumbel-max） |
| `trialDelta` | 时间点级 delta-style |
| `trial_delta` | 时间点级 delta-style |
| `trialTime_delta` | 时间点级 delta-style |
| `trialTime_trialDelta` | 时间点级 time_delta-style |

**推荐做法**：传入离散标签区分 trial 与 gap��如 `0 = gap`、`1 = trial`）。有离散标签时，`trialTime` 的 gap 策略从局部 ±窗口切换为**全类均匀采样**（Gumbel-max trick），迫使所有 gap 时间点在嵌入空间全局聚集，而非在 gap 内部保留时间链结构。

---

## 离散标签支持

所有 conditional 均支持传入离散标签数组。有离散标签时：

- `sample_prior` 使用**类平衡采样**（与原生 CEBRA `MixedDataLoader` 设计一致）。
- Trial 选取限制在**同类 trial** 之间。
- Gap 采样切换为**全类均匀**（Gumbel-max trick）。

```python
# 离散标签：0 = gap，1 = trial
y_disc = np.zeros(N, dtype=np.int64)
for s, e in zip(trial_starts, trial_ends):
    y_disc[s:e] = 1

model.fit(X, y_cont, y_disc, trial_starts=trial_starts, trial_ends=trial_ends)
```

### 仅离散标签（无连续标签）

仅传入离散标签（int 数组，无 float 数组）时，支持 `"trialTime"` conditional。模型从同类别的另一个 trial 中随机选取正样本，并在 ±`time_offsets` 时间窗内采样。

```python
y_disc = np.zeros(ntrial, dtype=np.int64)   # per-trial 标签（epoch 格式）
y_disc[ntrial // 2:] = 1

model.fit(X_ep, y_disc)   # X_ep: (ntrial, ntime, nneuro)
```

Delta-style conditional（`trialDelta`、`trial_delta`、`trialTime_delta`、`trialTime_trialDelta`）需要连续标签计算 trial 相似度，仅传离散标签时会抛出 `ValueError`。

---

## API 参考

### `TrialCEBRA`

继承 `cebra.CEBRA` 全部参数，新增以下行为：

```python
TrialCEBRA(
    conditional: str,      # trial-aware 或原生 CEBRA conditional
    time_offsets: int,     # 时间窗半宽；同时用于 Δstim 的 lag
    delta: float,          # trial 选取的 Gaussian kernel 标准差
    **cebra_kwargs,
)

# 扁平格式
model.fit(
    X,                              # (N, input_dim) 神经数据
    *y,                             # 连续和/或离散标签
    trial_starts: array-like,       # (T,) 起始索引（inclusive）
    trial_ends:   array-like,       # (T,) 结束索引（exclusive）
    adapt: bool = False,
    callback: Callable = None,
    callback_frequency: int = None,
) -> TrialCEBRA

# Epoch 格式 —— trial 边界自动推导
model.fit(
    X,                              # (ntrial, ntime, nneuro) epoch 数组
    *y,                             # per-trial 或 per-timepoint 标签
) -> TrialCEBRA

model.transform(X)         -> np.ndarray   # (N, output_dimension)
model.transform_epochs(X)  -> np.ndarray   # (ntrial, ntime, output_dimension)
model.distribution_                        # 训练后可访问的 TrialAwareDistribution 实例
```

### `TrialAwareDistribution`

采样分布类，可独立使用于诊断分析：

```python
from trial_cebra import TrialAwareDistribution
import torch

dist = TrialAwareDistribution(
    continuous   = torch.randn(500, 16),
    trial_starts = torch.tensor([0, 100, 200, 300, 400]),
    trial_ends   = torch.tensor([100, 200, 300, 400, 500]),
    conditional  = "trial_delta",
    time_offset  = 10,
    delta        = 0.3,
    device       = "cpu",
    seed         = 42,
    discrete     = None,            # 可选，(N,) int tensor
)

ref, pos = dist.sample_joint(num_samples=64)
```

### `TrialTensorDataset`

带 trial 元数据的 PyTorch 数据集，供 sklearn 接口之外使用：

```python
from trial_cebra import TrialTensorDataset

dataset = TrialTensorDataset(
    neural       = neural_tensor,
    continuous   = stim_tensor,
    discrete     = label_tensor,    # 可选
    trial_starts = starts_tensor,
    trial_ends   = ends_tensor,
    device       = "cpu",
)
```

---

## 实现原理

### Post-replace distribution 机制

`TrialCEBRA` 不修改 CEBRA 源码，而是：

1. 临时将 `self.conditional = "time_delta"` 以通过 CEBRA 内部校验。
2. 调用 `super()._prepare_loader(...)` 获取标准的 `ContinuousDataLoader` 或 `MixedDataLoader`。
3. 将 `loader.distribution` 原地替换为 `TrialAwareDistribution`。

两种 Loader 在 `get_indices` 中均只调用 `distribution.sample_prior` 和 `distribution.sample_conditional`，因此替换对训练循环完全透明。

### 混合标签路由

同时传入离散和连续标签时，CEBRA 内部路由始终创建 `MixedDataLoader`（硬编码，忽略 `conditional` 参数）。`TrialCEBRA` 继承该路由后立即替换分布，`conditional` 参数仅对 `TrialAwareDistribution` 生效。

---

## 项目结构

```
src/trial_cebra/
  __init__.py          # 公开 API：TrialCEBRA, TrialTensorDataset, TrialAwareDistribution, flatten_epochs
  cebra.py             # TrialCEBRA sklearn 估计器
  dataset.py           # TrialTensorDataset（PyTorch API）
  distribution.py      # TrialAwareDistribution（5 种 conditional 全部实现）
  epochs.py            # flatten_epochs（epoch 格式数据工具函数）

tests/
  conftest.py
  test_cebra.py
  test_dataset.py
  test_distribution.py
  test_epochs.py

resources/             # example 脚本生成的可视化图片
  fig_trial_sampling.png
  fig_sampling.png
  fig_3d_embeddings.png
  fig_loss.png
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

**发布新版本** —— 版本号从 git tag 自动读取，无需修改任何文件：

```bash
git tag v0.x.x
git push origin v0.x.x   # 触发构建并发布到 PyPI
```
