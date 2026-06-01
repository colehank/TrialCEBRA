# TrialCEBRA 对比学习采样机制

## 核心概念

TrialCEBRA 将数据视为 epoch 格式 `(ntrial, ntime, nneuro)`，每次对比学习需要为每个 **anchor** 找一个 **positive**。采样分两步：

1. **选 trial**：从哪个试次里取 positive？
2. **选时间点**：在该试次的哪个时刻取 positive？

`conditional` 控制这两步的策略，`y` 变量提供辅助信号。

---

## 三种 conditional

### `time`

| 步骤 | 行为 |
|---|---|
| 选 trial | **均匀随机**，排除 anchor 自身 trial |
| 选时间点 | 在 `±time_offsets` 窗口内均匀采样 |

- 不需要连续 y
- `time_offsets` 始终生效

### `delta`

| 步骤 | 行为 |
|---|---|
| 选 trial | Gaussian 加噪后在 y-space 做 **argmin**，选最相似的 trial |
| 选时间点 | 在目标 trial 内**完全随机**（不受 `time_offsets` 约束） |

- **必须提供连续 y**，形状 `(ntrial, nd)` 或 `(ntrial, ntime, nd)`
- `time_offsets` 对 `delta` **无效**

### `time_delta`

| 步骤 | 行为 |
|---|---|
| 选 trial + 时间点 | **联合 argmin**：在 `±time_offsets` 窗口内所有跨试次候选中，找 y-space 最近的 `(trial, t)` |

- **必须提供连续 y**，形状 `(ntrial, ntime, nd)`
- `time_offsets` 始终生效，约束候选池的时间范围

---

## 辅助变量

### 连续 y（float，`y_continuous`）

| 形状 | 适用 conditional | 用途 |
|---|---|---|
| `(ntrial, nd)` | `delta` | 每试次一个向量，计算试次间距离 |
| `(ntrial, ntime, nd)` | `delta` / `time_delta` | 每时间点一个向量；`delta` 取均值作为试次表征，`time_delta` 取 onset（`y[:, 0, :]`） |

### 离散 y（int，`y_discrete`）

形状平铺为 `(ntrial * ntime,)`，可选，作用独立于连续 y：

**① 影响 anchor 采样（prior）**

| `sample_prior` | 行为 |
|---|---|
| `"balanced"`（默认）| 先均匀选 class，再在该 class 内随机选时间点（少数类过采样） |
| `"uniform"` | 所有时间点均匀采样（保持自然类频率） |

**② 影响 positive 采样（同类约束）**

每种 conditional 的同类约束介入方式不同：

| conditional | 离散 y 如何作用 |
|---|---|
| `time` | 过滤候选 trial（目标 trial 的窗口内须有同类时间点）；窗口内 Gumbel-max 采同类时间点 |
| `delta` | Mode A/B/C 类条件化试次选择（见下节）；目标 trial 内 Gumbel-max 采同类时间点 |
| `time_delta` | 联合 argmin 前 mask 掉异类候选（`dist = inf`）；对距离计算本身无影响 |

---

## `delta` + 离散 y 的 Mode 检测

`delta` 会在 init 时根据离散 y 的结构自动选择试次选择策略：

| Mode | 触发条件 | 试次选择策略 |
|---|---|---|
| **Mode A** | 离散 y 在每个 trial 内恒定 | 只在同类 trial 中做 Gaussian 相似度 argmin |
| **Mode B** | 离散 y 随时间变化，连续 y 为 3-D | 构建 `trial_emb_per_class[c, trial] = mean(y[trial, t] for t where class==c)`，用类专属向量查询 |
| **Mode C** | 离散 y 随时间变化，连续 y 为 2-D | 2-D y 自动 broadcast 为 3-D，等价于 Mode B（无 warning） |

> `time` 和 `time_delta` 不做 Mode 区分，同类约束直接在采样时 mask 实现。

---

## `sample_fix_trial`

| 值 | 行为 |
|---|---|
| `False`（默认）| 每个训练步重新选目标 trial，多样性高 |
| `True` | init 时预计算 trial→trial 映射并锁定，训练全程不变 |

有离散 y 时，`fix_trial=True` 会按 `(n_classes, ntrial)` 维度分别锁定每个类的 trial 映射。

`time` 忽略此参数（始终随机）。

---

## `sample_exclude_intrial`

| 值 | 行为 |
|---|---|
| `True`（默认）| positive 必须来自不同 trial |
| `False` | 允许 positive 来自 anchor 自身 trial；`delta` 无离散 y 时改用 Gumbel-max 软采样，避免高维下 self-trial 永远最近 |

---

## 选择指南

```
只有离散 y，需要时间窗约束      → time
只有连续 y，不在意时间对齐      → delta
只有连续 y，需要时间对齐        → time_delta
连续 y + 离散 y，不在意时间对齐 → delta   （类条件化试次选择 + 同类时间点）
连续 y + 离散 y，需要时间对齐   → time_delta（连续 y 算距离，离散 y mask 异类）
```

---

## 连续 y 依赖总结

| conditional | 连续 y | 离散 y |
|---|---|---|
| `time` | 不需要 | 可选 |
| `delta` | **必须** | 可选 |
| `time_delta` | **必须** | 可选 |
