# F19 — 归档脚本不可从仓库复现（评审进行中发现的独立问题）

日期：2026-09-21
发现者：作者（自查，**在 review2 进行中**）
性质：**可复现性缺陷**，**不影响已测结论的正确性**

---

## 1. 现象

`results/20260921_phaseC_w6/crit_*.py`（24 个）**无法仅凭仓库运行**：

```
$ PYTHONPATH=/home/zy/car_rl/code0919 uv run python -c "import _ckpt_as_opponent"
ModuleNotFoundError: No module named '_ckpt_as_opponent'
```

**两个原因叠加**：

1. **导入解析到 `/tmp`**：脚本写 `sys.path.insert(0, "/tmp")`，
   而 helper 模块（`_ckpt_as_opponent.py` 等）**在仓库里的位置是**
   `results/20260920_phaseC_probe/scripts/`——**该目录不在任何脚本的 `sys.path` 上**。
   故 `import` 命中的始终是 `/tmp` 副本。
2. **部分 helper 从未归档**：`_criterion_v2.py`、`_criterion_v3.py`
   （`FULL` 因变量的定义处！）**只存在于 `/tmp`**，
   仓库 `scripts/` 下的 `_criterion*` 计数为 **0**。

**后果**：`results/20260921_phaseC_w6/` 里的"原始输出"**无法从仓库复现**，
违反 `DATA_MANAGEMENT.md` §2/§5 的可溯源要求。

---

## 2. 对结论的影响：**无**

**关键事实**：当前 `/tmp` 与仓库的 `_ckpt_as_opponent.py` **逐字节相同**
（`diff -q` 通过），**且都是修复后的版本**。

```
$ diff -q /tmp/_ckpt_as_opponent.py results/20260920_phaseC_probe/scripts/_ckpt_as_opponent.py
COPIES IDENTICAL -> conclusion unaffected
```

**故**：
- ✅ 本轮的实测数字（正例 FULL=1.00、loiter FULL=0.00 等）**有效**；
- ❌ 但"把脚本拷进仓库"**不等于**"结果可复现"——
  读者按仓库内容跑不出来。

**区分要点（重要）**：
> **F18（`/tmp` 陈旧副本导致测量被污染）** = **正确性**问题（已发生一次）。
> **F19（helper 未归档 + 路径指向 `/tmp`）** = **可复现性**问题（本轮发现）。
> 二者**同源**（代码有两份），但**严重度与处置不同**。

---

## 3. 处置（**✅ 已执行，2026-09-21**）

1. 把 `_criterion_v2.py`、`_criterion_v3.py`、`_rolefixed_probe.py` 等
   helper **归档进 `results/20260920_phaseC_probe/scripts/`**；
2. 改写 `crit_*.py` 的 `sys.path`：
   **仓库 `scripts/` 目录优先**，去掉 `/tmp` 优先项；
3. **验证**：每个归档脚本在**无 `/tmp`** 的环境下可导入/运行；
4. 把该验证加入 `adapter_fidelity_test_v2.py` 或独立成
   `scripts_reproducibility_test.py`。

> **执行说明**：改动**行为保持**（两份副本内容逐字节相同、且均为修复版），
> 故不改变已测数字。修改后**已复跑正例测量验证**：
> FULL=1.00、零碰撞、t_overtake 1.62 s —— 与改动前**完全一致**。

---

## 4. 流程教训（F19）

**"把脚本拷进仓库"是必要不充分的。** 归档时必须
**同时归档其依赖**，并**在被归档的位置实际跑一次**——
本轮正是因为"拷了主脚本、没拷 helper、路径还指向 `/tmp`"，
造成"看起来归档了、其实跑不起来"。

**建议并入 `DATA_MANAGEMENT.md` §10.4**：
> 归档实验脚本时，必须**从归档位置**执行一次（或做导入测试），
> 确认其依赖齐全、且**不依赖 `/tmp` 等易失路径**。


---

## 5. 执行结果（2026-09-21）

| 项 | 状态 |
|---|---|
| 归档缺失 helper（`_criterion_v2/v3`、`_rolefixed_probe`） | ✅ |
| 24 个 `crit_*.py` 的 `sys.path` 改为**仓库优先**、去掉 `/tmp` | ✅ |
| **helper 模块自身的 `/tmp` 插入**（真正的根因）也已移除 | ✅ |
| W5 期 14 个脚本（`_pilot_gate.py`、`adapter_fidelity_test*.py` 等）同样处理 | ✅ |
| **新增回归守卫** `archived_scripts_reproducible_test.py` | ✅ 全过（50 文件） |
| **行为保持验证**（复跑正例测量） | ✅ 数字完全一致 |

**根因补充（重要）**：仅改 `crit_*.py` 的 `sys.path` **不够**——
**helper 模块自己被 import 时也会**执行 `sys.path.insert(0, "/tmp")`，
把 `/tmp` 重新注入路径，使前面的清理失效。
**这正是"同一问题有两层"的典型**：修了调用方，没修被调用方。

**验证方式（可复现）**：把仓库树拷到干净目录、**移除 `/tmp` 下的 helper**，
再导入归档脚本 —— 5/5 加载成功、0 个 `ModuleNotFoundError`。
