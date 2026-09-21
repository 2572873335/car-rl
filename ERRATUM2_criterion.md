# ERRATUM 2 — W6 判据设计的撤回（2026-09-21）

> **性质**：对 `plan_W6_criterion.md`（送审稿）的撤回。独立评审 Reject，
> 作者**逐条独立复现，3 条拦截级全部确认**。
> 关联：`ERRATUM_phaseC_findings.md`（第一次适配器 bug）、
> `reviews/20260921_W6_criterion_review1.md`、`DATA_MANAGEMENT.md` §10.4/§10.6

---

## 0. 一句话

`plan_W6_criterion.md` 的**退化反例表三条支柱同时断裂**：
**正例是伪影**（B1）、**世界无随机源**（B2）、**两行退化策略通过且反超正例**（B3）。
**判据作废，不得登记 M1 阈值。**

---

## 1. 三条拦截级（**已逐条独立复现**）

### B1 ❌「正例 FROZEN CKPT = 1.00」是**同一个适配器 bug 的第二次发作**

**根因（我的疏漏）**：`ERRATUM_phaseC_findings.md` §6 修复了
`_verify_f5_fix.selfplay_to_frozen`，**但漏修 `_ckpt_as_opponent.frozen_layout`**——
而新判据的 6 个 `crit_*.py` **恰好 import 了未修的那个**。

**独立复现（适配器对照矩阵，本轮实测）**：

| follower | 对手 | win@1.0 | 碰撞率 |
|---|---|---|---|
| ckpt[BUGGY] | rule[BUGGY]（**plan 用的组合**） | **1.00** | 0.0% |
| ckpt[BUGGY] | rule[**FIXED**] | **0.00** | 100% |
| ckpt[**FIXED**] | rule[BUGGY] | **0.00** | 100% |
| ckpt[**FIXED**] | rule[**FIXED**] | **0.00** | 100% |

**修正任意一侧，正例即归零。**

**连带**：BUGGY 下对手规则机的隐含 gap **恒 ≥1.89 m**，而切入分支要求 `gap < 0.45`
⇒ 对手 `a_lane` 恒为 0.0，**"规则机是真对手（31/40 用内圈）"在判据配置下不成立**。
（31/40 是**另一配置**（规则 vs 规则、中性起跑）的测量，两者不是同一件事。）

**修复**：`_ckpt_as_opponent.frozen_layout` 的 gap 行已改为 `(-delta) % L`，
并加 F12 警示注释。

### B2 ❌ 判据世界**没有随机源** —— "40 局 3 seed 集"是**同一条 rollout**

**根因**：`RoleWorld.reset(rng, gap)` 只做 `self.rng = rng`，**从不读 rng**；
所有位置由 `A_START`/`gap0`/`build_paths()` 决定。

**独立复现（本轮实测）**：

```
rng(1) 与 rng(999999) 的车初始位置逐位相同 ?  True
确定性策略 40 局的终局 delta unique 值：
  do-nothing    : 1   ← 40 局完全相同
  ckpt[FIXED]   : 1
```

**后果**：
- 确定性策略的"胜率"只能取 **{0.00, 1.00}**；
- **M1.2 的阈值 0.8 在构造上不可达**，"边界补 seed"条款**形同虚设**；
- `crit_power_check.py` 的二项功效计算**数学正确但前提不成立**
  （它假设 n 个独立样本，实际只有 1 个世界被重复）。

**性质**：F14 说"初始条件决定结果"；本判据**消除了初始条件却没引入替代变异源**，
于是变成"策略单点决定结果"——**从一个退化换到另一个退化**。

**修复方向**：给 `reset` 加真实随机化（`gap0 ~ U(0.30,0.70)`、对手相位/参数随机化等），
并**先验证确定性策略的 40 局能产生多种终局（unique ≫ 1）**再使用。

### B3 ❌ 两行、零技能策略**通过判据，且在 thr=2.5 反超 plan 的正例**

**独立复现（本轮实测）**：

| 策略 | final_delta | 内圈时长 | 碰撞 | pass@1.0 | pass@2.5 |
|---|---|---|---|---|---|
| **loiter(brk=0.25)** | **+2.707** | 30.0 s | 0% | **1.00** | **1.00** |
| loiter(brk=0.5) | +2.055 | 30.0 | 0% | 1.00 | 0.00 |
| plan 正例（buggy 下） | +2.360 | 29.6 | 0% | 1.00 | 0.00 |

**loiter 的定义只有两行**：`[临近才刹车, 永远内圈]`——**零博弈决策**。

**技能排序被倒置**：判据把两行退化排在 plan 的正例之前。

**根因（评审 I4，本评审最本质的发现）**：
`delta` 被 `wrapL` 限在 **±L/2 = ±2.7425**，而内圈每圈短 ~0.94 m
⇒ **任何"常驻内圈"的策略 delta 都单调爬到 ~+L/2 并钉住**。
实测 loiter 的 +2.707 **正贴 wrap 上界 2.7425**。

**⇒ §2 的"干净 plateau [1.0, 2.0]"其实是 wrap 饱和带**，
它测的是"切内圈切得早不早"，**不是"超车超得好不好"**。
§2 把"thr=2.5 正例也掉"诊断为"过严"是**错误诊断**——2.5 掉的是
**正例**，而 loiter **仍在通过**。

---

## 2. 仍然成立的部分（评审确认）

| 结论 | 状态 |
|---|---|
| 方向正确（任务级判据 + 固定落后起跑打破对称性） | ✅ |
| §1.2 三处修正（+0.5 恒定、阈值≥1.0、V2V 基准） | ✅ 逐条复现成立 |
| §2 表数字未编造 | ✅ 评审原样跑出同一张表 |
| §3 四个 caveat（除 Caveat 4 论证方向外） | ✅ 实测确认 |
| Caveat 2（leader 列无区分力） | ✅ 12/12 策略作 leader 均 1.00 → **建议删掉 leader 列** |
| **Caveat 4「正例是真实超越、非对手失误」** | ❌ **论证方向相反**：越过的是**内圈几何**，不是对手 |

---

## 3. 流程教训（**F12 的第六次重演**）

**F12 已经指出"适配器是静默失效高发区"，并让我建立了
`adapter_fidelity_test.py`——但它没有拦住这次，原因有二：**

1. **单测没有测试真实的适配器**：v1 只自测了一个**局部正确**的 gap 式子，
   **没有 import 任何仓库里的适配器**。于是 `_ckpt_as_opponent` 里的同一 bug
   继续存活。
   **→ v2 已修（`adapter_fidelity_test_v2.py`）：自动发现并断言真实适配器；
   本次 v2 上线即抓出 2 处问题（含 v1 漏掉的那处）。**
2. **修 bug 时只修了报错的那一处**，没有全仓库搜同型模式。
   **→ 教训：修适配器 bug 必须全仓库 grep 同型表达式，不能只修被举报的那一行。**

**新增纪律（建议并入 `DATA_MANAGEMENT.md` §10.4）**：
> 修复任何"静默失效"型 bug 时，**必须全仓库检索同型模式**并逐处修复，
> 且**保真度单测必须 import 真实被测对象**（不得自测局部副本）。

---

## 4. 处置

- **`plan_W6_criterion.md` 判据作废**，M1 阈值**不登记**；
- 代码：`_ckpt_as_opponent.frozen_layout` 已修（B1）；
- 单测：`adapter_fidelity_test.py` → **v2**（守卫真实适配器，已全过）；
- **B2/B3 待修**：世界需加随机源；因变量需换掉"delta 终值"
  （因其被 wrap 饱和）——**这两项是判据重设计的核心**，见
  `plan_W6_criterion_v2.md`（待写）；
- 已推送的 `plan_W6_criterion.md` 将在下次 commit 加撤回标记。

---

## 5. 复现

```bash
cd /home/zy/car_rl/code0919
# B1：适配器矩阵（正例在 buggy/fixed 两侧的表现）
PYTHONPATH=$PWD:/tmp uv run python results/20260921_phaseC_w6/crit_verify_review_claims.py
# B2：世界无随机源
# B3：loiter 通过判据
PYTHONPATH=$PWD:/tmp uv run python results/20260921_phaseC_w6/crit_verify_b3_loiter.py
# 保真度单测（v2，守卫真实适配器）
PYTHONPATH=$PWD:/tmp uv run python results/20260920_phaseC_probe/scripts/adapter_fidelity_test_v2.py
```
