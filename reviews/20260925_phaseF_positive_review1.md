# 评审记录 — Phase F 正例锚点（`positive pole`）：第 11 轮

- **日期**：2026-09-25
- **对象**：`results/20260925_phaseF_positive/` 正例终版
  （`positive_pole.py` / `_out_pole_final.txt` / `_probe_forced_win.py` / `_out_pole_v4.txt` / `v2`·`v3`·`v4`）
- **评审方式**：逐条**实测**。仅在运行期 monkeypatch 冻结模块的常量（`fe.D_DES`，在 `finally` 中复位），
  **未修改任何冻结文件**。独立复现 1 次终验 + 6 轮评审探针（新增 6 个脚本，见文末）。
- **流程依据**：`DATA_MANAGEMENT.md` §10.3 / §10.4 / §10.5b / §10.5c / §10.9、铁律 1、铁律 2、
  `plan_phaseF_adversary_v1.md` v1.1 §2.3/§2.4/§4、`ERRATUM3`（F20）、`reviews/20260925_phaseF_holdout_B.md`
- **机器状态披露（F8）**：每次基准跑前 `pgrep -af 'uv run python'` 已核对。终验复现跑（02:35:56 →
  02:38:24）期间机器为空。期间**观察到一次并发的短命非训练进程**
  （`results/20260925_phaseF_attacker/_check_reward_scale.py`，属 Step 1 的奖励标定，数十秒结束），
  说明本轮部分探针与该进程有 CPU 时间重叠。**本轮全部指标是确定性事件计数**（撞车数、终止原因、
  稳态误差、收距率），**不是性能/耗时基准**，故 `measure-performance-in-isolation` 那条陷阱不适用；
  证据是同一探针在两个互不相干的攻击种子块上给出可解释的重复值（§S13 / §T3）。
- **指标口径**：沿用仓库既有 obs 仪器单位 `obs[0] = e/0.5`，即「obs-cm」= `200 × |e|[m]`。
  本文所有 err 数字均为该口径，与 `_out_pole_final.txt` 直接可比。

---

## 裁决：**Reject（正例作为 H-A3 锚点不成立；正例作为策略本身可用）**

正例**作为一条手工控制器是真的**且**可复现**：我独立重跑 `positive_pole.py`，
输出与 `_out_pole_final.txt` **逐字节相同**（`diff` 空），0/660，名义 0.88× / 1.00× / 1.00×。
冻结文件 5 个 sha256 全部未动。防泄漏干净。这些都不是问题。

**Reject 的理由只有一条，但它是承重的**：
本产物的**声明用途**是"H-A3 阈值的依据 / 阈值锚点"（plan §4、owner Q6）。
而实测表明 **H-A3 的度量（B 组撞车率）被"零动作"这一最懒惰的策略满分通过**
（B 组 0/100，与正例同为 0/100，见 §一.R1），且 **H-A2/H-A3/H-A4 三项合取仍被
"名义精确 + 遇扰冻结"这类退化策略整体通过**。
⇒ 该锚点**锚不住**它被要求锚定的那条判据；
"0 撞车"这个正例无法为"相对下降 ≥50%"提供任何阈值信息。

> **可挽救性（评审的明确预期）**：正例本身**不需要重做**。
> 需要重做的是 **H-A3 的口径**（§二.A 给了完整修正文本）。按修正口径改动后，
> 正例预计可直接通过复审；本轮 Reject **不**要求重新设计控制器。

---

## 一、拦截级问题（Reject 依据，逐条带实测命令与输出）

### R1 ★★（本轮最重要）H-A3 的度量被「零动作」满分通过——判据与目标脱钩（§10.5c 同型）

**问题**：H-A3 = "B 组撞车率相对**未训练基线**下降 ≥50%"。三个口径缺陷叠加：

1. **基线未命名**。`research/ASSUMPTIONS.md` 与 plan §2.3/§3 均只写"未训练方"，
   未指明是谁。而三种候选读数结论完全不同：
   - 基线 = P+FF：B 组 3/100 ⇒ 阈值 ≈ ≤1.5/100（**近乎无分母**，且见第 3 点）；
   - 基线 = 冻结 v1 RL：B 组 **36–41/100** ⇒ 阈值 ≈ ≤18–20/100（有实质门槛）；
   - 基线 = 正例：0/100 ⇒ 阈值 **不可满足**（0 无法再降 50%）。
2. **撞车率是可以被"什么都不做"买到的**。V2V 速度匹配先验使 `a ≡ 0` 天然保距。
3. **相对下降的噪声不可忽略**：同一条 v1 RL，仅换攻击种子块（50000+ vs 77000+）
   就从 **36/100 → 40/100**（§T3）。阈值因此自带 ±2 点噪声。

**实测命令与输出**（`_review_probe_round6_lite.py`，B 组 5 项 × 20 局，
env seed 60000+k、attack seed 50000+k，与 step05/eval 脚本选种一致）：

```
        policy       B1      B2      B3      B4      B5    TOTAL   nominal err
   zero action     0/20    0/20    0/20    0/20    0/20     0/100         83.22
          P+FF     3/20    0/20    0/20    0/20    0/20     3/100          2.49
          pole     0/20    0/20    0/20    0/20    0/20     0/100          2.19
```

同一脚本对**冻结 v1 RL**（`ckpt/follow_stage2_final_v1.zip`，H-A3 的"未训练方"）：

```
    item    v1 RL crash          (attack seeds 50000+, env seeds 60000+k 每项各自重置)
      B1           1/20
      B2          13/20
      B3          20/20
      B4           7/20
      B5           0/20
   TOTAL          41/100
```

> **口径披露（§10.6）**：上面这张表来自 `_review_probe_round3.py` §S10，
> 其 env seed 是**每项**都从 60000 起算（B1..B5 各用 60000–60019）。
> `_review_probe_round4.py` §T3 改用**跨项连续**的 env seed（60000–60099），
> 同一批攻击实例给出 **36/100**。**两数是同一定义下的合法变体，差 5 点。**
> 本文其余处引用 36 与 40 时，均指 §T3 那个（env seed 连续）的口径——
> 因为 §T3 才是"只换攻击种子块"的干净对照（50000+ → 36；77000+ → 40）。
> **这正是 §R1 第 3 点（噪声不可忽略）的直接证据：基线本身在 ±2–5 点内浮动。**

**⇒ 零动作在 B 组拿到 100% 的"相对下降"。** H-A3 作为**撞车率**判据，
对"训练后防御方"与"永不动作的防御方"**给出相同判定**——这与 §10.5c 记录的
M1.3 事故（随机行为得分最高、按判据行事者完成率为 0）是**同一失效模式**。

**为什么 H-A4 救不回来**：`A4 = 名义 settled 误差 ≤ 基线 1.5×` 只在**名义工况**上取值。
零动作名义 err 83.22 ⇒ 确实过不了 A4。但**「名义用 P+FF、遇扰冻结」**这类策略
**同时**过 H-A2（无撞车）、H-A3（100% 下降）、H-A4（名义精确）。
而正例自己就是这个形状：**名义段逐动作等于 P+FF**（§R2、§U4）。
⇒ 三项判据的合取**仍然**不排除"名义精确 + 遇扰不跟"。

**建议修正（H-A3 改写为合取，五条）**：

```
H-A3'（修订版）：
  (a) 命名基线：baseline := 该防御方血缘在对抗训练之前的同一条策略
      （v1 RL 血缘 → 冻结 v1 RL；P+FF 血缘 → P+FF）。A-2 的 30 seed 必须用在
      **成对**攻击实例上（B 组同一批实例、同一 seed，训练前后各跑一次），
      而不是"基线 20 局 + 训练后另 20 局"。
  (b) 绝对上限：B 组撞车率 ≤ 10%（不依赖基线）。
  (c) 相对下降 ≥50% vs (a) 的命名基线，且**同时公示基线的 seed 波动带**
      （本轮实测 v1 基线在两组 B 攻击种子上为 36/100 与 40/100）。
  (d) **必列零动作对照行**（§10.4/§10.5c 硬性要求）：B 组评估表必须逐项列出
      zero-action 的撞车率**与跟踪误差**。若训练后的防御方在撞车率上不优于
      零动作（本轮 zero-action = 0/100，已触顶），则该项判据无分辨力，
      不得据其判定；须改由 (e) 判定。
  (e) **跟踪质量门（新增，承重）**：B 组 settled 误差 ≤ 正例的 1.5×
      （正例 d=0.2 为 2.19 obs-cm，故门为 ≤3.28）。这是排除"名义精确 + 遇扰
      冻结"退化的**唯一**手段；缺此一条，H-A2∧H-A3∧H-A4 的整体判据仍为空。
```

### R2 ★ 承重机制与文档陈述不符："预判-避让"的**预判**那一半是死代码

**plan §4** 把正例描述为「手写"**预判**-避让"跟随脚本」；`positive_pole.py` 的 docstring 同样声称
`This is the "anticipate"`；RUNLOG 表述为「**收距超限即全力刹车，否则退化为纯 P**」。
**代码实际是**：

```python
if dv > YTHRESH:                     # ← "预判/anticipate" 支路
    self.extra = min(self.extra + YG*dv, YMAX)
...
if c > c_max: return [-1.0]          # ← 全力刹车支路（按**达成**收距率）
a = KP*(e - self.extra)/ACT
return clip(a, -1.0, min(1.0, c_max/ACT))   # ← 夹持支路
```

**实测三条，全部证伪了上述三种陈述**：

1. **"预判"支路是死代码。** `YTHRESH = 0.20`，而单步内跟随车速度**最大可能变化量**
   是 `a_max·DT = 1.5 × 0.02 = 0.030 m/s`（DR 打开时为 0.020–0.040）。
   `0.20 / 0.030 = 6.7×` ⇒ `dv > 0.20` **物理上不可达**。

```
U1  MAXIMUM ACHIEVABLE |dv| per step -- is YTHRESH=0.20 reachable?
  dr=False            nominal  max|dv_follower| per step = 0.0300 m/s
  dr=False     step-down 0.1s  max|dv_follower| per step = 0.0300 m/s
  dr=False   full-range 0.05s  max|dv_follower| per step = 0.0300 m/s
  dr= True            nominal  max|dv_follower| per step = 0.0307 m/s
  dr= True     step-down 0.1s  max|dv_follower| per step = 0.0307 m/s
  dr= True   full-range 0.05s  max|dv_follower| per step = 0.0307 m/s
  WORST over all: 0.0307 m/s vs YTHRESH 0.2  ->  branch is UNREACHABLE (dead code)
```

   跨全部 11 个攻击族 × 20 局 × 3 个间距，该支路**触发 0 次**：

```
W3  BRANCH OCCUPANCY, 11 families, d=0.20, 20 ep each
            attack   steps   brake%   clamp%   yield%
    step-down 0.1s    1001     6.9%     0.0%     0.0%
    step-down 0.2s    1001     3.8%     0.0%     0.0%
    step-down 0.4s    1001     2.4%     0.0%     0.0%
    step-down 0.8s    1001     6.0%     0.0%     0.0%
         osc 0.5Hz    1001     6.6%     0.0%     0.0%
         osc 0.3Hz    1001     8.1%     0.0%     0.0%
         osc 0.8Hz    1001     5.7%     0.0%     0.0%
             chirp    1001     4.0%     0.0%     0.0%
          pw-const    1001     3.2%     0.0%     0.0%
           nominal    1001     5.3%     0.0%     0.0%
             crawl    1001     7.9%     0.0%     0.0%
```

2. **夹持支路（`c_max/ACT` 上界）从不生效**。名义工况 1001 步中，
   P 项需求超过夹持上限的步数为 **0**：

```
U4  steps where the P demand exceeds the clamp: 0/1001
    clamp cap c_max/ACT: at gap 0.20 -> 0.704, at gap 0.61 -> 0.704
```

3. **因此有效控制器是**：`P+FF`，**除了**那 2.4–8.1% 触发全力刹车的步。
   即 `a = -1 if (-de > c_max) else clip(0.8·e/ACT, -1, 1)`。

**结论**：正例的**承重机制只有一条**——按**达成**收距率 `c = -d(gap)/dt` 与
间隙相关包络 `c_max = 0.60·sqrt(2·1.0·(gap−0.17))` 的比较结果全力刹车。
`margin/safety/a_max_lo` 三个常量有效，`ythresh/yg/ymax/recover` 四个**全部无效**
（死代码）。**文档必须按此更正**——否则论文会把一个不存在的机制写进去，
这正是 §10.6 诚实条款与 F20 血缘所针对的情形（`_probe_forced_win.py` docstring 中
"约束是**收距速率**"这句话，在本轮实测下应改为"约束是**达成**收距速率**的上界**，
且其实现是**全力刹车**而非**命令夹持**"）。

---

## 二、重要问题 / 建议

### A. ★ 核心问题回答：正例**不是**"把收距速率限死"的平凡解，但也不是它自称的那个设计

owner 的提问是"这个正例是否真的在测**抗扰能力**，还是只是把**收距速率**限死？
后者是 F20 的同型风险：一个平凡的恒低速解也能零碰撞"。
**分三点作答，每点都有实测**：

**(1) 它确实不是"恒低速跟随"——三条判据全部否证平凡解。**

```
S2  CATCH-UP AGILITY, nominal, d=0.20 (leader const 0.5, gap0 ~ U(0.2,1.2))
              pole (final)  coll  0/20  t_catch mean  3.18s max  3.60s  settled err   2.24  peak closing  0.57 m/s
                P+FF (ref)  coll  0/20  t_catch mean  3.18s max  3.60s  settled err   2.54  peak closing  0.57 m/s
    const-cap 0.15 (crawl)  coll  0/20  t_catch mean  4.24s max  6.36s  settled err   3.81  peak closing  0.41 m/s
            const-cap 0.30  coll  0/20  t_catch mean  3.36s max  5.08s  settled err   2.59  peak closing  0.54 m/s
     slowP kp=0.15 (crawl)  coll  0/20  t_catch mean 13.29s max 16.88s  settled err  22.58  peak closing  0.37 m/s
               zero action  coll  0/20  t_catch mean   nans max   nans  settled err 113.52  peak closing  0.28 m/s
```

   追赶时间 `3.18 s` 与 P+FF **逐位相同**，名义 setted 误差 `2.24` **优于** P+FF 的 `2.54`，
   峰值收距率 `0.57 m/s` 与 P+FF **相同**。两个"慢慢跟"基线（恒定收距率夹持、小增益 P）
   **两项都更差**。⇒ **"靠不追换零碰撞"被否证**。

   **矛盾的解释**（owner 特别要求解释"限死却名义不退化"）：包络 `c_max` 是**间隙相关**的——
   在大 gap 时它**放得很松**（gap=0.61 时 `c_max = 0.60·sqrt(2·1·0.44) = 0.563 m/s`，
   已接近执行器可达峰值 0.57），在小 gap 时才收紧（gap=0.20 时 `0.147 m/s`）。
   因此**追击阶段几乎不受限，稳态阶段收距率本就≈0（名义均值 +0.007 m/s）**，
   包络只在"**已经积累起收距率、而间隙又小**"的那一瞬间起作用。
   这就是"名义精度不受影响"的成因，它是**设计使然，不是巧合**。

**(2) 但"限死收距速率"这个读法本身被实测否证——夹持那一半是无效的。**

```
S6  SHARPENED ABLATION, d=0.20 (hard subset = step-down 0.1/0.2/0.4s)
                pole (final)        0/220
       clamp only (no brake)       60/220     ← 只夹持命令：崩
       brake only (no clamp)        0/220     ← 只按达成率全力刹车：全过
           P+FF, no envelope       60/220
```

   **只做"把命令收距速率限死"（`clamp only`）＝ P+FF ＝ 60/220 崩**；
   把夹持去掉、只留"按**达成**收距率全力刹车"（`brake only`）＝ 0/220 全过。
   ⇒ owner 假设的 **F20 同型风险（"平凡地把收距速率限死"）在本例中不成立**：
   那个形式**根本活不下来**。承重的是"**对已达成量的全权刹车**"这一非平凡要素。

**(3) 与"显然的一行替代"相比，包络确实买到了东西。**
最自然的一行替代是"gap 小于阈值就全力刹车"（固定阈值）：

```
V4/S6/S11 (d=0.20)
    pole (final)                     0/220   （hard subset step-down 0.1/0.2/0.4s: 0/60）
    P+FF + hard brake g<0.20        26/220   （hard subset 26/60）← 过不了硬攻击
    P+FF + hard brake g<0.25          —      （hard subset 4/60） nominal err 10.74
    P+FF + hard brake g<0.30         0/220   （hard subset 0/60）  nominal err 20.39 ← 过不了 A4（1.5×）
    P+FF, no envelope               60/220   （hard subset 60/60）
```
   固定阈值要么过不了硬攻击（0.20 → hard 26/60 崩），要么靠牺牲名义精度换安全
   （0.30 → 名义 err 20.39，是 P+FF 2.54 的 8.0×）。
   间隙相关的**包络**正是"两者兼得"的原因。**这是正例真正的技术含量**。

**综上（A 的判决）**：正例**不是平凡解**，"限死收距速率"这一读法**被实测否证**；
但它**也不是** plan §4 所称的"**预判**-避让"脚本——没有预判（§R2）。
准确表述应为：**"P+FF + 一条按达成收距率与间隙相关包络比较的全力刹车规则（触发率 2.4–8.1%）"**。

### B. C 组攻击族充分性：11 族**未发现漏掉能击穿正例的攻击类**，但 B2 的参数与正例验证集重合

**(a) 攻击族扩测**（`_review_probe_round3.py` §S7/S15，`_review_probe_round6_lite.py` §W2）：

```
  rand sine f~U(.2,1.0) a~U(.3,.5) ph~U(0,2pi)   pole  0/20   P+FF  0/20
          small steps +-0.15 / 0.15s              pole  0/20   P+FF  0/20
           tiny steps +-0.08 / 0.10s              pole  0/20   P+FF  0/20
          small steps +-0.25 / 0.25s              pole  0/20   P+FF  0/20
             square 0.05s (20Hz)                  pole  0/20   P+FF 20/20   ← 真攻击，正例免疫
             step 1.0->0.05 once                  pole  0/20   P+FF  0/20
               staircase down 5x                  pole  0/20   P+FF  0/20
              random phase 0.5Hz                  pole  0/20   P+FF  0/20
         big osc 0.5+-0.45 1.2Hz                  pole  0/20   P+FF  0/20
   full-range hold 0.05/1.0 0.1s                  pole  0/20   P+FF 20/20   ← 真攻击，正例免疫
```
   另加 **1500 个随机波形参数搜索**（方波/正弦/分段常值/上冲后撤/宽带 chirp/窄脉冲，
   周期 0.05–1.5 s、频率 0.1–3.0 Hz、幅值 0.1–0.5、双脉冲间隔 0.1–1.5 s）：

```
W2  candidates: 1500   crashes: 0   => no crossing waveform found
```
   **⇒ 未能构造出使正例撞车的波形。** 结构原因（不只是运气）：包络随 gap 增长，
   而在**小 gap** 处让跟随车维持高收距率需要超出执行器能力
   （`a_max·DT = 0.03 m/s/步`，见 §R2-U1）。

**(b) 但 §C 的"重叠度"问题是真的，且指向锚点的角色**：
`reviews/20260925_phaseF_holdout_B.md` 的 **B2 = 正弦 f ∈ {0.3, 0.8} Hz**，
而 **`positive_pole.py` 自己的验证字典里就有 `osc 0.3Hz` 与 `osc 0.8Hz`**。
B3（chirp 0.1→0.6 Hz）落在正例 chirp 频带（0.1→1.1 Hz）内；B1（周期 {0.3, 0.7} s）
是正例 {0.2, 0.4, 0.8, 1.6} 的插值。
**这不违反 §2.4**（B 组的"held-out"是相对 **Step 2 训练代码**，正例不是训练代码），
**但它使"正例 0/660"与"B 组"不构成两份独立证据**：
正例在 B 组上的 0/100 **不能**当作"泛化到 held-out 族"的证据，
因为 B2 的两个频率就是它的调参/验证频率。
⇒ **报告措辞必须限定**：正例只锚 **A 组风格**攻击；对 B 组的零碰撞是**同分布确认**，非泛化证据。
（旁证：`results/20260924_phaseF_probe/_out_positive_pole.txt` 那时只测 `osc f=0.5`，
0.3/0.8 是在 `holdout_B.md` 落盘（00:09）**之后**才进入正例探针集的（正例终版 00:57）。
不构成违规，但顺序就是如此，须如实记录。）

### C. `d=1.0 / step-down 0.2s` 的 err = 78.28：**不是** wrap 伪影，是**执行器受限的死格**（Scenario 有效性问题）

owner 问的正是这个，**先给出直接答案**：
`_measure()` 返回 `(s_l - s_f) % path.length`，**确实**有"套圈后与"落后一整圈"同读数"的理论风险。
我**独立验证了是否发生**（`_review_probe_round4.py` §T2，同时记录包裹 gap 与
**未包裹**纵向间距）：

```
T2  (d=1.0, step-down 0.2s, seed 2000 —— 单局；归档的 78.28 是 seed 2000+k 的 20 局均值）
   pole  wrapped gap: mean 0.607  min 0.581  max 0.636   -> err  78.65
        UNWRAPPED separation: mean 0.607  min 0.581  max 0.636   -> drift -0.003 m over 20.0 s
   P+FF  wrapped gap: mean 0.607  min 0.581  max 0.636   -> err  78.65
        UNWRAPPED separation: mean 0.607  min 0.581  max 0.636   -> drift -0.003 m over 20.0 s
  CONTROL nominal d=1.0: wrapped mean 0.999  unwrapped mean 0.999  err 3.63
  另：20 局均值口径下 pole 与 P+FF **同为 78.28**（`_review_probe_round2.py` §S5 与 S14 两次独立测得）
```

**⇒ 未包裹间距 ≡ 包裹 gap（漂移 −0.003 m），没有套圈 ⇒ 不是 wrap 伪影。**
（lap 长 5.4849 m，20 s 内根本不套圈。）

**真正的成因**（`_review_probe_round5.py` §U3）：

```
U3  (d=1.0, step-down 0.2s)
   pole  mean actual leader v 0.240  mean follower v 0.240  mean commander-mean 0.501
         gap0 0.610  mean gap 0.607  frac of steps with v_cmd clipped at 0  0.50
   P+FF  完全相同的六个数字（0.240 / 0.240 / 0.501 / 0.610 / 0.607 / 0.50）
  CONTRAST nominal d=1.0: leader v 0.500  follower v 0.499  gap 0.999
```
`a_max = 1.5`、`DT = 0.02` ⇒ 领车自己的执行器**跟不上 0.2 s 方波**：指令均值 0.501，
**实际**速度均值只有 0.240。跟随车的 `v_cmd` 在下半周期被 clip 在 0（**50% 的步**），
因此它的均值速度**不可能低于**领车 → **gap 永久停在 0.607，与策略无关**。

**⇒ 结论与建议**（对 plan 的误差口径有直接影响）：
1. 该格的 `78.28 obs-cm` **不是防御方"没在跟"**，也**不是**防御方的失败——
   **P+FF 给出逐位相同的 78.65**。该格**任何策略都无法**达到 1.0 m 间距。
2. 因此该格的 **"0/20 零碰撞"是空信用**（零动作同样 0/20）：它度量的是**领车执行器**，
   不是防御方。**在 `_out_pole_final.txt` 与 RUNLOG 中把它当作正例的鲁棒性证据是不成立的**，
   须加脚注或用下限 gap_max 重跑。
3. **更普遍的风险**：`d=1.0` 这一整档的 `step-down *` 列（40.41 / 78.28 / 22.71）
   都在同一个执行器受限机制下产生。**plan 的误差口径（settled |gap − d_des|）
   在"指令摆动快于执行器带宽"时退化为对执行器的度量。**
   建议：评估脚本对每一格**同时**报告领车**实际**速度的均值/峰峰，
   或在 d 档位上改用 `gap_actual` 而非 `gap_setpoint` 作误差基准；
   至少须标注该列不可解释为防御方性能。

### D. `_probe_forced_win.py` 的**事实**成立，**推论**须收窄（§D 的回答）

**独立复现**（`_review_probe_round2.py` §S4，全新进程）：

```
S4  FORCED-WIN REPRODUCTION (independent re-run of _probe_forced_win.py)
  zero action, gap_max=1.2          coll  0/20
  P+FF,        gap_max=1.2          coll 20/20
  zero action, gap_max=0.20         coll  0/20
  P+FF,        gap_max=0.20         coll 20/20
```
**⇒ "零动作 0/20"这一关键事实确认**，且"把 gap_max 钉死为 0.20（无追赶过程）后仍 20/20"
也确认。**该探针的力学读法是对的**：崩的不是刹车能力，是追赶阶段积累的收距率。

**但 RUNLOG 的措辞"攻击惩罚的是激进收距而非刹车能力"需要一个补句**：
本轮实测表明**零动作在 11 族与 B 组上全部零碰撞**（§W1：0/100）。
⇒ 该事实的完整含义是"**攻击甚至不能惩罚'什么都不做'**"，
而**不是**"只有正例那种'节制型'策略能活"。
`plan §4` 退化表把"恒低速（远离领车）"标注为"**必须被 A3/A4 排除**"——
**实测显示 A3 排不掉它**（撞车率这一列上 A3 与零动作同值），
只有 A4 能排（零动作名义 err 83.22 ≫ 1.5×2.49）。**§4 的这条假设与 A3 的实际形式不符**。

### E. 正例常量是**在被验收的探针集上网格搜出来的** ⇒ 验收表不自证；但**不在刀尖上**

`_out_pole_v4.txt` 的 D2 网格 + `_tune_pole.py` 说明
`margin/safety/kp/a_max_lo` 是在**同一批 11 族的硬子集**上选出来的。
⇒ `_out_pole_final.txt` 的 0/660 **部分是拟合结果**，单独不足以支撑"通用抗扰"。

**缓解证据（两条，都在拟合集之外）**：
1. **D2 网格是一大片平台而非尖峰**：`margin ∈ {0.05, 0.08}` × `safety ∈ {0.60, 0.70, 0.80}`
   × `yth ∈ {0.20, 0.40}` 共 12 组**全部 0/140**，`nom_err` 2.24–2.40。
   ⇒ **不是刀尖上的拟合**。
2. **本轮的全部样本外测试均通过**：1500 随机波形 0 崩（§W2）、
   6 个家族外攻击类 0/20（§B-above）、B 组 0/100（§R1 表）。
   ⇒ 结论**存活**，但**承重的是这些样本外测试，不是那张 0/660 表**。
   **报告引用顺序建议调整**：先给样本外证据，`_out_pole_final.txt` 只作回归基线。

---

## 三、确认正确的事项（逐条实测）

| 项 | 结论 | 实测证据 |
|---|---|---|
| **验收可复现性** | ✅ **0/660 逐字节一致** | `uv run python results/20260925_phaseF_positive/positive_pole.py > /tmp/review_out_pole_final.txt`（02:35:56→02:38:24，exit 0）；`diff _out_pole_final.txt _review_rerun_pole_final.txt` → **空**。三档名义比值 0.88× / 1.00× / 1.00× 重现 |
| **验收表自洽性** | ✅ 11 族 × 3 间距 × 20 局 = 660，三档 TOTAL 均 0/220 | 重跑输出逐行相同 |
| **冻结纪律（F）** | ✅ **5 个冻结文件 sha256 全部未变** | `sha256sum car_following_sim.py follow_env.py overtake_env.py train_ppo.py train_ot.py` 与 `DATA_MANAGEMENT.md §8` 表**逐字节相同**；`git status --short` 对冻结文件干净 |
| **落点纪律（§5b）** | ✅ 正例全部脚本/输出**在仓库内且已入版本库** | `git ls-files results/20260925_phaseF_positive/` → 9 个文件全在（`positive_pole.py`、`_search`、`_v2/_v3/_v4`、`_probe_forced_win.py`、`_tune_pole.py` + 2 个 `_out_*.txt`）；plan §1.3 引用的 `_probe_positive_pole.py` 在 `results/20260924_phaseF_probe/`（亦已入版本库） |
| **防泄漏（G）** | ✅ 正例**只用 obs 4 维 + 冻结常量** | `positive_pole.py` 的 `__call__` 仅读 `obs[0..3]`、`fe.D_DES`、`fe.COLLISION_GAP`、`ACT`。**不读** `env._fn`、不读领车未来轨迹、不读 B 组参数。"预判"项用的是**自身**速度变化（因果可得）——且它是死代码（§R2） |
| **未被"恒低速"冒充** | ✅ 名义精度优于 P+FF，追赶时间相同 | §A(1) 表 |
| **未被固定阈值一行式冒充** | ✅ 固定阈值版本要么崩（0.20→26/220）要么毁名义精度（0.30→err 20.39） | §A(3) |
| **DR 打开时同样零碰撞** | ✅ 0/220（`domain_randomize=True`） | `_review_probe_round3.py` §S12 |
| **终止原因审计** | ✅ 正例全部 `timeout`，**无** `lost`/`offtrack` 混入 | §S9：11 族 × 20 局，pole 全为 `{'timeout': 20}` |

---

## 四、给 owner 的修订清单（按优先级）

1. **【拦截级】改写 H-A3**：采用 §一.R1 的 (a)–(e) 五条合取式，尤其
   **(d) 零动作对照行**（因 zero-action 已触顶 0/100，撞车率单独不可作判据）
   与 **(e) B 组跟踪质量门**（≤ 正例 1.5×）。同步改 `research/ASSUMPTIONS.md` 的 H-A3 行与 plan §2.3/§3。
2. **【拦截级】更正正例的机制陈述**：plan §4 的"预判-避让"、
   `positive_pole.py` docstring 的 `This is the "anticipate"`、
   RUNLOG 的"收距超限即全力刹车，否则退化为纯 P"，
   三处均须改为 §一.R2 给出的准确表述；`ythresh/yg/ymax/recover` 四个常量应标注为**无效参数**。
   （**代码无需改**，只改文档与注释；若要保持 README 一致，可把这 4 个常量删除并在 git 记录中说明。）
3. **【重要】`d=1.0` 档的执行器受限格**：在 `_out_pole_final.txt`/RUNLOG/论文中加脚注，
   说明该列度量的是领车执行器而非防御方（P+FF 同值 78.65）；
   评估脚本对每格增报领车**实际**速度均值，或改用 `gap_actual` 基准。
4. **【重要】B 组的措辞限定**：报告须写明正例对 B 组的零碰撞**不是**泛化证据
   （B2 的两个频率即正例验证频率）；并把正例的定位改为"A 组风格攻击的锚点"。
5. **【流程】把本轮 6 个评审探针入版本库**：`results/20260925_phaseF_positive/_review_probe_{triviality,round2,round3,round4,round5,round6_lite}.py`
   与 `_review_rerun_pole_final.txt` 当前为 untracked（`git status --short` 列出），
   按 §5b 必须随评审记录一并提交，否则该评审不可复现。
6. **【顺带观测，非本轮对象】**：`train_attacker.py` 落在**仓库根**，而其配套
   `eval_attacker.py` 在 `results/20260925_phaseF_attacker/`。§2 目录规范要求实验产物进
   `results/YYYYMMDD_<名>/`；建议归位或在 RUNLOG 说明理由。

---

## 五、附：本轮新增脚本与原始输出落点

| 脚本 | 输出 | 内容 |
|---|---|---|
| `_review_probe_triviality.py` | （stdout 见评审引用） | S1 支路活性 / S2 追赶能力 / S3 消融 |
| `_review_probe_round2.py` | 同上 | S4 forced-win 复现 / S5 d=1.0 退化 / S6 半支路消融 / S7 家族外攻击 / S8 B 组重叠 |
| `_review_probe_round3.py` | 同上 | S9 终止审计 / **S10 H-A3 基线（决定性）** / S11 替代式判别力 / S12 DR / S13 seed 敏感性 / S14·S15 |
| `_review_probe_round4.py` | 同上 | T1 err 列同值之谜 / **T2 wrap 伪影否证** / T3 基线种子波动 / T4 收距率分布 / T5 P+FF 等价性 |
| `_review_probe_round5.py` | 同上 | U1 死代码判定 / U2 真实收距率 / U3 死格机理 / U4·U5 支路占用 |
| `_review_probe_round6_lite.py` | 同上 | **W1 零动作在 B 组** / W2 1500 随机波形搜索 / W3 支路占用（11 族） |
| — | `_review_rerun_pole_final.txt` | 终验独立重跑原样输出（与 `_out_pole_final.txt` 逐字节相同） |

**说明**：以上脚本全部**只读**冻结文件，仅在运行期 monkeypatch `fe.D_DES`（`finally` 复位）。
**未修改任何冻结文件**；`git status --short` 对本轮的改动仅为上述新增 untracked 文件。
