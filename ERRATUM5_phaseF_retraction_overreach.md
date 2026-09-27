# ERRATUM 5 — Phase F Step 1 的撤回**过头了**：缺陷属实，但只有一半结论该废

日期：2026-09-27
性质：对 commit `a9007c7`（handoff 撤回）与 `ad1a5b6`（缺陷定位）的**部分更正**。
本次由接手 agent 在按 §8 复现交接文档时发现，**逐条实测确认**。
严重度：缺陷本身**属实**（不撤回）；但**撤回的范围超出了证据**——
把两个未被缺陷触及的结论（P+FF 半边、Step 0.5）一并作废，
且写反了一条机制（P+FF 的前馈项）。

---

## 0. 一句话

**`AttackerEnv` 的 `obs[1]≡0` 缺陷属实且必须修；但**

1. **P+FF 完全不受该缺陷影响**（它根本不读 `obs[1]`）⇒ **Step 1 的「0/20 vs P+FF」是干净测量**，
   而 H-A1' 要求**对两个防御方都 ≥60%**，故**判据失败这一结论本身成立**；
2. **Step 0.5 从未经过这条通路**（用的是另一个 env 子类，不 import `AttackerEnv`、不调 `_obs()`）
   ⇒ 「幅值轴可训」**无需复核**；
3. **v1 半边：缺陷确实污染了对照，但没污染结论**——修好观测后 v1 **更难打**（脚本攻击 18/20 → 0/20），
   学习到的攻击者仍是 **0/20**。

**⇒ 需要改的不是「0/20 作废」，而是「撤回时把范围划大了，并把 P+FF 的公式写错了」。**

---

## 1. 缺陷本身：**确认属实**（此部分不撤回）

`adversary_env.py:96` 在 `self.env.step()` **之前**调用 `self.env._obs()`；
而 `_obs()`（`follow_env.py:146-151`）的 `de = (e - self.prev_e)/DT` 依赖 `self.prev_e`，
后者**只在 `step()` 末尾**（`follow_env.py:182`）更新 ⇒ 旁路调用恒得 `de ≡ 0`。

**作者独立实测**（`_diag_obs_probe_safety.py`，已有产物）：

```
PLAIN env（正确用法：obs 取自 step() 返回）
    step 1: obs[1] = -0.056290
    step 2: obs[1] = +0.080000
旁路探针（step() 之前单独调 _obs()）
    step 1: side-probe obs[1] = +0.000000   step-return obs[1] = -0.056290
    step 2: side-probe obs[1] = +0.000000   step-return obs[1] = +0.080000
```

**且只有槽 1 被污染**（`_diag_defender_damage.py`，新入库）：

```
slot 0 e (gap err)   : max |true - side| = 0.000000   (identical)
slot 1 de (gap rate) : max |true - side| = 0.221657   <-- CORRUPTED
slot 2 v_f (own spd) : max |true - side| = 0.000000   (identical)
slot 3 e_lat         : max |true - side| = 0.000000   (identical)
```

**⇒ 缺陷、机制、影响面（`adversary_env.py:96` + `eval_attacker.py:120`）均如原文所述。**
**通则「`_obs()` 不可作 `step()` 之前的旁路探针」成立，保留。**

---

## 2. 更正一：**P+FF 不受缺陷影响**，原文的机制写反了

### R1-a：`baseline_action(use_ff=True)` **不读 `de`** ✅ **确认**

原文（`AGENT_HANDOFF.md:214`）称 P+FF 的 `a = (kp·e − de)/ACT` 里 `de=0` 使
**前馈项静默消失、退化为纯 P**。**冻源码是**（`follow_env.py:66-71`）：

```python
e, de = obs[0] * 0.5, obs[1] * 2.0   # un-normalize (see FollowEnv._obs)
# env computes v_cmd = v_f + de + ACT_GAIN*a, so:
#   use_ff=True  -> v_cmd = v_l + kp*e      (classic P + speed feedforward)
#   use_ff=False -> v_cmd = v_f + kp*e      (pure P; must fight the prior)
a = (kp * e - (0.0 if use_ff else de)) / ACT_GAIN
```

**`de` 只在 `use_ff=False`（纯 P）分支被读**；`use_ff=True` 的表达式是
`(kp*e − 0.0)/ACT`，**根本不碰 `obs[1]`**。原文引用的是**纯 P 的公式**，张冠李戴。

### R1-b：逐位实测 ✅ **确认**

```
P+FF  (use_ff=True) : 动作受缺陷影响的步数 =   0/106   （逐位相同）
pure P (use_ff=False): 动作受缺陷影响的步数 = 105/106   （Δa 均值 0.265）
v1 RL (frozen)      : 动作受缺陷影响的步数 =  16/106   （Δa 最大 2.000）
```

**更强的证据（逐回合逐位相同）**：把 P+FF 分别放在**污染 harness** 与**修正 harness** 下跑同一 seed，
**领头速度序列逐位相同**（`_diag_harness_robustness.py`）：

```
ep 0..4: len 1001 vs 1001   term timeout vs timeout   identical=True
=> all episodes bit-identical: True
```

**⇒ 对 P+FF 而言，「污染通路」与「正确通路」是同一次测量。
缺陷在 P+FF 身上没有制造任何差异，因此「0/20 vs P+FF」不受缺陷解释。**

### R1-c：**原文误判的来源**（可定位）✅ **确认**

`follow_env.py:67` 的**陈旧注释**写 `env computes v_cmd = v_f + de + ACT_GAIN*a`，
而冻结 env 实际是 `v_cmd = v_l + ACT_GAIN*a`（`follow_env.py:163`，无 `de` 项）。
**这条注释把 `de` 说成进入了动作式**，很可能就是「P+FF 读 de」这一误判的来源。
它是**冻结文件**，按铁律 2 不能改，**只能记录**（见 §5 处置 4）。

---

## 3. 更正二：**Step 0.5 与缺陷无关**，原文把它一并作废是错的

原文（`AGENT_HANDOFF.md:219`）称「Step 0.5 的『幅值轴可训』结论**也走同一通路**，同样需要复核」。

**实测**：Step 0.5 的训练与评估（`results/20260925_phaseF_step05/step05_train.py`、
`step05_gate2.py`）用的是**另一个 env 子类** `RandomWaveLeader(fe.FollowEnv)`，
**不 `import AttackerEnv`、不调用 `_obs()`**：

```
$ grep -n 'AttackerEnv\|_obs()' results/20260925_phaseF_step05/*.py
（无输出）
```

它训的是**防御方对脚本随机波形**，走的是标准 gym 接口（`model.learn` + `step()` 返回值），
**没有任何旁路探针**。

**⇒ Step 0.5 的两道门（std max 1.433 < 2.0；早停率 5%）不受缺陷影响，
「幅值轴可训」结论有效、无需复核。原文 §3.3 与 §7 的「两个结论均作废」对 Step 0.5 不成立。**

---

## 4. 更正三：**Step 1 的 0/20 有效**，缺陷只打掉**一半归因**

### R3-a：学习到的攻击者，修与不修都是 0/20 ✅ **确认**

同一 checkpoint、**不重训**，只把防御方改成消费**上一步 `step()` 返回的真实 obs**
（即原文 §4.1 选项 1 建议的修法之一）：

| 防御方 | 污染 harness | 修正 harness |
|---|---|---|
| 学到的攻击者 vs **P+FF** | **0/20** | **0/20** |
| 学到的攻击者 vs **v1** | **0/20** | **0/20** |

（`_out_defect_impact.txt`，新入库）

**⇒ H-A1' 的判据失败**在修正 harness 下**复现**。因 H-A1' 要求**对两个防御方都 ≥60%**，
**单凭 P+FF 半边**（一个缺陷无法触及的测量）**就已判定 FAIL**。

### R3-b：v1 的**对照**被缺陷夸大，但**结论**反而被加强 ✅ **确认**

脚本 step-down 攻击（review1 的 forced-win 探针）：

| | 污染 harness | 修正 harness |
|---|---|---|
| vs P+FF, step-down 0.1s | 20/20 | **20/20** |
| vs P+FF, step-down 0.2s | 20/20 | **20/20** |
| vs P+FF, step-down 0.4s | 20/20 | **20/20** |
| vs v1, step-down 0.1s | 18/20 | **0/20** |
| vs v1, step-down 0.2s | 18/20 | **0/20** |
| vs v1, step-down 0.4s | 13/20 | **4/20** |

**⇒ 对 P+FF，对照在两版 harness 下都成立**（攻击面完好）；
**对 v1，污染 harness 把 0/20 虚报成 18/20**——修正后**几乎打不动 v1**。

**关键推论（比原文更强）**：
学习到的攻击者**是在被削弱的 v1 上训练的，评估也在被削弱的 v1 上**，
即便如此仍是 **0/20**——**它连更弱的靶子都没打穿**。
修正只会让 v1 **更强**，**不可能翻盘**。
**⇒ 「攻击者 vs v1 失败」是有效结论，且修正后更牢固。**

### R3-c：**独立佐证——修正 harness 复原了预登记数字** ✅ **确认**

`research/ASSUMPTIONS.md`（**预注册文件**）§已知前提写：

> step-down 方波对 P+FF **20/20**、对 v1 **0/20**

**修正 harness 给出 P+FF 20/20、v1 0/20 —— 与预登记逐项吻合**；
**污染 harness 给出 v1 18/20 —— 与预登记矛盾**。
**⇒ 修正 harness 才是忠实的那一版**，由一份**独立的、先于本次争议的**数字佐证。

### R3-d：一个**更强**的 harness 被否证（防止过度更正）✅ **确认**

为排除「harness 选得不够强」的质疑，作者试了**比公平更强**的变体 `PostMove`
（防御方看到**领车已动、跟随车未动**的状态，信息严格更多）。
结果它给出 vs v1 **20/20**——**方向可疑**（v1 本应强于 P+FF）。
**公平性检验**（`_diag_postmove_fairness.py`，**无任何对抗内容**，领车恒速 0.50）：

```
v1 under corrected  harness: mean|e| =   7.990 obs-cm   term={'timeout': 10}
v1 under post-move  harness: mean|e| =  76.236 obs-cm   term={'collision': 10}
```

**⇒ `PostMove` 对 v1 是分布外输入**（领车动了而自车未动，v1 从未见过），
在**无攻击**时即 10/10 撞车 ⇒ **该变体无效，其 20/20 是 harness 假象，已弃用**。
**修正 harness（`Corrected`）是三者中唯一通过公平性检验的**。

---

## 5. 错误模式（本次三类，均可命名）

| # | 错误 | 性质 |
|---|---|---|
| 1 | 把**纯 P 的公式**（`kp*e − de`）当成 P+FF 的公式，据此宣称 P+FF 被削弱 | **未回对冻结源码**；很可能源自 `follow_env.py:67` 陈旧注释 |
| 2 | 由「控制走同一通路」推出「控制失效」，**未检验该通路对每个防御方是否等价** | **污染面按策略逐一下判**，而非整体作废 |
| 3 | 把**未经过该通路**的 Step 0.5 一并作废 | **影响面 grep 做了，但只 grep 了 `_obs()` 调用点，没查 Step 0.5 用的是哪个 env 类** |

**共同根因**：**「缺陷存在」被当成「缺陷影响一切」**——
与 `ERRATUM4` 的「每一次修正都停在了第一层」同型：
**这次停在了「通路被污染」，没往下走一层问「这条通路对谁等价」。**

> **可复用的检查项（建议入 `DATA_MANAGEMENT.md` §10）**：
> **发现测量通路缺陷时，须按【每个被测量的对象】逐一判定影响**，
> 不得由一个对象受影响推及全体；**判定依据须是实测的动作/输出差异，不是结构性推测。**

---

## 6. 仍然成立（**不受本轮影响的资产**）

| 资产 | 状态 |
|---|---|
| **`obs[1]≡0` 缺陷本身、根因、影响面 2 处** | ✅ 属实，须修 |
| **通则「`_obs()` 不可作 `step()` 前的旁路探针」** | ✅ 成立 |
| **H-A1' FAIL（0/20 对两个防御方）** | ✅ **有效**（P+FF 半边缺陷无法触及；v1 半边修正后复现） |
| **Step 0.5 双门 PASS（幅值轴可训）** | ✅ **有效，无需复核** |
| **正例锚点 0/660 碰撞** | ✅ 有效（不经 wrapper，原文已标） |
| **52 条退化策略判据攻击（55/55 fail）** | ✅ 有效（不经 wrapper，原文已标） |
| **D0 包络 / 平台速度上界 / Phase A** | ✅ 不受影响（原文本就声明不含冻结文件改动） |
| **冻结文件完整性** | ✅ `sha256 -c` 与 §8 表**逐字节一致** |

---

## 7. 处置

1. **`AGENT_HANDOFF.md` §3.3 重写**：缺陷保留；删去「P+FF 前馈项静默消失/退化为纯 P」；
   把「两个防御方都被削弱」改为「**P+FF 不受影响；v1 受影响**」；
   把「Step 0.5 同样需要复核」改为「**Step 0.5 与本案无关**」；
   把「**上一轮归因同时作废**」改为「**归因对 P+FF 有效；对 v1 应改述为『攻击面本身不存在』**」。
2. **`AGENT_HANDOFF.md` §4.1 / §7 / §8 同步**：待决项由「修+重跑 / 只修不跑」改为
   **「修 harness（必修）＋ 是否重跑 Step 1（可选，预期仍 0/20）」**；
   §7 的「Step 0.5 与 Step 1 结论均作废」改为「**Step 1 的 0/20 有效；Step 0.5 有效**」。
3. **`AGENT_HANDOFF.md` §6 陷阱表第 354 行**：删「P+FF 的前馈项」，
   改为「**v1 的第 2 槽**（P+FF 不受影响——它不读 `obs[1]`）」。
4. **记录（不改）冻结文件的陈旧注释**：`follow_env.py:67` 的
   `v_cmd = v_f + de + ACT_GAIN*a` 与实际 `v_cmd = v_l + ACT_GAIN*a`（`:163`）不符；
   **按铁律 2 不得改动**，在 `DATA_MANAGEMENT.md`「已知陷阱」中留一条
   「引用冻结文件行为前**以代码为准，勿信行内注释**」。
5. **`RUNLOG.md` 新增一条**（本轮实测）。
6. **重跑 Step 1 的判断**：**可选**。预期仍是 0/20，价值在于「干净存档」而非翻盘；
   若重跑，**必须同时修正 `eval_attacker.py:120`**（两处同修）。
7. **真正浮出的新问题（建议列为 Phase F 下一步）**：
   **「正确观测下，v1 的攻击面几乎不存在（0–4/20）」**——
   这比「是不是实现 bug」更根本，是「本任务从 leader 座位是否可解」的直接证据；
   **与被削弱的 v1 相比，正确 v1 直接把脚本攻击从 18/20 压到 0/20**。

---

## 8. 复现

```bash
cd /home/zy/car_rl/code0919
uv run python results/20260925_phaseF_attacker/_diag_defender_damage.py   # R1-a/R1-b
uv run python results/20260925_phaseF_attacker/_diag_defect_impact.py     # R3-a/R3-b
uv run python results/20260925_phaseF_attacker/_diag_harness_robustness.py # R1-b(逐位)/R3-d
uv run python results/20260925_phaseF_attacker/_diag_postmove_fairness.py  # R3-d 公平性
uv run python results/20260925_phaseF_attacker/_diag_obs_probe_safety.py   # 缺陷机制（已有）
```

产物（**均已入库**，F19；均在 `results/20260925_phaseF_attacker/` 下）：
`_out_defender_damage.txt`、`_out_defect_impact.txt`、
`_out_harness_robustness.txt`、`_out_postmove_fairness.txt`。

> **机器状态披露（F8）**：各跑前本机无并发训练；`make check` 先过（规则基线 10/10、零碰撞）。
> **冻结文件**：`follow_env.py` / `overtake_env.py` 的 `sha256` 与 `DATA_MANAGEMENT.md` §8 表
> **逐字节一致**（本次未改任何冻结文件）。
